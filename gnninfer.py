import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fpgraph
from gnnmodel import FloorGAT, ModelConfig
from gnndata import collate, make_batches, to_device
from gnneval import decode_instances, panoptic_counts, pq_from_counts, f1_from_conf

CLASSES = fpgraph.TARGET_CLASSES
OTHER = CLASSES.index("other")


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FloorGAT(ModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def read_threshold(path, default=0.5):
    p = Path(path)
    return float(json.loads(p.read_text())["threshold"]) if p.exists() else default


def graph_config(phase_a, mm_per_unit=1.0):
    return fpgraph.GraphConfig(**{**phase_a["graph"], "mm_per_unit": mm_per_unit})


def read_primitives(svg_path, cfg):
    return fpgraph.split_long_lines(fpgraph.parse_svg(svg_path, cfg.mm_per_unit), cfg.max_prim_len_mm)


def build_tiles(P, cfg, lut, stuff, src=1):
    tiles = []
    for t, (win, idx) in enumerate(fpgraph.tile_windows(fpgraph.midpoints(P), cfg)):
        g = fpgraph.build_graph(P, idx, win, cfg, lut, stuff)
        g.update({"idx": idx.astype(np.int64), "window": win, "src": src})
        tiles.append(g)
    return tiles


@torch.no_grad()
def predict_tiles(model, tiles, device, node_budget=12000, max_tiles=8, amp=False):
    sizes = np.array([len(t["y"]) for t in tiles])
    outs = [None] * len(tiles)
    for group in make_batches(sizes, np.arange(len(tiles)), node_budget, max_tiles):
        batch = to_device(collate([tiles[i] for i in group]), device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(batch["x"], batch["edge_index"], batch["edge_attr"])
        nl = out["node_logits"].float().cpu().numpy()
        el = out["edge_logits"].float().cpu().numpy()
        nptr, eptr = batch["node_ptr"].cpu().numpy(), batch["edge_ptr"].cpu().numpy()
        for b, i in enumerate(group):
            outs[i] = (nl[nptr[b]:nptr[b + 1]], el[eptr[b]:eptr[b + 1]])
    return outs


def merge_tiles(n_prims, tiles, outs):
    acc = np.zeros((n_prims, len(CLASSES)))
    cnt = np.zeros(n_prims)
    keys, vals = [], []
    for t, (nl, el) in zip(tiles, outs):
        idx = t["idx"]
        acc[idx] += nl
        cnt[idx] += 1
        ei = t["edge_index"].astype(np.int64)
        a, b = idx[ei[0]], idx[ei[1]]
        keys.append(np.minimum(a, b) * n_prims + np.maximum(a, b))
        vals.append(el.astype(np.float64))
    covered = cnt > 0
    logits = np.where(covered[:, None], acc / np.maximum(cnt, 1)[:, None], 0.0)
    if keys:
        key, val = np.concatenate(keys), np.concatenate(vals)
        uk, inv = np.unique(key, return_inverse=True)
        edge_logit = np.bincount(inv, weights=val) / np.bincount(inv)
        edge_index = np.stack([uk // n_prims, uk % n_prims])
    else:
        edge_logit, edge_index = np.zeros(0), np.zeros((2, 0), dtype=np.int64)
    return {"logits": logits, "covered": covered, "n_tiles_per_prim": cnt, "edge_index": edge_index, "edge_logit": edge_logit}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def decode_scene(P, merged, thr):
    n = len(P["ptype"])
    prob_all = softmax(merged["logits"])
    covered = merged["covered"]
    pred = prob_all.argmax(1)
    pred[~covered] = OTHER
    edge_prob = 1.0 / (1.0 + np.exp(-merged["edge_logit"]))
    seg, seg_cls, seg_len = decode_instances(pred, edge_prob, merged["edge_index"], P["length"].astype(np.float64), thr)
    cls = pred.copy()
    thing = seg >= 0
    cls[thing] = seg_cls[seg[thing]]
    prob = prob_all[np.arange(n), cls]
    prob[~covered] = 0.0
    return {"pred": pred, "cls": cls, "prob": prob, "seg": seg, "seg_cls": seg_cls, "seg_len": seg_len,
            "covered": covered, "thr": float(thr)}

def silver_arrays(P_gt, lut, stuff):
    y = lut[P_gt["raw"]].astype(np.int64)
    thing = ~stuff[y] & (y != OTHER) & (P_gt["inst"] >= 0)
    inst = np.full(len(y), -1, dtype=np.int64)
    if thing.any():
        key = P_gt["raw"].astype(np.int64)[thing] * 10_000_000 + P_gt["inst"].astype(np.int64)[thing]
        _, inv = np.unique(key, return_inverse=True)
        inst[thing] = inv
    return y, inst


def scene_from_silver(P_gt, lut, stuff):
    y, inst = silver_arrays(P_gt, lut, stuff)
    n_inst = int(inst.max()) + 1 if (inst >= 0).any() else 0
    seg_cls = np.zeros(n_inst, dtype=np.int64)
    seg_len = np.zeros(n_inst)
    has = inst >= 0
    seg_cls[inst[has]] = y[has]
    np.add.at(seg_len, inst[has], P_gt["length"].astype(np.float64)[has])
    return {"pred": y.copy(), "cls": y.copy(), "prob": np.ones(len(y)), "seg": inst, "seg_cls": seg_cls, "seg_len": seg_len,
            "covered": np.ones(len(y), dtype=bool), "thr": None}


def silver_agreement(P_gt, scene, lut, stuff):
    y, inst = silver_arrays(P_gt, lut, stuff)
    length = P_gt["length"].astype(np.float64)
    C = len(CLASSES)
    pred = scene["pred"]
    conf = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    conf_len = np.bincount(y * C + pred, weights=length, minlength=C * C).reshape(C, C)
    counts = panoptic_counts(y, inst, length, scene["seg"], scene["seg_cls"], scene["seg_len"])
    pq = pq_from_counts(counts)
    f1c, sup, acc = f1_from_conf(conf)
    f1l, supl, _ = f1_from_conf(conf_len)
    table = pd.DataFrame({"class": CLASSES, "silver_prims": sup.astype(int), "f1": f1c.round(3), "f1_length_weighted": f1l.round(3),
                          "pq": pq["pq_c"].round(3), "silver_instances": (counts[:, 0] + counts[:, 2]).astype(int),
                          "matched": counts[:, 0].astype(int), "extra_pred": counts[:, 1].astype(int)})
    summary = {"accuracy": float(acc), "wF1_length": float((f1l * supl).sum() / max(supl.sum(), 1e-9)),
               "macro_f1_present": float(f1c[sup > 0].mean()), "pq": pq["pq"], "rq": pq["rq"], "sq": pq["sq"]}
    return table, summary