import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fpgraph import TARGET_CLASSES, STUFF_TARGETS
from gnndata import node_length_mm, to_device

NUM_CLASSES = len(TARGET_CLASSES)
IS_THING = np.array([c not in STUFF_TARGETS and c != "other" for c in TARGET_CLASSES])
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def decode_instances(pred, prob, edge_index, length, thr):
    n = len(pred)
    thing = IS_THING[pred]
    keep = (prob > thr) & thing[edge_index[0]] & thing[edge_index[1]]
    ei = edge_index[:, keep]
    graph = coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    seg = np.full(n, -1, dtype=np.int64)
    if not thing.any():
        return seg, np.zeros(0, dtype=np.int64), np.zeros(0)
    _, inv = np.unique(comp[thing], return_inverse=True)
    votes = np.zeros((inv.max() + 1, NUM_CLASSES))
    np.add.at(votes, (inv, pred[thing]), length[thing])
    seg[thing] = inv
    return seg, votes.argmax(1), votes.sum(1)


def panoptic_counts(y, inst, length, seg, seg_cls, seg_len):
    out = np.zeros((NUM_CLASSES, 4))
    has_gt = inst >= 0
    if has_gt.any():
        gt_ids, gi = np.unique(inst[has_gt], return_inverse=True)
        n_gt = len(gt_ids)
        gt_len = np.bincount(gi, weights=length[has_gt], minlength=n_gt)
        gt_cls = np.zeros(n_gt, dtype=np.int64)
        gt_cls[gi] = y[has_gt]
    else:
        n_gt, gt_cls = 0, np.zeros(0, dtype=np.int64)
    m = len(seg_cls)
    tp = np.zeros(NUM_CLASSES)
    iou_sum = np.zeros(NUM_CLASSES)
    both = has_gt & (seg >= 0)
    if both.any():
        full_gi = np.full(len(y), -1, dtype=np.int64)
        full_gi[has_gt] = gi
        key = full_gi[both] * m + seg[both]
        u, inv = np.unique(key, return_inverse=True)
        inter = np.bincount(inv, weights=length[both])
        g_idx, p_idx = u // m, u % m
        iou = inter / (gt_len[g_idx] + seg_len[p_idx] - inter)
        ok = (iou > 0.5) & (gt_cls[g_idx] == seg_cls[p_idx])
        np.add.at(tp, gt_cls[g_idx][ok], 1)
        np.add.at(iou_sum, gt_cls[g_idx][ok], iou[ok])
    n_pred_c = np.bincount(seg_cls, minlength=NUM_CLASSES)
    n_gt_c = np.bincount(gt_cls, minlength=NUM_CLASSES)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = tp, n_pred_c - tp, n_gt_c - tp, iou_sum
    return out


class Evaluator:
    def __init__(self, thresholds=DEFAULT_THRESHOLDS, num_classes=NUM_CLASSES):
        self.thr, self.C = list(thresholds), num_classes
        T = len(self.thr)
        self.conf = np.zeros((self.C, self.C))
        self.conf_len = np.zeros((self.C, self.C))
        self.pq = np.zeros((T, self.C, 4))
        self.edge = np.zeros((T, 3))
        self.loss = np.zeros(4)

    def update(self, batch, out, parts):
        C = self.C
        pred = out["node_logits"].float().argmax(1).cpu().numpy()
        y = batch["y"].cpu().numpy()
        length = node_length_mm(batch["x"].cpu().numpy())
        inst = batch["inst"].cpu().numpy()
        ei = batch["edge_index"].cpu().numpy()
        prob = torch.sigmoid(out["edge_logits"].float()).cpu().numpy()
        ey = batch["edge_y"].cpu().numpy() > 0.5
        em = batch["edge_mask"].cpu().numpy()
        nptr, eptr = batch["node_ptr"].cpu().numpy(), batch["edge_ptr"].cpu().numpy()
        self.conf += np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
        self.conf_len += np.bincount(y * C + pred, weights=length, minlength=C * C).reshape(C, C)
        for t, thr in enumerate(self.thr):
            p = prob > thr
            self.edge[t] += [(p & ey & em).sum(), (p & ~ey & em).sum(), (~p & ey & em).sum()]
            for b in range(len(nptr) - 1):
                a, z, c, d = nptr[b], nptr[b + 1], eptr[b], eptr[b + 1]
                seg, cls, slen = decode_instances(pred[a:z], prob[c:d], ei[:, c:d] - a, length[a:z], thr)
                self.pq[t] += panoptic_counts(y[a:z], inst[a:z], length[a:z], seg, cls, slen)
        self.loss += [float(parts["total"]), float(parts["node"]), float(parts["edge"]), 1.0]

    def vector(self):
        return np.concatenate([self.conf.ravel(), self.conf_len.ravel(), self.pq.ravel(), self.edge.ravel(), self.loss])

    def load(self, vec):
        C, T = self.C, len(self.thr)
        sizes = [C * C, C * C, T * C * 4, T * 3, 4]
        parts = np.split(np.asarray(vec, dtype=np.float64), np.cumsum(sizes)[:-1])
        self.conf, self.conf_len = parts[0].reshape(C, C), parts[1].reshape(C, C)
        self.pq, self.edge, self.loss = parts[2].reshape(T, C, 4), parts[3].reshape(T, 3), parts[4]

    def summarize(self, thr=0.7):
        t = int(np.argmin(np.abs(np.array(self.thr) - thr)))
        f1c, sup, acc = f1_from_conf(self.conf)
        f1l, supl, _ = f1_from_conf(self.conf_len)
        pq = pq_from_counts(self.pq[t])
        tp, fp, fn = self.edge[t]
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        names = TARGET_CLASSES
        n = max(self.loss[3], 1)
        return {
            "thr": self.thr[t], "acc": acc,
            "f1_macro": float(f1c[sup > 0].mean()) if (sup > 0).any() else 0.0,
            "f1_weighted": float((f1c * sup).sum() / max(sup.sum(), 1)),
            "wf1": float((f1l * supl).sum() / max(supl.sum(), 1e-9)),
            "f1_per_class": {c: float(v) for c, v in zip(names, f1c)},
            "support": {c: int(v) for c, v in zip(names, sup)},
            "pq": pq["pq"], "rq": pq["rq"], "sq": pq["sq"], "n_gt_instances": pq["n_gt"],
            "pq_per_class": {c: float(v) for c, v in zip(names, pq["pq_c"])},
            "edge_precision": float(prec), "edge_recall": float(rec),
            "edge_f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "loss": float(self.loss[0] / n), "loss_node": float(self.loss[1] / n), "loss_edge": float(self.loss[2] / n),
            "conf": self.conf.tolist(),
            "sweep": [{"thr": th, "pq": pq_from_counts(self.pq[i])["pq"],
                       "edge_f1": float(2 * self.edge[i, 0] / max(2 * self.edge[i, 0] + self.edge[i, 1] + self.edge[i, 2], 1))}
                      for i, th in enumerate(self.thr)],
        }


def f1_from_conf(conf):
    tp = np.diag(conf)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return f1, conf.sum(1), float(tp.sum() / max(conf.sum(), 1e-12))


def pq_from_counts(c):
    tp, fp, fn, iou = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    den = tp + 0.5 * fp + 0.5 * fn
    pq_c = np.where(den > 0, iou / np.maximum(den, 1e-12), 0.0)
    present = IS_THING & (tp + fn > 0)
    rq_c = np.where(den > 0, tp / np.maximum(den, 1e-12), 0.0)
    sq_c = np.where(tp > 0, iou / np.maximum(tp, 1e-12), 0.0)
    mean = lambda v: float(v[present].mean()) if present.any() else float("nan")
    return {"pq": mean(pq_c), "rq": mean(rq_c), "sq": mean(sq_c), "pq_c": pq_c, "n_gt": int((tp + fn)[IS_THING].sum())}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, thresholds=DEFAULT_THRESHOLDS, amp=False):
    model.eval()
    ev = Evaluator(thresholds)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch["x"], batch["edge_index"], batch["edge_attr"])
        _, parts = loss_fn(out, batch["y"], batch["edge_y"], batch["edge_mask"], batch["node_src"])
        ev.update(batch, out, parts)
    return ev


def evaluate_checkpoint(ckpt_path, run_config_path, root, split, src, thresholds=DEFAULT_THRESHOLDS, device=None,
                        node_budget=12000, max_tiles=8, amp=False, max_shards=None):
    from torch.utils.data import DataLoader
    from gnnmodel import FloorGAT, ModelConfig, PanopticLoss
    from gnndata import load_tiles, make_batches, BatchDataset
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FloorGAT(ModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    rc = json.loads(Path(run_config_path).read_text())
    loss_fn = PanopticLoss(rc["class_weights"], rc["edge_pos_weight"]).to(device)
    tiles = load_tiles(root, split, src, max_shards)
    sizes = np.array([len(t["y"]) for t in tiles])
    batches = make_batches(sizes, np.arange(len(tiles)), node_budget, max_tiles)
    loader = DataLoader(BatchDataset(tiles, batches), batch_size=None, shuffle=False)
    ev = evaluate(model, loader, loss_fn, device, thresholds, amp=amp and device.type == "cuda")
    return ev, len(tiles)
