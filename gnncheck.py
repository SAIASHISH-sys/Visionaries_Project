import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import fpgraph
from fpgraph import TARGET_CLASSES
from gnnmodel import FloorGAT, PanopticLoss, count_parameters
from gnndata import (collate, to_device, transform_pts, recompute_node_features, class_stats, make_batches,
                     node_length_mm, NUM_CLASSES, F_COS2T, F_SIN2T, F_CX, F_CY)

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
SERIES = ["#2a6fdb", "#e07b00", "#1a9850", "#8e44ad", "#c0392b"]


def row(status, check, finding, action=""):
    return {"status": status, "check": check, "finding": finding, "what to change": action}


def show(rows):
    df = pd.DataFrame(rows)
    try:
        from IPython.display import display
        with pd.option_context("display.max_colwidth", 200, "display.width", 250):
            display(df.style.hide(axis="index").map(
                lambda v: {"PASS": "color:#1a7f37;font-weight:600", "WARN": "color:#b35900;font-weight:600",
                           "FAIL": "color:#c0392b;font-weight:600"}.get(v, ""), subset=["status"]))
    except Exception:
        print(df.to_string(index=False))
    return df


def read_jsonl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def check_data(tiles, name="train"):
    cfg = fpgraph.GraphConfig()
    problems = [(i, p) for i, t in enumerate(tiles) for p in fpgraph.validate_tile(t, cfg)]
    rows = [row(PASS if not problems else FAIL, f"{name}: tile invariants",
                f"{len(tiles)} tiles, {len(problems)} problems" + (f", first: {problems[0]}" if problems else ""),
                "" if not problems else "rebuild shards with fpgraph; do not train on invalid tiles")]
    stats = class_stats(tiles)
    counts = stats["counts"]
    share = counts / max(counts.sum(), 1)
    absent = [c for c, n in zip(TARGET_CLASSES, counts) if n == 0]
    rows.append(row(INFO, f"{name}: class balance", ", ".join(f"{c} {s:.1%}" for c, s in zip(TARGET_CLASSES, share) if s > 0),
                    ""))
    rows.append(row(INFO if absent else PASS, f"{name}: absent classes", ", ".join(absent) or "none",
                    "absent classes get weight 0 and can never be learned from this set" if absent else ""))
    n_inst = np.array([int(t["inst"].max() + 1) if len(t["inst"]) else 0 for t in tiles])
    rows.append(row(PASS if (n_inst > 0).mean() > 0.5 else WARN, f"{name}: tiles with instances",
                    f"{(n_inst > 0).mean():.0%} of tiles have at least one thing instance",
                    "Head 2 needs positives; check the instance-id parsing" if (n_inst > 0).mean() <= 0.5 else ""))
    pos = sum(int(t["edge_y"].sum()) for t in tiles)
    masked = sum(int(t["edge_mask"].sum()) for t in tiles)
    rows.append(row(INFO, f"{name}: Head 2 edges", f"{pos:,} positive of {masked:,} masked edges, pos_weight {stats['pos_weight']:.2f}", ""))
    return rows


def check_augmentation(tiles, n=8):
    rows = []
    sel = tiles[:n]
    ident = max(float(np.abs(recompute_node_features(t["x"], t["pts"], t["ptype"]) - t["x"]).max()) for t in sel)
    rows.append(row(PASS if ident < 1e-3 else FAIL, "augmentation: identity reproduces Phase A features",
                    f"max abs difference {ident:.2e}",
                    "" if ident < 1e-3 else "recompute_node_features disagrees with fpgraph.build_graph; augmentation would inject label-free noise"))
    cyc = max(float(np.abs(transform_pts(transform_pts(transform_pts(transform_pts(t["pts"], 1, False), 1, False), 1, False), 1, False) - t["pts"]).max()) for t in sel)
    mir = max(float(np.abs(transform_pts(transform_pts(t["pts"], 0, True), 0, True) - t["pts"]).max()) for t in sel)
    rows.append(row(PASS if max(cyc, mir) < 0.05 else FAIL, "augmentation: 4 rotations and 2 mirrors return to start",
                    f"rotation drift {cyc:.3g} mm, mirror drift {mir:.3g} mm", "" if max(cyc, mir) < 0.05 else "transform_pts is not a rigid motion"))
    worst = 0.0
    for t in sel:
        if t["edge_index"].shape[1] == 0:
            continue
        i, j = t["edge_index"]
        for k, m in ((1, False), (2, True), (3, True)):
            p0, p1 = t["pts"].astype(np.float64).mean(1), transform_pts(t["pts"], k, m).astype(np.float64).mean(1)
            d0 = np.linalg.norm(p0[i] - p0[j], axis=1)
            d1 = np.linalg.norm(p1[i] - p1[j], axis=1)
            worst = max(worst, float(np.abs(d0 - d1).max()))
    rows.append(row(PASS if worst < 0.05 else FAIL, "augmentation: edge geometry preserved",
                    f"max change in neighbour distance {worst:.3g} mm",
                    "" if worst < 0.05 else "edge_attr would no longer match pts after augmentation"))
    batch = collate(sel, [(1, True)] * len(sel))
    plain = collate(sel)
    same = bool((batch["y"] == plain["y"]).all() and (batch["edge_y"] == plain["edge_y"]).all())
    cx = batch["x"][:, F_CX]
    rows.append(row(PASS if same else FAIL, "augmentation: labels untouched", f"node and edge labels identical: {same}", ""))
    span = (float(cx.min()), float(cx.max()))
    rows.append(row(PASS if -0.6 < span[0] and span[1] < 1.8 else WARN, "augmentation: positions stay in tile range",
                    f"cx range after rotate+mirror {span[0]:.2f} to {span[1]:.2f}", "positions drifting outside the tile mean the rotation centre is wrong"))
    return rows


def _batch(tiles, device):
    return to_device(collate(tiles), device)


def check_model(cfg, tiles, loss_fn, device, seed=0):
    torch.manual_seed(seed)
    model = FloorGAT(cfg).to(device)
    sel = [t for t in tiles if t["edge_index"].shape[1] > 0][:2]
    rows = [row(INFO, "model: parameters", f"{count_parameters(model):,} trainable parameters", "")]
    model.eval()
    b = _batch(sel, device)
    with torch.no_grad():
        out = model(b["x"], b["edge_index"], b["edge_attr"])
        _, parts = loss_fn(out, b["y"], b["edge_y"], b["edge_mask"], b["node_src"])
    finite = bool(torch.isfinite(out["node_logits"]).all() and torch.isfinite(out["edge_logits"]).all())
    rows.append(row(PASS if finite else FAIL, "model: forward is finite", f"shapes {tuple(out['node_logits'].shape)}, {tuple(out['edge_logits'].shape)}",
                    "" if finite else "NaN at initialisation: inspect scatter_softmax and feature scaling"))
    exp_node = float(np.log(cfg.num_classes))
    got_node = float(parts["node"])
    rows.append(row(PASS if abs(got_node - exp_node) / exp_node < 0.25 else WARN, "model: initial node loss is about ln(12)",
                    f"expected {exp_node:.3f}, got {got_node:.3f}",
                    "" if abs(got_node - exp_node) / exp_node < 0.25 else "head initialisation too large or class weights extreme"))
    em, ey = b["edge_mask"].float(), b["edge_y"]
    pos = float((ey * em).sum() / em.sum().clamp(min=1))
    exp_edge = float(np.log(2) * (float(loss_fn.pw) * pos + (1 - pos)))
    got_edge = float(parts["edge"])
    rows.append(row(PASS if abs(got_edge - exp_edge) / max(exp_edge, 1e-9) < 0.25 else WARN, "model: initial edge loss matches the label mix",
                    f"expected {exp_edge:.3f}, got {got_edge:.3f}", "check pos_weight and edge_mask"))
    one = _batch(sel[:1], device)
    n = one["x"].shape[0]
    perm = torch.randperm(n, device=device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=device)
    with torch.no_grad():
        o1 = model(one["x"], one["edge_index"], one["edge_attr"])
        o2 = model(one["x"][perm], inv[one["edge_index"]], one["edge_attr"])
        dn = float((o1["node_logits"][perm] - o2["node_logits"]).abs().max())
        de = float((o1["edge_logits"] - o2["edge_logits"]).abs().max())
        both = _batch(sel, device)
        ob = model(both["x"], both["edge_index"], both["edge_attr"])
        db = float((ob["node_logits"][:n] - o1["node_logits"]).abs().max())
    rows.append(row(PASS if max(dn, de) < 1e-3 else FAIL, "model: node-order equivariance", f"max difference node {dn:.2e}, edge {de:.2e}",
                    "" if max(dn, de) < 1e-3 else "the model depends on node order; message passing has an indexing bug"))
    rows.append(row(PASS if db < 1e-3 else FAIL, "model: tiles in a batch do not leak into each other", f"max difference {db:.2e}",
                    "" if db < 1e-3 else "batch offsets are wrong in collate or the layer mixes graphs"))
    model.train()
    b = _batch(sel, device)
    model.zero_grad()
    out = model(b["x"], b["edge_index"], b["edge_attr"])
    loss, _ = loss_fn(out, b["y"], b["edge_y"], b["edge_mask"], b["node_src"])
    loss.backward()
    dead = [nm for nm, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
    bad = [nm for nm, p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    rows.append(row(PASS if not dead else WARN, "model: every parameter receives gradient",
                    f"{len(dead)} without gradient" + (f": {dead[:4]}" if dead else ""),
                    "" if not dead else "unused layers; check head wiring (zero grads at step 0 can be normal for the zero-init head only)"))
    rows.append(row(PASS if not bad else FAIL, "model: gradients are finite", f"{len(bad)} parameters with NaN or inf", ""))
    return rows


def _pick(tiles, n, max_nodes=2500):
    cand = [t for t in tiles if 0 < len(t["y"]) <= max_nodes and t["edge_index"].shape[1] > 0]
    cand.sort(key=lambda t: -int(t["inst"].max() + 1 if len(t["inst"]) else 0))
    return cand[:n]


def overfit_test(cfg, tiles, loss_fn, device, steps=300, lr=2e-3, n_tiles=4, seed=0):
    torch.manual_seed(seed)
    model = FloorGAT(cfg).to(device)
    b = _batch(_pick(tiles, n_tiles), device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for s in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        out = model(b["x"], b["edge_index"], b["edge_attr"])
        loss, parts = loss_fn(out, b["y"], b["edge_y"], b["edge_mask"], b["node_src"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 10 == 0 or s == 1:
            with torch.no_grad():
                acc = float((out["node_logits"].argmax(1) == b["y"]).float().mean())
                p = out["edge_logits"] > 0
                ey, em = b["edge_y"] > 0.5, b["edge_mask"]
                tp, fp, fn = [float(v.sum()) for v in (p & ey & em, p & ~ey & em, ~p & ey & em)]
            hist.append({"step": s, "loss": float(loss), "node_acc": acc, "edge_f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    h = pd.DataFrame(hist)
    last = h.iloc[-1]
    drop = 1 - last["loss"] / h.iloc[0]["loss"]
    rows = [
        row(PASS if last["node_acc"] >= 0.97 else (WARN if last["node_acc"] >= 0.9 else FAIL), "overfit 4 tiles: node accuracy",
            f"{last['node_acc']:.3f} after {steps} steps (loss down {drop:.0%})",
            "" if last["node_acc"] >= 0.97 else "a correct model memorises 4 tiles; try more steps, then suspect a bug in labels, features or the loss"),
        row(PASS if last["edge_f1"] >= 0.8 else (WARN if last["edge_f1"] >= 0.5 else FAIL), "overfit 4 tiles: Head 2 edge F1",
            f"{last['edge_f1']:.3f}", "" if last["edge_f1"] >= 0.8 else "Head 2 is not learning: check edge_mask, edge_y and pos_weight"),
    ]
    return h, rows


def lr_sweep(cfg, tiles, loss_fn, device, lrs=(1e-4, 3e-4, 1e-3, 3e-3), steps=80, seed=0):
    sel = _pick(tiles, 6)
    batches = [_batch(sel[:3], device), _batch(sel[3:6] or sel[:3], device)]
    res = []
    for lr in lrs:
        torch.manual_seed(seed)
        model = FloorGAT(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        for s in range(steps):
            b = batches[s % 2]
            opt.zero_grad()
            out = model(b["x"], b["edge_index"], b["edge_attr"])
            loss, _ = loss_fn(out, b["y"], b["edge_y"], b["edge_mask"], b["node_src"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        res.append({"lr": lr, "start": losses[0], "end": float(np.mean(losses[-10:])), "finite": bool(np.isfinite(losses).all())})
    df = pd.DataFrame(res)
    ok = df[df["finite"]]
    best = float(ok.loc[ok["end"].idxmin(), "lr"]) if len(ok) else float("nan")
    return df, best


def dry_run(tiles, cfg, loss_fn, device, budgets=(6000, 12000, 20000), max_tiles=8, steps=4, amp=False, seed=0):
    rng = np.random.default_rng(seed)
    sizes = np.array([len(t["y"]) for t in tiles])
    cases = [("largest single tile", [int(sizes.argmax())])]
    for bud in budgets:
        bs = make_batches(sizes, np.arange(len(tiles)), bud, max_tiles, rng)
        cases.append((f"budget {bud}", max(bs, key=lambda g: sizes[g].sum())))
    res = []
    for name, group in cases:
        torch.manual_seed(seed)
        model = FloorGAT(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
        b = _batch([tiles[i] for i in group], device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for s in range(steps + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                out = model(b["x"], b["edge_index"], b["edge_attr"])
            loss, _ = loss_fn(out, b["y"], b["edge_y"], b["edge_mask"], b["node_src"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if s > 0:
                times.append(time.time() - t0)
        sec = float(np.mean(times))
        res.append({"case": name, "tiles": len(group), "nodes": int(b["x"].shape[0]), "edges": int(b["edge_index"].shape[1]),
                    "peak_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else float("nan"),
                    "sec_per_step": sec, "tiles_per_s": len(group) / sec})
    return pd.DataFrame(res)


def project_time(df, case, n_tiles, epochs, world=1):
    r = df[df["case"] == case].iloc[0]
    per_epoch_min = n_tiles / r["tiles_per_s"] / world / 60
    return {"case": case, "measured_tiles_per_s_per_gpu": float(r["tiles_per_s"]), "projected_min_per_epoch": per_epoch_min,
            "projected_hours_total": per_epoch_min * epochs / 60}


def analyze_run(run_dir, reference_wf1=0.85):
    run = Path(run_dir)
    cfg = json.loads((run / "run_config.json").read_text())
    args = cfg["args"]
    log, steps = read_jsonl(run / "metrics.jsonl"), read_jsonl(run / "steps.jsonl")
    ep = [r for r in log if r["event"] == "epoch"]
    val = [r for r in log if r["event"] == "val" and r["set"] == args["val_set"]]
    if not ep:
        return show([row(FAIL, "log", "no epoch finished", "read the training output for the traceback")]), None
    rows = []
    loss_ep = np.array([r["loss"] for r in ep])
    skipped, tot_steps = sum(r["skipped"] for r in ep), sum(r["steps"] for r in ep)
    bad = (not np.isfinite(loss_ep).all()) or skipped / max(tot_steps, 1) > 0.005
    rows.append(row(FAIL if not np.isfinite(loss_ep).all() else (WARN if bad else PASS), "numerics",
                    f"{skipped} of {tot_steps} steps skipped for non-finite gradients",
                    "lower --lr, keep --grad-clip 1, or drop --amp" if bad else ""))
    sl = np.array([s["loss"] for s in steps if np.isfinite(s["loss"])])
    if len(sl) >= 40:
        k = max(len(sl) // 5, 10)
        drop = 1 - sl[-k:].mean() / sl[:k].mean()
        rows.append(row(PASS if drop >= 0.3 else (WARN if drop >= 0.1 else FAIL), "learning",
                        f"train loss fell {drop:.0%} from the first to the last fifth of logged steps",
                        "" if drop >= 0.3 else "barely learning: run the overfit test, then try lr 3e-4 or 3e-3, and check class weights"))
    elif len(ep) >= 2:
        drop = 1 - loss_ep[-1] / loss_ep[0]
        rows.append(row(PASS if drop >= 0.2 else WARN, "learning", f"epoch loss fell {drop:.0%}", "" if drop >= 0.2 else "train longer or raise lr"))
    last = ep[-1]
    rows.append(row(INFO, "loss composition", f"node {last['loss_node']:.3f}, edge {last['loss_edge']:.3f} x lambda {args['lam']}",
                    "if the edge term dominates and node F1 lags, lower --lam to 1" if args["lam"] * last["loss_edge"] > 3 * last["loss_node"] else ""))
    ratio = np.median([r["grad_max"] / max(r["grad_mean"], 1e-9) for r in ep])
    rows.append(row(WARN if ratio > 10 else PASS, "gradient spikes", f"median max/mean gradient norm {ratio:.1f}",
                    "lower --lr or --grad-clip" if ratio > 10 else ""))
    clipped = np.mean([r["grad_mean"] > 3 * args["grad_clip"] for r in ep])
    rows.append(row(WARN if clipped > 0.8 else PASS, "clipping", f"mean gradient norm above 3x clip in {clipped:.0%} of epochs",
                    "clip is active almost always: lower --lr" if clipped > 0.8 else ""))
    best_val = None
    if val:
        sc = np.array([v["summary"]["pq"] if v["summary"]["n_gt_instances"] > 0 else v["summary"]["wf1"] for v in val])
        best_i = int(np.nanargmax(sc))
        best_val, final = val[best_i]["summary"], val[-1]["summary"]
        if len(val) == 1:
            rows.append(row(INFO, "validation trend", "only one validation so far", ""))
        elif best_i == len(val) - 1:
            rows.append(row(WARN if len(ep) >= args["epochs"] else INFO, "validation trend",
                            f"best score {sc[best_i]:.3f} is the latest evaluation",
                            "still improving: raise --epochs or resume" if len(ep) >= args["epochs"] else ""))
        elif sc[-1] < sc[best_i] - 0.03:
            rows.append(row(WARN, "validation trend", f"best {sc[best_i]:.3f} at epoch {val[best_i]['epoch']}, latest {sc[-1]:.3f}",
                            "val fell after its peak: more augmentation, --dropout 0.1, or a smaller --lr-step"))
        else:
            rows.append(row(PASS, "validation trend", f"best {sc[best_i]:.3f} at epoch {val[best_i]['epoch']}, latest {sc[-1]:.3f}", ""))
        gap = final["loss"] / max(last["loss"], 1e-9)
        rows.append(row(WARN if gap > 1.6 else PASS, "train/val loss gap",
                        f"val loss {final['loss']:.3f} vs train {last['loss']:.3f} (ratio {gap:.2f}; train is augmented, val is not)",
                        "overfitting: more data, --dropout 0.1, stronger augmentation" if gap > 1.6 else ""))
        weak = [c for c in TARGET_CLASSES if final["support"][c] > 0 and final["f1_per_class"][c] < 0.5]
        rows.append(row(WARN if weak else PASS, "per-class F1", "below 0.5: " + (", ".join(f"{c} {final['f1_per_class'][c]:.2f}" for c in weak) or "none"),
                        "raise those class weights or add --focal-gamma 2; confirm the class exists in train" if weak else ""))
        rows.append(row(WARN if min(final["edge_precision"], final["edge_recall"]) < 0.6 else PASS, "Head 2",
                        f"precision {final['edge_precision']:.2f}, recall {final['edge_recall']:.2f} at threshold {final['thr']}",
                        "low recall splits instances; low precision merges neighbours: sweep the threshold, then adjust --lam"
                        if min(final["edge_precision"], final["edge_recall"]) < 0.6 else ""))
        sw = pd.DataFrame(final["sweep"])
        if sw["pq"].notna().any():
            top = sw.loc[sw["pq"].idxmax()]
            cur = sw.loc[(sw["thr"] - final["thr"]).abs().idxmin()]
            rows.append(row(INFO, "threshold sweep", f"best PQ {top['pq']:.3f} at {top['thr']}, current {cur['pq']:.3f} at {cur['thr']}",
                            f"use threshold {top['thr']}" if top["pq"] - cur["pq"] > 0.01 else ""))
        conf = np.array(final["conf"])
        off = conf - np.diag(np.diag(conf))
        flat = np.argsort(off.ravel())[::-1][:3]
        rows.append(row(INFO, "top confusions", "; ".join(f"{TARGET_CLASSES[i // NUM_CLASSES]} read as {TARGET_CLASSES[i % NUM_CLASSES]} ({int(off.ravel()[i]):,})"
                                                        for i in flat if off.ravel()[i] > 0) or "none", ""))
        rows.append(row(INFO, "reference", f"length-weighted F1 {final['wf1']:.3f} vs GAT-CADNet 0.850 (published, full FloorPlanCAD, different class set; not directly comparable)", ""))
    else:
        rows.append(row(WARN, "validation", "no validation entry yet", "lower --eval-every"))
    rows.append(row(INFO, "speed", f"{last['tiles_per_s']:.1f} tiles/s, {last['seconds']:.0f} s per epoch, peak GPU {last['gpu_peak_gb']:.1f} GB", ""))
    return show(rows), best_val


def plot_run(run_dir):
    run = Path(run_dir)
    log, steps = read_jsonl(run / "metrics.jsonl"), read_jsonl(run / "steps.jsonl")
    args = json.loads((run / "run_config.json").read_text())["args"]
    val = [r for r in log if r["event"] == "val" and r["set"] == args["val_set"]]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    st = pd.DataFrame(steps)
    if len(st):
        ax[0, 0].plot(st["step"], st["node"], color=SERIES[0], label="node CE")
        ax[0, 0].plot(st["step"], st["edge"], color=SERIES[1], label="edge BCE")
        ax[0, 0].set_yscale("log")
        ax[0, 1].plot(st["step"], st["grad_norm"], color=SERIES[2], label="grad norm")
        ax[0, 1].axhline(args["grad_clip"], color="#777", lw=1, ls="--", label="clip")
        ax[0, 1].set_yscale("log")
        ax[1, 1].plot(st["step"], st["lr"], color=SERIES[3], label="lr")
    if val:
        e = [v["epoch"] + 1 for v in val]
        for key, c in (("wf1", SERIES[0]), ("f1_macro", SERIES[1]), ("pq", SERIES[2]), ("edge_f1", SERIES[3])):
            ax[1, 0].plot(e, [v["summary"][key] for v in val], marker="o", color=c, label=key)
        ax[1, 0].set_ylim(0, 1)
    titles = [("training loss", "step"), ("gradient norm", "step"), ("validation", "epoch"), ("learning rate", "step")]
    for a, (t, xl) in zip(ax.ravel(), titles):
        a.set_title(t, loc="left", fontsize=11)
        a.set_xlabel(xl)
        a.grid(alpha=0.2)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False)
    fig.tight_layout()
    return fig


def draw_tile(ax, tile, labels, title):
    pts = tile["pts"].astype(np.float64)
    segs = np.concatenate([np.stack([pts[:, k], pts[:, k + 1]], 1) for k in range(3)])
    cmap = plt.get_cmap("tab20")
    cols = np.array([(0.85, 0.85, 0.85, 1.0) if c == TARGET_CLASSES.index("other") else cmap(c) for c in range(NUM_CLASSES)])
    ax.add_collection(LineCollection(segs, colors=np.tile(cols[labels], (3, 1)), linewidths=0.9))
    ax.set_xlim(pts[..., 0].min(), pts[..., 0].max())
    ax.set_ylim(pts[..., 1].min(), pts[..., 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)


@torch.no_grad()
def show_prediction(model, tile, device, title="validation tile"):
    model.eval()
    b = to_device(collate([tile]), device)
    pred = model(b["x"], b["edge_index"], b["edge_attr"])["node_logits"].argmax(1).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_tile(ax[0], tile, tile["y"].astype(int), f"{title}: ground truth")
    acc = float((pred == tile["y"]).mean())
    draw_tile(ax[1], tile, pred, f"prediction (primitive accuracy {acc:.2f})")
    cmap = plt.get_cmap("tab20")
    handles = [plt.Line2D([0], [0], color=(0.85, 0.85, 0.85) if c == "other" else cmap(i), lw=3, label=c) for i, c in enumerate(TARGET_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
