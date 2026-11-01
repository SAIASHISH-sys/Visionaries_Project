import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from fpgraph import TARGET_CLASSES

CLASS_COLORS = {"wall": "#1f4e9c", "curtain_wall": "#17becf", "railing": "#8c564b", "door_single": "#d62728", "door_double": "#ff7f0e",
                "door_sliding": "#e377c2", "door_other": "#bcbd22", "window": "#2ca02c", "opening": "#7f7f7f", "stairs": "#9467bd",
                "elevator": "#111111", "other": "#d9d9d9"}


def room_polys(m):
    return [Polygon(r["polygon"]).buffer(0) for r in m["rooms"]]


def words(s):
    return set((s or "").upper().replace("/", " ").split())


def labels_agree(a, b):
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    return (a or "").upper().strip() == (b or "").upper().strip() or len(wa & wb) / max(len(wa | wb), 1) >= 0.5


def match_rooms(cv, ref, iou_min=0.5):
    A, B = room_polys(cv), room_polys(ref)
    iou = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if a.intersects(b):
                inter = a.intersection(b).area
                iou[i, j] = inter / max(a.area + b.area - inter, 1e-9)
    ri, ci = linear_sum_assignment(-iou) if iou.size else (np.array([], int), np.array([], int))
    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(ri, ci) if iou[i, j] >= iou_min]
    return pairs, iou


def match_openings(cv, ref, kind, radius=600.0):
    A = [o for o in cv["openings"] if o["kind"] == kind]
    B = [o for o in ref["openings"] if o["kind"] == kind]
    if not A or not B:
        return A, B, []
    pa, pb = np.array([o["position"] for o in A]), np.array([o["position"] for o in B])
    D = np.hypot(pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1])
    ri, ci = linear_sum_assignment(np.where(D <= radius, D, 1e9))
    return A, B, [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= radius]


def compare_manifests(cv, ref, iou_min=0.5):
    rows = []
    add = lambda group, metric, value: rows.append({"group": group, "metric": metric, "value": value})
    add("schema", "same top-level keys and order", list(cv.keys()) == list(ref.keys()))
    for k in ("rooms", "openings", "walls", "adjacency", "corridors", "vertical_circulation", "unassigned_labels"):
        add("counts", f"{k}: cv / reference", f"{len(cv[k])} / {len(ref[k])}")
    pairs, _ = match_rooms(cv, ref, iou_min)
    add("rooms", f"matched at IoU >= {iou_min}", f"{len(pairs)} of {len(cv['rooms'])} cv, {len(ref['rooms'])} reference")
    if pairs:
        add("rooms", "mean IoU of matched", round(float(np.mean([p[2] for p in pairs])), 3))
        err = [abs(cv["rooms"][i]["area_m2"] - ref["rooms"][j]["area_m2"]) / max(ref["rooms"][j]["area_m2"], 1e-9) * 100 for i, j, _ in pairs]
        add("rooms", "median area error %", round(float(np.median(err)), 2))
        agree = [labels_agree(cv["rooms"][i]["raw_label"], ref["rooms"][j]["raw_label"]) for i, j, _ in pairs if cv["rooms"][i]["raw_label"]]
        if agree:
            add("rooms", "label agreement among matched", round(float(np.mean(agree)), 3))
    for kind in ("door", "window"):
        A, B, m = match_openings(cv, ref, kind)
        add(kind + "s", "matched / cv / reference", f"{len(m)} / {len(A)} / {len(B)}")
        if m:
            add(kind + "s", "precision, recall", f"{len(m) / len(A):.3f}, {len(m) / len(B):.3f}")
            add(kind + "s", "mean abs width error mm", round(float(np.mean([abs(A[i]["width_mm"] - B[j]["width_mm"]) for i, j in m])), 1))
    cv_edges, ref_edges = matched_edges(cv, ref, pairs)
    tp = len(cv_edges & ref_edges)
    add("adjacency", "edges among matched rooms: cv / reference / shared", f"{len(cv_edges)} / {len(ref_edges)} / {tp}")
    add("adjacency", "edge F1", round(f1(tp, len(cv_edges), len(ref_edges)) or 0.0, 3))
    add("labels", "unassigned: cv / reference", f"{len(cv['unassigned_labels'])} / {len(ref['unassigned_labels'])}")
    return pd.DataFrame(rows)


def f1(tp, n_a, n_b):
    return 2 * tp / (n_a + n_b) if n_a + n_b else None


def matched_edges(cv, ref, pairs):
    to_ref = {cv["rooms"][i]["id"]: ref["rooms"][j]["id"] for i, j, _ in pairs}
    ref_ids = set(to_ref.values())
    cv_edges = {frozenset((to_ref[e["a"]], to_ref[e["b"]])) for e in cv["adjacency"] if e["a"] in to_ref and e["b"] in to_ref}
    ref_edges = {frozenset((e["a"], e["b"])) for e in ref["adjacency"] if e["a"] in ref_ids and e["b"] in ref_ids}
    return cv_edges, ref_edges


def matches(cv, ref, iou_min=0.5):
    pairs, _ = match_rooms(cv, ref, iou_min)
    cv_ids, ref_ids = [r["id"] for r in cv["rooms"]], [r["id"] for r in ref["rooms"]]
    rooms = [{"cv": cv_ids[i], "ref": ref_ids[j], "iou": round(iou, 3),
              "area_error_pct": round(100 * (cv["rooms"][i]["area_m2"] - ref["rooms"][j]["area_m2"]) / max(ref["rooms"][j]["area_m2"], 1e-9), 1)}
             for i, j, iou in pairs]
    out = {"rooms": {"pairs": rooms, "cv_only": [x for x in cv_ids if x not in {p["cv"] for p in rooms}],
                     "ref_only": [x for x in ref_ids if x not in {p["ref"] for p in rooms}]}, "openings": {}}
    for kind in ("door", "window"):
        A, B, m = match_openings(cv, ref, kind)
        paired = [{"cv": A[i]["id"], "ref": B[j]["id"]} for i, j in m]
        out["openings"][kind] = {"pairs": paired, "cv_only": [o["id"] for o in A if o["id"] not in {p["cv"] for p in paired}],
                                 "ref_only": [o["id"] for o in B if o["id"] not in {p["ref"] for p in paired}]}
    return out


def similarity(cv, ref, iou_min=0.5):
    pairs, _ = match_rooms(cv, ref, iou_min)
    parts = {"room_f1": f1(len(pairs), len(cv["rooms"]), len(ref["rooms"])),
             "room_mean_iou": float(np.mean([p[2] for p in pairs])) if pairs else (0.0 if cv["rooms"] and ref["rooms"] else None)}
    for kind in ("door", "window"):
        A, B, m = match_openings(cv, ref, kind)
        parts[f"{kind}_f1"] = f1(len(m), len(A), len(B))
    cv_edges, ref_edges = matched_edges(cv, ref, pairs)
    parts["adjacency_f1"] = f1(len(cv_edges & ref_edges), len(cv_edges), len(ref_edges))
    parts = {k: None if v is None else round(v, 4) for k, v in parts.items()}
    scored = [v for v in parts.values() if v is not None]
    return {"score": round(float(np.mean(scored)), 4) if scored else 0.0, "components": parts}


def plot_scene(Pw, scene, bbox=None, ax=None, max_prims=400000, lw=0.4):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    ax = ax or plt.subplots(figsize=(10, 10))[1]
    ends = Pw["pts"][:, [0, 3]]
    keep = np.ones(len(ends), dtype=bool)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        mid = Pw["pts"].mean(1)
        keep = (mid[:, 0] >= x0) & (mid[:, 0] <= x1) & (mid[:, 1] >= y0) & (mid[:, 1] <= y1)
    sel = np.where(keep)[0]
    if len(sel) > max_prims:
        sel = np.random.default_rng(0).choice(sel, max_prims, replace=False)
    for k, name in enumerate(TARGET_CLASSES):
        s = sel[scene["cls"][sel] == k]
        if len(s):
            ax.add_collection(LineCollection(ends[s], colors=CLASS_COLORS[name], linewidths=lw if name == "other" else lw * 2.5, label=name))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    return ax


def plot_rooms(cv, ref=None, bbox=None, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(10, 10))[1]
    for m, color, label in ((ref, "#d62728", "classical"), (cv, "#1f77b4", "cv_model")):
        if m is None:
            continue
        for k, r in enumerate(m["rooms"]):
            xy = np.array(r["polygon"])
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=0.8, label=label if k == 0 else None)
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    return ax


def plan_bbox(manifest, pad=3000.0):
    xy = np.vstack([np.array(r["polygon"]) for r in manifest["rooms"]])
    return (*(xy.min(0) - pad), *(xy.max(0) + pad))