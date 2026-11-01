import re
import io
import json
import math
import time
import random
import zipfile
import hashlib
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from pathlib import Path as FsPath

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from svgelements import SVG, Path, Circle, Ellipse, Arc, Line, Move, Close, CubicBezier, QuadraticBezier, Shape, Text

PRIM_TYPES = ["line", "arc", "circle", "ellipse", "curve"]

RAW_CLASS_NAMES = [
    "wall", "curtain wall",
    "single door", "double door", "sliding door", "folding door", "revolving door", "rolling door",
    "window", "bay window", "blind window", "opening symbol",
    "sofa", "bed", "chair", "table", "TV cabinet", "wardrobe", "cabinet",
    "gas stove", "sink", "refrigerator", "airconditioner", "bath", "bath tub", "washing machine",
    "squat toilet", "urinal", "toilet",
    "stairs", "elevator", "escalator",
    "railing", "row chairs", "parking spot",
]

TARGET_CLASSES = [
    "wall", "curtain_wall", "railing",
    "door_single", "door_double", "door_sliding", "door_other",
    "window", "opening", "stairs", "elevator", "other",
]

STUFF_TARGETS = {"wall", "curtain_wall", "railing"}

HARMONIZATION = {
    "single door": "door_single", "double door": "door_double", "sliding door": "door_sliding",
    "folding door": "door_other", "revolving door": "door_other", "rolling door": "door_other",
    "window": "window", "bay window": "window", "blind window": "window", "opening symbol": "opening",
    "stairs": "stairs", "elevator": "elevator",
    "wall": "wall", "curtain wall": "curtain_wall", "railing": "railing",
}


@dataclass
class GraphConfig:
    mm_per_unit: float = 1.0
    max_prim_len_mm: float = 2000.0
    tile_size_mm: float = 10000.0
    tile_overlap_mm: float = 1000.0
    max_nodes: int = 4096
    min_nodes: int = 8
    radius_mm: float = 300.0
    max_neighbors: int = 30
    collinear_angle_deg: float = 2.0
    collinear_offset_mm: float = 20.0
    collinear_gap_mm: float = 1000.0
    touch_tol_mm: float = 5.0


def build_label_map(raw_names=RAW_CLASS_NAMES, harmonization=HARMONIZATION, targets=TARGET_CLASSES):
    other = targets.index("other")
    lut = np.full(len(raw_names) + 1, other, dtype=np.int16)
    for i, name in enumerate(raw_names):
        lut[i] = targets.index(harmonization.get(name, "other"))
    return lut


def stuff_mask(targets=TARGET_CLASSES):
    return np.array([t in STUFF_TARGETS for t in targets])


TAG_RE = re.compile(rb"<(path|circle|ellipse|line|polyline|polygon|rect)\b([^>]*)>", re.S)
SEM_RE = re.compile(rb'semantic[-_]?id\s*=\s*"(-?\d+)"', re.I)
INST_RE = re.compile(rb'instance[-_]?id\s*=\s*"(-?\d+)"', re.I)
VB_RE = re.compile(rb'viewBox\s*=\s*"([^"]+)"')
TEXT_RE = re.compile(rb"<text\b")


def find_svgs(root):
    return sorted(FsPath(root).rglob("*.svg"))


def project_key(stem):
    m = re.match(r"^(\d+)[-_]", stem)
    return m.group(1) if m else stem


def scan_svg(path, n_raw=len(RAW_CLASS_NAMES)):
    data = FsPath(path).read_bytes()
    prims = np.zeros(n_raw + 1, dtype=np.int64)
    insts = [set() for _ in range(n_raw + 1)]
    n = 0
    for m in TAG_RE.finditer(data):
        attrs = m.group(2)
        s, i = SEM_RE.search(attrs), INST_RE.search(attrs)
        raw = int(s.group(1)) - 1 if s else n_raw
        raw = raw if 0 <= raw < n_raw else n_raw
        prims[raw] += 1
        n += 1
        if i and int(i.group(1)) >= 0:
            insts[raw].add(int(i.group(1)))
    w = h = np.nan
    vb = VB_RE.search(data)
    if vb:
        parts = re.split(rb"[ ,]+", vb.group(1).strip())
        if len(parts) == 4:
            w, h = float(parts[2]), float(parts[3])
    row = {"n_prims": n, "n_text": len(TEXT_RE.findall(data)), "vb_w": w, "vb_h": h}
    for k in range(n_raw + 1):
        row[f"p{k}"] = int(prims[k])
        row[f"i{k}"] = len(insts[k])
    return row


def _scan_job(item):
    path, archive = item
    try:
        row = scan_svg(path)
        row["error"] = ""
    except Exception as e:
        row = {"error": repr(e)}
    row.update({"file": str(path), "archive": archive, "stem": FsPath(path).stem, "project": project_key(FsPath(path).stem)})
    return row


def scan_many(items, workers=4, chunksize=16):
    from multiprocessing import Pool
    if workers <= 1:
        rows = [_scan_job(it) for it in items]
    else:
        with Pool(workers) as pool:
            rows = list(pool.imap_unordered(_scan_job, items, chunksize))
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def target_matrices(df, lut, n_raw=len(RAW_CLASS_NAMES), targets=TARGET_CLASSES):
    inst = np.zeros((len(df), len(targets)), dtype=np.int64)
    prim = np.zeros((len(df), len(targets)), dtype=np.int64)
    for k in range(n_raw + 1):
        inst[:, lut[k]] += df[f"i{k}"].fillna(0).to_numpy(dtype=np.int64)
        prim[:, lut[k]] += df[f"p{k}"].fillna(0).to_numpy(dtype=np.int64)
    return inst, prim


TAG_RE = re.compile(rb"<(path|circle|ellipse|line|polyline|polygon|rect)\b([^>]*)>", re.S)
SEM_RE = re.compile(rb'semantic[-_]?id\s*=\s*"(-?\d+)"', re.I)
INST_RE = re.compile(rb'instance[-_]?id\s*=\s*"(-?\d+)"', re.I)
VB_RE = re.compile(rb'viewBox\s*=\s*"([^"]+)"')
TEXT_RE = re.compile(rb"<text\b")


def find_svgs(root):
    return sorted(FsPath(root).rglob("*.svg"))


def project_key(stem):
    m = re.match(r"^(\d+)[-_]", stem)
    return m.group(1) if m else stem


def scan_svg(path, n_raw=len(RAW_CLASS_NAMES)):
    data = FsPath(path).read_bytes()
    prims = np.zeros(n_raw + 1, dtype=np.int64)
    insts = [set() for _ in range(n_raw + 1)]
    n = 0
    for m in TAG_RE.finditer(data):
        attrs = m.group(2)
        s, i = SEM_RE.search(attrs), INST_RE.search(attrs)
        raw = int(s.group(1)) - 1 if s else n_raw
        raw = raw if 0 <= raw < n_raw else n_raw
        prims[raw] += 1
        n += 1
        if i and int(i.group(1)) >= 0:
            insts[raw].add(int(i.group(1)))
    w = h = np.nan
    vb = VB_RE.search(data)
    if vb:
        parts = re.split(rb"[ ,]+", vb.group(1).strip())
        if len(parts) == 4:
            w, h = float(parts[2]), float(parts[3])
    row = {"n_prims": n, "n_text": len(TEXT_RE.findall(data)), "vb_w": w, "vb_h": h}
    for k in range(n_raw + 1):
        row[f"p{k}"] = int(prims[k])
        row[f"i{k}"] = len(insts[k])
    return row


def _scan_job(item):
    path, archive = item
    try:
        row = scan_svg(path)
        row["error"] = ""
    except Exception as e:
        row = {"error": repr(e)}
    row.update({"file": str(path), "archive": archive, "stem": FsPath(path).stem, "project": project_key(FsPath(path).stem)})
    return row


def scan_many(items, workers=4, chunksize=16):
    from multiprocessing import Pool
    if workers <= 1:
        rows = [_scan_job(it) for it in items]
    else:
        with Pool(workers) as pool:
            rows = list(pool.imap_unordered(_scan_job, items, chunksize))
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def target_matrices(df, lut, n_raw=len(RAW_CLASS_NAMES), targets=TARGET_CLASSES):
    inst = np.zeros((len(df), len(targets)), dtype=np.int64)
    prim = np.zeros((len(df), len(targets)), dtype=np.int64)
    for k in range(n_raw + 1):
        inst[:, lut[k]] += df[f"i{k}"].fillna(0).to_numpy(dtype=np.int64)
        prim[:, lut[k]] += df[f"p{k}"].fillna(0).to_numpy(dtype=np.int64)
    return inst, prim


def _xy(p):
    return (float(p.x), float(p.y))


def _rgb(c):
    try:
        return (int(c.red), int(c.green), int(c.blue)) if c is not None and c.value is not None else (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def _int_attr(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


SEM_KEYS = ("semantic-id", "semanticId", "semantic_id", "semanticID")
INST_KEYS = ("instance-id", "instanceId", "instance_id", "instanceID")


def _label(values, keys):
    for k in keys:
        if values.get(k) not in (None, ""):
            return values[k]
    return None


def parse_svg(path, mm_per_unit=1.0, n_raw=len(RAW_CLASS_NAMES)):
    svg = SVG.parse(str(path), reify=True)
    ptype, pts, length, center, radius, sweep, raw, inst, color, width = ([] for _ in range(10))
    n_text = 0

    def add(t, p, L, c, r, s, rw, it, col, wd):
        ptype.append(t); pts.append(p); length.append(L); center.append(c)
        radius.append(r); sweep.append(s); raw.append(rw); inst.append(it); color.append(col); width.append(wd)

    for el in svg.elements():
        if isinstance(el, Text):
            n_text += 1
            continue
        if not isinstance(el, Shape):
            continue
        rw = _int_attr(_label(el.values, SEM_KEYS), 0) - 1
        rw = rw if 0 <= rw < n_raw else n_raw
        it = _int_attr(_label(el.values, INST_KEYS), -1)
        col, wd = _rgb(el.stroke), float(el.stroke_width or 0.0)
        if isinstance(el, (Circle, Ellipse)):
            bb = el.bbox()
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            if rx <= 0 or ry <= 0:
                continue
            t = 2 if abs(rx - ry) <= 1e-6 * max(rx, ry) else 3
            p = [(cx + rx, cy), (cx, cy + ry), (cx - rx, cy), (cx, cy - ry)]
            add(t, p, 2 * math.pi * math.sqrt((rx * rx + ry * ry) / 2), (cx, cy), (rx + ry) / 2, 2 * math.pi, rw, it, col, wd)
            continue
        for seg in Path(el).segments():
            if isinstance(seg, Move) or seg.start is None or seg.end is None:
                continue
            a, b = _xy(seg.start), _xy(seg.end)
            if isinstance(seg, (Line, Close)):
                L = math.dist(a, b)
                if L <= 1e-9:
                    continue
                p = [a, (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3),
                     (a[0] + 2 * (b[0] - a[0]) / 3, a[1] + 2 * (b[1] - a[1]) / 3), b]
                add(0, p, L, (np.nan, np.nan), 0.0, 0.0, rw, it, col, wd)
            elif isinstance(seg, Arc):
                p = [_xy(seg.point(t)) for t in (0.0, 1 / 3, 2 / 3, 1.0)]
                L = float(seg.length())
                if L <= 1e-9:
                    continue
                add(1, p, L, _xy(seg.center), float(seg.rx + seg.ry) / 2, float(seg.sweep), rw, it, col, wd)
            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                p = [_xy(seg.point(t)) for t in (0.0, 1 / 3, 2 / 3, 1.0)]
                L = float(seg.length(error=1e-4))
                if L <= 1e-9:
                    continue
                add(4, p, L, (np.nan, np.nan), 0.0, 0.0, rw, it, col, wd)
    k = float(mm_per_unit)
    n = len(ptype)
    return {
        "ptype": np.array(ptype, dtype=np.int8),
        "pts": (np.array(pts, dtype=np.float64).reshape(n, 4, 2) * k).astype(np.float32),
        "length": (np.array(length, dtype=np.float64) * k).astype(np.float32),
        "center": (np.array(center, dtype=np.float64).reshape(n, 2) * k).astype(np.float32),
        "radius": (np.array(radius, dtype=np.float64) * k).astype(np.float32),
        "sweep": np.array(sweep, dtype=np.float32),
        "raw": np.array(raw, dtype=np.int16),
        "inst": np.array(inst, dtype=np.int32),
        "color": np.array(color, dtype=np.uint8).reshape(n, 3),
        "width": np.array(width, dtype=np.float32),
        "n_text": n_text,
    }


def split_long_lines(P, max_len):
    long = (P["ptype"] == 0) & (P["length"] > max_len)
    if not long.any():
        return P
    keep = ~long
    parts = {k: [v[keep]] for k, v in P.items() if isinstance(v, np.ndarray)}
    for i in np.where(long)[0]:
        n = int(math.ceil(P["length"][i] / max_len))
        a, b = P["pts"][i, 0].astype(np.float64), P["pts"][i, 3].astype(np.float64)
        ts = np.linspace(0, 1, n + 1)
        for s, e in zip(ts[:-1], ts[1:]):
            q0, q1 = a + (b - a) * s, a + (b - a) * e
            seg = np.stack([q0 + (q1 - q0) * f for f in (0, 1 / 3, 2 / 3, 1)]).astype(np.float32)
            for k, v in P.items():
                if not isinstance(v, np.ndarray):
                    continue
                if k == "pts":
                    parts[k].append(seg[None])
                elif k == "length":
                    parts[k].append(np.array([np.linalg.norm(q1 - q0)], dtype=v.dtype))
                else:
                    parts[k].append(v[i:i + 1])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["n_text"] = P["n_text"]
    return out


def midpoints(P):
    return P["pts"].mean(axis=1)


def estimate_units(df, sample_paths, raw_names=RAW_CLASS_NAMES, block_mm=10000.0, door_mm=900.0):
    side = np.nanmax(df[["vb_w", "vb_h"]].to_numpy(dtype=float), axis=1)
    med_side = float(np.nanmedian(side))
    door = [i for i, n in enumerate(raw_names) if n == "single door"]
    radii = []
    for p in sample_paths:
        P = parse_svg(p, 1.0)
        m = (P["ptype"] == 1) & np.isin(P["raw"], door) & (np.abs(np.abs(P["sweep"]) - math.pi / 2) < math.radians(15))
        radii.extend(P["radius"][m].tolist())
    med_r = float(np.median(radii)) if radii else float("nan")
    by_block = block_mm / med_side
    by_door = door_mm / med_r if radii else float("nan")
    ratio = by_door / by_block if radii else float("nan")
    agree = bool(radii) and 0.6 <= ratio <= 1.6
    return {"median_viewbox_side": med_side, "mm_per_unit_from_10m_blocks": round(by_block, 4),
            "single_door_quarter_arcs": len(radii), "median_single_door_radius_units": round(med_r, 4),
            "mm_per_unit_from_doors": round(by_door, 4), "door_vs_block_ratio": round(ratio, 3),
            "suggested_mm_per_unit": round(by_block, 4), "evidence_agrees": agree,
            "implied_door_leaf_mm": round(med_r * by_block, 1) if radii else None}


def class_signatures(sample_paths, mm_per_unit, raw_names=RAW_CLASS_NAMES):
    n_raw = len(raw_names)
    acc = defaultdict(lambda: {"n_prims": 0, "types": Counter(), "stuff": 0, "lengths": [], "extents": []})
    for p in sample_paths:
        P = parse_svg(p, mm_per_unit)
        for k in np.unique(P["raw"]):
            m = P["raw"] == k
            a = acc[int(k)]
            a["n_prims"] += int(m.sum())
            a["types"].update(P["ptype"][m].tolist())
            a["stuff"] += int((P["inst"][m] < 0).sum())
            a["lengths"].extend(P["length"][m].tolist())
            for i in np.unique(P["inst"][m]):
                if i < 0:
                    continue
                q = P["pts"][m & (P["inst"] == i)].reshape(-1, 2)
                a["extents"].append(float(np.ptp(q, axis=0).max()))
    rows = []
    for k in range(n_raw + 1):
        a = acc.get(k)
        name = raw_names[k] if k < n_raw else "(unlabelled)"
        if not a or a["n_prims"] == 0:
            rows.append({"raw_id": k, "semanticId": k + 1 if k < n_raw else None, "name": name, "n_prims": 0, "check": "absent in sample"})
            continue
        tot = a["n_prims"]
        arc = (a["types"][1] + a["types"][2] + a["types"][3]) / tot
        stuff = a["stuff"] / tot
        ext = float(np.median(a["extents"])) if a["extents"] else float("nan")
        flags = []
        low = name.lower()
        if low in ("wall", "curtain wall", "railing", "parking spot") and stuff < 0.9 and low != "parking spot":
            flags.append("expected stuff (instanceId -1)")
        if "door" in low and low not in ("sliding door", "rolling door") and arc < 0.05:
            flags.append("door without arcs")
        if low in ("window", "blind window", "stairs") and arc > 0.3:
            flags.append("unexpectedly many arcs")
        if k < n_raw and low not in ("wall", "curtain wall", "railing") and stuff > 0.9:
            flags.append("thing class with no instances")
        rows.append({"raw_id": k, "semanticId": k + 1 if k < n_raw else None, "name": name, "n_prims": tot,
                     "n_instances": len(a["extents"]), "arc_frac": round(arc, 3), "stuff_frac": round(stuff, 3),
                     "median_prim_len_mm": round(float(np.median(a["lengths"])), 1),
                     "median_instance_extent_mm": round(ext, 1), "check": "; ".join(flags) or "ok"})
    return pd.DataFrame(rows)


def contact_sheet(sample_paths, mm_per_unit, out_png, raw_names=RAW_CLASS_NAMES, per_class=3, cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_raw = len(raw_names)
    found = defaultdict(list)
    for p in sample_paths:
        P = parse_svg(p, mm_per_unit)
        for k in np.unique(P["raw"]):
            k = int(k)
            if k >= n_raw or len(found[k]) >= per_class:
                continue
            m = P["raw"] == k
            ids = [i for i in np.unique(P["inst"][m]) if i >= 0] or [-1]
            for i in ids[: per_class - len(found[k])]:
                found[k].append(P["pts"][m & (P["inst"] == i)])
    rows = int(math.ceil(n_raw / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for k, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= n_raw:
            ax.axis("off")
            continue
        ax.set_title(f"semanticId {k + 1}: {raw_names[k]}", fontsize=8)
        for j, inst_pts in enumerate(found.get(k, [])):
            off = j * (np.ptp(inst_pts.reshape(-1, 2)[:, 0]) * 1.3 + 1)
            for q in inst_pts:
                ax.plot(q[:, 0] - inst_pts.reshape(-1, 2)[:, 0].min() + off, -(q[:, 1]), lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)
    return out_png


def assign_splits(df, val_drawings, seed=0):
    rnd = random.Random(seed)
    split = np.where(df["archive"].str.lower().str.startswith("test"), "test", "train_pool").astype(object)
    pool = df[split == "train_pool"]
    groups = sorted(pool["project"].unique())
    rnd.shuffle(groups)
    sizes = pool.groupby("project").size()
    limit = min(val_drawings, len(pool) // 2)
    val_groups, n = set(), 0
    for g in groups:
        if n >= limit:
            break
        val_groups.add(g)
        n += int(sizes[g])
    split[(split == "train_pool") & df["project"].isin(val_groups).to_numpy()] = "val_pool"
    out = df.copy()
    out["split_pool"] = split
    return out, {"n_projects_train_pool": len(groups), "n_val_projects": len(val_groups),
                 "per_file_grouping": len(groups) == len(pool)}


def stratified_sample(df, n, seed=0, bins=5):
    if n is None or n >= len(df):
        return df.index.to_numpy()
    q = pd.qcut(df["n_prims"].rank(method="first"), q=min(bins, len(df)), labels=False)
    rng = np.random.default_rng(seed)
    picks = []
    for b in np.unique(q):
        idx = df.index[q == b].to_numpy()
        take = int(round(n * len(idx) / len(df)))
        picks.extend(rng.choice(idx, size=min(take, len(idx)), replace=False).tolist())
    rest = np.setdiff1d(df.index.to_numpy(), picks)
    if len(picks) < n:
        picks.extend(rng.choice(rest, size=n - len(picks), replace=False).tolist())
    return np.array(sorted(picks[:n]))


def select_subset(df_pool, inst, prim, n_subset, quota_instances=400, quota_stuff_prims=20000,
                  greedy_share=0.6, seed=0, targets=TARGET_CLASSES):
    if len(df_pool) == 0:
        raise ValueError("training pool is empty: check archive names (non-'test' archives form the pool) and VAL_POOL_DRAWINGS")
    other = targets.index("other")
    smask = stuff_mask(targets)
    units = np.where(smask[None, :], prim, inst).astype(np.float64)
    units[:, other] = 0
    total = units.sum(0)
    quota = np.where(smask, quota_stuff_prims, quota_instances).astype(np.float64)
    rem = np.minimum(quota, total)
    w = 1.0 / np.sqrt(np.maximum(total, 1.0))
    w[other] = 0
    chosen = np.zeros(len(df_pool), dtype=bool)
    budget = int(n_subset * greedy_share)
    for _ in range(budget):
        gain = (np.minimum(units, rem[None, :]) * w[None, :]).sum(1)
        gain[chosen] = -1
        best = int(gain.argmax())
        if gain[best] <= 0:
            break
        chosen[best] = True
        rem = np.maximum(rem - units[best], 0)
    rest = df_pool[~chosen]
    fill = stratified_sample(rest, max(0, n_subset - int(chosen.sum())), seed=seed)
    pos = np.concatenate([np.where(chosen)[0], df_pool.index.get_indexer(fill)])
    sel = np.zeros(len(df_pool), dtype=bool)
    sel[pos] = True
    sub_units = units[sel].sum(0)
    report = pd.DataFrame({
        "class": targets, "kind": np.where(smask, "stuff (primitives)", "thing (instances)"),
        "pool_total": total.astype(int), "subset_total": sub_units.astype(int),
        "pool_share_%": np.round(100 * total / max(total[~smask].sum(), 1), 2),
        "subset_share_%": np.round(100 * sub_units / max(sub_units[~smask].sum(), 1), 2),
        "quota": quota.astype(int), "quota_met": sub_units >= np.minimum(quota, total),
    })
    report.loc[smask, ["pool_share_%", "subset_share_%"]] = np.nan
    report.loc[other, ["pool_total", "subset_total", "quota"]] = 0
    return np.where(sel)[0], int(chosen.sum()), report


NODE_FEATURES = ["is_line", "is_arc", "is_circle", "is_ellipse", "is_curve", "log_length_m", "cos2theta", "sin2theta",
                 "sweep_frac", "is_closed", "cx_tile", "cy_tile", "log_radius_m"]
EDGE_FEATURES = ["min_dist_rel", "centre_dist_m", "cos_dtheta", "abs_sin_dtheta", "log_len_ratio", "touch", "collinear", "perp_offset_m"]


def tile_windows(mid, cfg):
    n = len(mid)
    if n == 0:
        return []
    x0, y0 = mid.min(0)
    x1, y1 = mid.max(0)
    T, ov = cfg.tile_size_mm, cfg.tile_overlap_mm
    if n <= cfg.max_nodes and max(x1 - x0, y1 - y0) <= 1.5 * T:
        return [((float(x0), float(y0), float(x1), float(y1)), np.arange(n))]
    stride = T - ov
    xs = np.arange(x0, max(x1 - ov, x0) + 1e-6, stride)
    ys = np.arange(y0, max(y1 - ov, y0) + 1e-6, stride)
    out = []

    def rec(win, depth):
        a, b, c, d = win
        idx = np.where((mid[:, 0] >= a) & (mid[:, 0] <= c) & (mid[:, 1] >= b) & (mid[:, 1] <= d))[0]
        if len(idx) < cfg.min_nodes:
            return
        if len(idx) <= cfg.max_nodes or depth >= 6:
            out.append(((float(a), float(b), float(c), float(d)), idx[: cfg.max_nodes]))
            return
        mx, my, o = (a + c) / 2, (b + d) / 2, ov / (2 ** (depth + 1))
        for w in ((a, b, mx + o, my + o), (mx - o, b, c, my + o), (a, my - o, mx + o, d), (mx - o, my - o, c, d)):
            rec(w, depth + 1)

    for gx in xs:
        for gy in ys:
            rec((gx, gy, gx + T, gy + T), 0)
    return out


def _geoms(pts, ptype):
    closed = (ptype == 2) | (ptype == 3)
    rings = np.concatenate([pts, pts[:, :1]], axis=1)
    return np.array([LineString(rings[i] if closed[i] else pts[i]) for i in range(len(pts))], dtype=object)


def _cross2(u, v):
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def _cap_degree(i, j, n, k):
    deg = np.zeros(n, dtype=np.int32)
    keep = np.zeros(len(i), dtype=bool)
    for e in range(len(i)):
        a, b = i[e], j[e]
        if deg[a] < k and deg[b] < k:
            keep[e] = True
            deg[a] += 1
            deg[b] += 1
    return keep


def build_graph(P, idx, window, cfg, lut, stuff):
    x0, y0, x1, y1 = window
    T = cfg.tile_size_mm
    pts = P["pts"][idx].astype(np.float64) - np.array([x0, y0])
    ptype = P["ptype"][idx]
    length = P["length"][idx].astype(np.float64)
    n = len(idx)
    geoms = _geoms(pts, ptype)
    tree = STRtree(geoms)
    pi, pj = tree.query(geoms, predicate="dwithin", distance=cfg.radius_mm)
    m = pi < pj
    pairs = {(int(a), int(b)) for a, b in zip(pi[m], pj[m])}
    d = pts[:, 3] - pts[:, 0]
    theta = np.arctan2(d[:, 1], d[:, 0])
    is_line = ptype == 0
    ci, cj = tree.query(geoms, predicate="dwithin", distance=cfg.collinear_gap_mm)
    m = (ci < cj) & is_line[ci] & is_line[cj]
    ci, cj = ci[m], cj[m]
    dth = np.abs(np.sin(theta[ci] - theta[cj]))
    u = np.stack([np.cos(theta[ci]), np.sin(theta[ci])], 1)
    off = np.maximum(np.abs(_cross2(u, pts[cj, 0] - pts[ci, 0])), np.abs(_cross2(u, pts[cj, 3] - pts[ci, 0])))
    colin = (dth < math.sin(math.radians(cfg.collinear_angle_deg))) & (off < cfg.collinear_offset_mm)
    col_pairs = {(int(a), int(b)) for a, b in zip(ci[colin], cj[colin])}
    allp = sorted(pairs | col_pairs)
    if allp:
        ei = np.array([p[0] for p in allp], dtype=np.int64)
        ej = np.array([p[1] for p in allp], dtype=np.int64)
    else:
        ei = ej = np.zeros(0, dtype=np.int64)
    is_col = np.array([p in col_pairs for p in allp], dtype=bool)
    dist = shapely.distance(geoms[ei], geoms[ej]) if len(ei) else np.zeros(0)
    if len(ei):
        order = np.lexsort((dist, ~is_col))
        ei, ej, dist, is_col = ei[order], ej[order], dist[order], is_col[order]
        keep = _cap_degree(ei, ej, n, cfg.max_neighbors)
        ei, ej, dist, is_col = ei[keep], ej[keep], dist[keep], is_col[keep]
    mid = pts.mean(1)
    radius = P["radius"][idx].astype(np.float64)
    sweep = P["sweep"][idx].astype(np.float64)
    x = np.zeros((n, len(NODE_FEATURES)), dtype=np.float32)
    x[np.arange(n), ptype.astype(int)] = 1
    x[:, 5] = np.log1p(length / 1000)
    x[:, 6] = np.cos(2 * theta)
    x[:, 7] = np.sin(2 * theta)
    x[:, 8] = np.abs(sweep) / (2 * math.pi)
    x[:, 9] = ((ptype == 2) | (ptype == 3)).astype(np.float32)
    x[:, 10] = mid[:, 0] / T
    x[:, 11] = mid[:, 1] / T
    x[:, 12] = np.log1p(radius / 1000)
    ends = pts[:, [0, 3]]
    touch = np.zeros(len(ei), dtype=bool)
    perp = np.zeros(len(ei))
    if len(ei):
        dd = np.linalg.norm(ends[ei][:, :, None, :] - ends[ej][:, None, :, :], axis=-1).reshape(len(ei), -1).min(1)
        touch = dd <= cfg.touch_tol_mm
        both = is_line[ei] & is_line[ej]
        u = np.stack([np.cos(theta[ei]), np.sin(theta[ei])], 1)
        perp = np.where(both, np.abs(_cross2(u, mid[ej] - pts[ei, 0])), 0.0)
    dtheta = theta[ei] - theta[ej]
    ea = np.zeros((len(ei), len(EDGE_FEATURES)), dtype=np.float32)
    ea[:, 0] = dist / cfg.radius_mm
    ea[:, 1] = np.linalg.norm(mid[ei] - mid[ej], axis=1) / 1000
    ea[:, 2] = np.cos(dtheta)
    ea[:, 3] = np.abs(np.sin(dtheta))
    ea[:, 4] = np.log((length[ei] + 1) / (length[ej] + 1))
    ea[:, 5] = touch
    ea[:, 6] = is_col
    ea[:, 7] = perp / 1000
    y = lut[P["raw"][idx]].astype(np.int64)
    raw_inst = P["inst"][idx].astype(np.int64)
    thing = ~stuff[y] & (y != TARGET_CLASSES.index("other")) & (raw_inst >= 0)
    key = np.where(thing, P["raw"][idx].astype(np.int64) * 10_000_000 + raw_inst, -1)
    uniq = {k: c for c, k in enumerate(sorted(set(key[key >= 0].tolist())))}
    inst = np.array([uniq[k] if k >= 0 else -1 for k in key], dtype=np.int32)
    edge_y = ((inst[ei] == inst[ej]) & (inst[ei] >= 0)).astype(np.uint8)
    edge_mask = (inst[ei] >= 0) | (inst[ej] >= 0)
    return {
        "x": x, "pts": pts.astype(np.float32), "ptype": ptype.astype(np.int8), "y": y.astype(np.int16), "inst": inst,
        "raw": P["raw"][idx].astype(np.int16), "edge_index": np.stack([ei, ej]).astype(np.int32),
        "edge_attr": ea, "edge_y": edge_y, "edge_mask": edge_mask,
    }


def validate_tile(t, cfg):
    n, e = len(t["y"]), t["edge_index"].shape[1]
    problems = []
    if n > cfg.max_nodes:
        problems.append(f"{n} nodes > cap")
    if e and (t["edge_index"].min() < 0 or t["edge_index"].max() >= n):
        problems.append("edge index out of range")
    if e and (t["edge_index"][0] >= t["edge_index"][1]).any():
        problems.append("edges not stored as i<j")
    deg = np.bincount(t["edge_index"].ravel(), minlength=n) if e else np.zeros(n)
    if deg.max(initial=0) > cfg.max_neighbors:
        problems.append(f"degree {deg.max()} > cap")
    if not np.isfinite(t["x"]).all() or not np.isfinite(t["edge_attr"]).all():
        problems.append("non-finite features")
    if (t["y"] < 0).any() or (t["y"] >= len(TARGET_CLASSES)).any():
        problems.append("label out of range")
    if (t["edge_y"].astype(bool) & ~t["edge_mask"]).any():
        problems.append("positive edge outside mask")
    return problems


def process_drawing(path, cfg, lut, stuff):
    P = parse_svg(path, cfg.mm_per_unit)
    P = split_long_lines(P, cfg.max_prim_len_mm)
    tiles = []
    for t, (win, idx) in enumerate(tile_windows(midpoints(P), cfg)):
        g = build_graph(P, idx, win, cfg, lut, stuff)
        g["meta"] = {"drawing": FsPath(path).stem, "tile": t, "window": [round(v, 1) for v in win],
                     "n_nodes": int(len(idx)), "n_edges": int(g["edge_index"].shape[1]), "n_text": int(P["n_text"])}
        tiles.append(g)
    return tiles


NODE_KEYS = ["x", "pts", "ptype", "y", "inst", "raw"]
EDGE_KEYS = ["edge_attr", "edge_y", "edge_mask"]


class ShardWriter:
    def __init__(self, out_dir, split, tiles_per_shard=512):
        self.dir = FsPath(out_dir) / "graphs" / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split, self.per = split, tiles_per_shard
        self.buf, self.index, self.k = [], [], 0

    def add(self, tile):
        self.buf.append(tile)
        if len(self.buf) >= self.per:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        name = f"{self.split}-{self.k:05d}.npz"
        node_ptr = np.cumsum([0] + [len(t["y"]) for t in self.buf]).astype(np.int64)
        edge_ptr = np.cumsum([0] + [t["edge_index"].shape[1] for t in self.buf]).astype(np.int64)
        arrays = {"node_ptr": node_ptr, "edge_ptr": edge_ptr,
                  "edge_index": np.concatenate([t["edge_index"] for t in self.buf], axis=1)}
        for key in NODE_KEYS + EDGE_KEYS:
            arrays[key] = np.concatenate([t[key] for t in self.buf], axis=0)
        np.savez_compressed(self.dir / name, **arrays)
        for pos, t in enumerate(self.buf):
            row = dict(t["meta"])
            row["window"] = json.dumps(row["window"])
            counts = np.bincount(t["y"].astype(np.int64), minlength=len(TARGET_CLASSES))
            row.update({"split": self.split, "shard": f"graphs/{self.split}/{name}", "pos": pos,
                        "n_instances": int(t["inst"].max() + 1) if len(t["inst"]) else 0,
                        "pos_edges": int(t["edge_y"].sum()), "masked_edges": int(t["edge_mask"].sum())})
            row.update({f"n_{c}": int(v) for c, v in zip(TARGET_CLASSES, counts)})
            self.index.append(row)
        self.buf = []
        self.k += 1


def _process_job(args):
    path, cfg, lut, stuff = args
    try:
        return path, process_drawing(path, cfg, lut, stuff), ""
    except Exception as e:
        return path, [], repr(e)


def process_split(paths, split, out_dir, cfg, lut, stuff, workers=4, tiles_per_shard=512, log_every=200):
    writer = ShardWriter(out_dir, split, tiles_per_shard)
    errors, t0 = [], time.time()
    jobs = [(str(p), cfg, lut, stuff) for p in paths]
    if workers <= 1:
        results = map(_process_job, jobs)
        pool = None
    else:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap_unordered(_process_job, jobs, chunksize=4)
    for k, (path, tiles, err) in enumerate(results, 1):
        if err:
            errors.append({"file": path, "error": err})
        for t in tiles:
            writer.add(t)
        if k % log_every == 0:
            print(f"{split}: {k}/{len(jobs)} drawings, {len(writer.index) + len(writer.buf)} tiles, {time.time() - t0:.0f}s")
    if pool is not None:
        pool.close()
        pool.join()
    writer.flush()
    index = pd.DataFrame(writer.index)
    index.to_parquet(FsPath(out_dir) / f"index_{split}.parquet", index=False)
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(FsPath(out_dir) / f"errors_{split}.csv", index=False)
    print(f"{split}: done {len(jobs)} drawings -> {len(index)} tiles, {len(errors)} errors, {time.time() - t0:.0f}s")
    return index, errors


def load_shard(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def get_tile(shard, pos):
    a, b = shard["node_ptr"][pos], shard["node_ptr"][pos + 1]
    c, d = shard["edge_ptr"][pos], shard["edge_ptr"][pos + 1]
    tile = {k: shard[k][a:b] for k in NODE_KEYS}
    tile.update({k: shard[k][c:d] for k in EDGE_KEYS})
    tile["edge_index"] = shard["edge_index"][:, c:d]
    return tile


def to_pyg(tile, undirected=True):
    import torch
    from torch_geometric.data import Data
    ei = torch.from_numpy(tile["edge_index"].astype(np.int64))
    ea = torch.from_numpy(tile["edge_attr"])
    ey = torch.from_numpy(tile["edge_y"].astype(np.float32))
    em = torch.from_numpy(tile["edge_mask"])
    if undirected:
        ei, ea, ey, em = torch.cat([ei, ei.flip(0)], 1), torch.cat([ea, ea]), torch.cat([ey, ey]), torch.cat([em, em])
    return Data(x=torch.from_numpy(tile["x"]), pts=torch.from_numpy(tile["pts"]), edge_index=ei, edge_attr=ea,
                y=torch.from_numpy(tile["y"].astype(np.int64)), inst=torch.from_numpy(tile["inst"].astype(np.int64)),
                edge_y=ey, edge_mask=em)


def compute_stats(out_dir, split, cfg, max_problems=20):
    idx = pd.read_parquet(FsPath(out_dir) / f"index_{split}.parquet")
    counts = np.zeros(len(TARGET_CLASSES), dtype=np.int64)
    pos = masked = 0
    problems = []
    for shard_path in sorted(idx["shard"].unique()):
        S = load_shard(FsPath(out_dir) / shard_path)
        counts += np.bincount(S["y"].astype(np.int64), minlength=len(TARGET_CLASSES))
        pos += int(S["edge_y"].sum())
        masked += int(S["edge_mask"].sum())
        for k in range(len(S["node_ptr"]) - 1):
            for p in validate_tile(get_tile(S, k), cfg):
                if len(problems) < max_problems:
                    problems.append(f"{shard_path}#{k}: {p}")
    freq = counts / max(counts.sum(), 1)
    present = counts > 0
    med = np.median(freq[present]) if present.any() else 1.0
    weights = np.where(present, np.clip(med / np.maximum(freq, 1e-12), 0.1, 10.0), 0.0)
    return {
        "split": split, "tiles": int(len(idx)), "drawings": int(idx["drawing"].nunique()),
        "nodes": int(counts.sum()), "edges": int(idx["n_edges"].sum()),
        "nodes_per_tile": {"median": float(idx["n_nodes"].median()), "max": int(idx["n_nodes"].max())},
        "edges_per_tile": {"median": float(idx["n_edges"].median()), "max": int(idx["n_edges"].max())},
        "class_counts": dict(zip(TARGET_CLASSES, counts.tolist())),
        "class_weights": dict(zip(TARGET_CLASSES, np.round(weights, 4).tolist())),
        "edge_pos": pos, "edge_masked": masked,
        "edge_pos_weight": round((masked - pos) / max(pos, 1), 4),
        "absent_classes": [c for c, p in zip(TARGET_CLASSES, present) if not p],
        "problems": problems,
    }


def write_dataset_card(out_dir, info):
    t = info["taxonomy"]
    rows = "\n".join(f"| {i} | `{c}` | {'stuff' if c in STUFF_TARGETS else 'thing'} | {', '.join(info['harmonization_inverse'].get(c, [])) or '-'} |"
                     for i, c in enumerate(t))
    stats = "\n".join(f"| {s['split']} | {s['drawings']} | {s['tiles']} | {s['nodes']:,} | {s['edges']:,} | {s['edge_pos_weight']} |"
                      for s in info["stats"])
    card = f"""---
license: cc-by-nc-4.0
pretty_name: FloorPlanCAD primitive graphs (Visionaries subset)
task_categories:
- graph-ml
tags:
- floorplan
- cad
- panoptic-symbol-spotting
- gnn
---

# FloorPlanCAD primitive graphs: Visionaries pre-training subset

Graph-structured training data for a GAT-CADNet-style panoptic symbol spotting model (CV5570, Team Visionaries).
Each sample is one tile of a FloorPlanCAD drawing: nodes are vector primitives (line, arc, circle, ellipse, curve),
edges link primitives within {info['config']['radius_mm']:.0f} mm plus collinear neighbours, capped at {info['config']['max_neighbors']} per node and
{info['config']['max_nodes']} nodes per tile.

**Source and license.** Derived from FloorPlanCAD (Fan et al., ICCV 2021, https://floorplancad.github.io/). The annotations are
CC BY-NC 4.0, and the dataset authors state they do not own the copyright of the drawings. This derivative is therefore
**non-commercial only** and is kept private by default. No raw SVGs are included unless `INCLUDE_RAW_SVGS` was enabled.

## Splits

| split | drawings | tiles | nodes | edges | edge pos_weight |
|---|---|---|---|---|---|
{stats}

`train` is a stratified {info['subset']['n_subset']}-drawing subset of the FloorPlanCAD training pool ({info['subset']['n_greedy']} chosen greedily to
cover rare classes, the rest size-stratified random). `val` is drawn from held-out projects. `test` comes from the official test split.
The file lists are in `splits/`.

## Taxonomy (harmonised from the 35 FloorPlanCAD classes)

| id | class | kind | FloorPlanCAD source classes |
|---|---|---|---|
{rows}

Everything not listed, including unlabelled primitives, maps to `other`. Instance ids exist only for thing classes.

## Files

- `graphs/<split>/<split>-NNNNN.npz`: shards of concatenated tiles. Slice a tile with `node_ptr` and `edge_ptr`.
  - node arrays: `x` [N,{len(NODE_FEATURES)}] float32, `pts` [N,4,2] (4 sampled points, tile-local mm), `ptype`, `y` (class id), `inst` (-1 = none), `raw` (original FloorPlanCAD id, 0-based, 35 = unlabelled)
  - edge arrays: `edge_index` [2,E] with i<j (undirected, stored once), `edge_attr` [E,{len(EDGE_FEATURES)}], `edge_y` (same instance), `edge_mask` (edge counts in the instance loss)
- `index_<split>.parquet`: one row per tile (drawing, window, shard, position, sizes, per-class node counts).
- `stats.json`: class counts and weights, edge pos_weight, validation problems. `config.json`: every parameter used.
- `fpgraph.py`: the parser and graph builder. Use the same file for fine-tuning and inference.

Node features: {', '.join(NODE_FEATURES)}.
Edge features: {', '.join(EDGE_FEATURES)}.
Stroke colour is deliberately **not** a feature, because FloorPlanCAD colours encode the class.

```python
from fpgraph import load_shard, get_tile, to_pyg
tile = get_tile(load_shard("graphs/train/train-00000.npz"), 0)
data = to_pyg(tile)
```
"""
    (FsPath(out_dir) / "README.md").write_text(card)
    return FsPath(out_dir) / "README.md"


def push_to_hub(out_dir, repo_id, private=True, token=None):
    from huggingface_hub import HfApi, get_token
    token = token or get_token()
    if not token:
        print("No Hugging Face token found. Run `from huggingface_hub import notebook_login; notebook_login()` "
              "or set the HF_TOKEN environment variable, then re-run this cell.")
        return None
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    out_dir = FsPath(out_dir)
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir))
    else:
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir),
                          ignore_patterns=["graphs/**", "raw_svg/**"], commit_message="Phase A metadata, card, code")
        for sub in sorted(p for p in (out_dir / "graphs").iterdir() if p.is_dir()):
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(sub),
                              path_in_repo=f"graphs/{sub.name}", commit_message=f"Phase A shards: {sub.name}")
            print("uploaded", sub.name)
        if (out_dir / "raw_svg").exists():
            api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir / "raw_svg"),
                              path_in_repo="raw_svg", commit_message="Phase A raw SVGs")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print("pushed to", url)
    return url


def upload_plan(out_dir):
    files = [p for p in FsPath(out_dir).rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    by_top = Counter(p.relative_to(out_dir).parts[0] for p in files)
    return {"files": len(files), "total_mb": round(total / 1e6, 2), "by_folder": dict(by_top)}
