import re
import math
import time
import contextlib
import html
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon, LineString, box
from scipy.spatial.distance import cdist

import tracka_core as core
from fpgraph import TARGET_CLASSES

CV_CONFIG = {
    "instance_size_mm": {"door": (450.0, 6000.0), "window": (300.0, 10000.0)},
    "axis_search_pad": 300.0,
    "axis_bin_deg": 2.0,
    "axis_tol_deg": 3.0,
    "arc_flatten_tol": 5.0,
    "circle_segments": 16,
    "merge_tol": 1.0,
    "swing_span_rad": (math.radians(20.0), math.radians(180.0) + 0.01),
    "min_extent": 10.0,
    "size_attach_radius": 1500.0,
    "room_text_max_len": 40,
    "min_room_letters": 3,
    "door_code_regex": r"^[A-Z]{1,3}[- ]?\d+(\s*[X*]\s*\d+)?[A-Z]?(-\d+)?$",
    "boundary_conf_pad": 30.0,
    "wall_conf_pad": 20.0,
}

BOUNDARY_LAYER = {"wall": "A-WALL", "curtain_wall": "A-CW", "railing": "A-RAIL"}
DOOR_CLASSES = {c for c in TARGET_CLASSES if c.startswith("door")}
OPENING_CLASSES = DOOR_CLASSES | {"window"}
DXF_TYPE = {0: "LINE", 1: "ARC", 2: "CIRCLE", 3: "ELLIPSE", 4: "SPLINE"}


@contextlib.contextmanager
def patched(**overrides):
    saved = {k: getattr(core, k) for k in overrides}
    for k, v in overrides.items():
        setattr(core, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(core, k, v)


def to_wcs(P, origin):
    x0, y1 = origin
    pts = P["pts"].astype(np.float64)
    pts = np.stack([pts[..., 0] + x0, y1 - pts[..., 1]], -1)
    ctr = P["center"].astype(np.float64)
    ctr = np.stack([ctr[:, 0] + x0, y1 - ctr[:, 1]], 1)
    return {"pts": pts, "center": ctr, "ptype": P["ptype"], "radius": P["radius"].astype(np.float64),
            "sweep": np.abs(P["sweep"]).astype(np.float64), "length": P["length"].astype(np.float64)}


def prim_polyline(Pw, i, cv):
    t, p = int(Pw["ptype"][i]), Pw["pts"][i]
    if t == 0:
        return [tuple(p[0]), tuple(p[3])], False
    c, r = Pw["center"][i], Pw["radius"][i]
    if t in (2, 3):
        rx, ry = abs(p[0][0] - c[0]), abs(p[1][1] - c[1])
        a = np.linspace(0, 2 * math.pi, cv["circle_segments"], endpoint=False)
        return [(float(c[0] + rx * math.cos(k)), float(c[1] + ry * math.sin(k))) for k in a], True
    span = Pw["sweep"][i]
    if t == 1 and r > 0 and span > 0 and np.isfinite(c).all():
        a0 = math.atan2(p[0][1] - c[1], p[0][0] - c[0])
        a1 = math.atan2(p[1][1] - c[1], p[1][0] - c[0])
        sign = 1.0 if (a1 - a0 + math.pi) % (2 * math.pi) - math.pi >= 0 else -1.0
        step = 2 * math.acos(max(-1.0, 1.0 - cv["arc_flatten_tol"] / r))
        n = max(2, int(math.ceil(span / max(step, 1e-3))))
        pts = [(float(c[0] + r * math.cos(a0 + sign * span * k / n)), float(c[1] + r * math.sin(a0 + sign * span * k / n)))
               for k in range(n + 1)]
        pts[0], pts[-1] = tuple(p[0]), tuple(p[3])
        return pts, False
    return [tuple(q) for q in p], False

def build_instances(Pw, scene, cv):
    cls, prob, seg = scene["cls"], scene["prob"], scene["seg"]
    idx = np.where(seg >= 0)[0]
    if not len(idx):
        return []
    order = idx[np.argsort(seg[idx], kind="stable")]
    groups = np.split(order, np.flatnonzero(np.diff(seg[order])) + 1)
    out = []
    for members in groups:
        name = TARGET_CLASSES[int(cls[members[0]])]
        if name not in OPENING_CLASSES:
            continue
        kind = "door" if name in DOOR_CLASSES else "window"
        size = float(np.ptp(Pw["pts"][members].reshape(-1, 2), axis=0).max())
        lo, hi = cv["instance_size_mm"][kind]
        out.append({"id": int(seg[members[0]]), "cls": name, "kind": kind, "members": members, "size_mm": size,
                    "accepted": lo <= size <= hi, "conf": float(prob[members].min())})
    return out


def merge_touching_lines(ends, tol=1.0):
    n = len(ends)
    if n < 2:
        return ends
    d = ends[:, 1] - ends[:, 0]
    L = np.hypot(d[:, 0], d[:, 1])
    u = d / np.maximum(L, 1e-12)[:, None]
    flip = (u[:, 0] < -1e-9) | ((np.abs(u[:, 0]) <= 1e-9) & (u[:, 1] < 0))
    u = np.where(flip[:, None], -u, u)
    a = np.where(flip[:, None], ends[:, 1], ends[:, 0])
    t0 = (a * u).sum(1)
    t1 = t0 + L
    off = u[:, 0] * a[:, 1] - u[:, 1] * a[:, 0]
    kt = np.round(np.arctan2(u[:, 1], u[:, 0]) / 1e-4).astype(np.int64)
    ko = np.round(off).astype(np.int64)
    order = np.lexsort((t0, ko, kt))
    kt_s, ko_s, t0_s, t1_s = (v[order].tolist() for v in (kt, ko, t0, t1))
    firsts, tops = [], []
    for k in range(n):
        if k and kt_s[k] == kt_s[k - 1] and ko_s[k] == ko_s[k - 1] and t0_s[k] <= tops[-1] + tol:
            tops[-1] = max(tops[-1], t1_s[k])
        else:
            firsts.append(k)
            tops.append(t1_s[k])
    f = order[firsts]
    start = a[f]
    stop = start + u[f] * (np.array(tops) - t0[f])[:, None]
    return np.stack([start, stop], 1)


def cv_leaves(Pw, scene, instances, cv):
    cls, ptype = scene["cls"], Pw["ptype"]
    keep_window = np.zeros(len(ptype), dtype=bool)
    for ins in instances:
        if ins["accepted"] and ins["kind"] == "window":
            keep_window[ins["members"]] = True
    leaves, boundary = [], []
    for name, layer in BOUNDARY_LAYER.items():
        k = TARGET_CLASSES.index(name)
        sel = np.where(cls == k)[0]
        boundary.append(sel)
        lines = sel[ptype[sel] == 0]
        for a, b in merge_touching_lines(Pw["pts"][lines][:, [0, 3]], cv["merge_tol"]):
            leaves.append(core.Leaf("LINE", layer, "", name, [tuple(a), tuple(b)], False, 0.0))
        for i in sel[ptype[sel] != 0]:
            pts, closed = prim_polyline(Pw, i, cv)
            leaves.append(core.Leaf(DXF_TYPE[int(ptype[i])], layer, "", name, pts, closed, float(Pw["radius"][i])))
    for i in np.where((cls == TARGET_CLASSES.index("window")) & keep_window)[0]:
        pts, closed = prim_polyline(Pw, i, cv)
        leaves.append(core.Leaf(DXF_TYPE[int(ptype[i])], "A-WIN", "A-WIN", "window", pts, closed, float(Pw["radius"][i])))
    return leaves, np.concatenate(boundary).astype(np.int64)

def wall_axis(M, bnd_segs, seg_tree, cv):
    pad = cv["axis_search_pad"]
    lo, hi = M.min(0) - pad, M.max(0) + pad
    hit = seg_tree.query(box(lo[0], lo[1], hi[0], hi[1]), predicate="intersects")
    if not len(hit):
        return None, None
    s = bnd_segs[hit]
    d = s[:, 2:] - s[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 1e-6
    s, d, L = s[ok], d[ok], L[ok]
    if not len(L):
        return None, None
    ang = np.arctan2(d[:, 1], d[:, 0]) % math.pi
    nb = int(round(180.0 / cv["axis_bin_deg"]))
    bins = np.round(np.degrees(ang) / cv["axis_bin_deg"]).astype(int) % nb
    mode = math.radians(int(np.bincount(bins, weights=L, minlength=nb).argmax()) * cv["axis_bin_deg"])
    dev = (ang - mode + math.pi / 2) % math.pi - math.pi / 2
    near = np.abs(dev) <= math.radians(cv["axis_tol_deg"])
    theta = mode + float((L[near] * dev[near]).sum() / L[near].sum())
    return theta, s[near]


def wall_offset(par, v, v0, v1):
    mid = (v0 + v1) / 2
    if par is None or not len(par):
        return mid
    off = ((par[:, :2] + par[:, 2:]) / 2) @ v
    dist = np.maximum(np.maximum(v0 - off, off - v1), 0.0) + 1e-6 * np.abs(off - mid)
    return float(off[int(dist.argmin())])


def opening_from_instance(ins, Pw, bnd_segs, endpoints, ep_tree, seg_tree, cfg, cv):
    mem = ins["members"]
    M = Pw["pts"][mem].reshape(-1, 2)
    theta, par = wall_axis(M, bnd_segs, seg_tree, cv)
    if theta is None:
        return None
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    pu, pv = M @ u, M @ v
    u0, u1, v0, v1 = float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())
    if u1 - u0 < cv["min_extent"]:
        u0, u1 = (u0 + u1) / 2 - cv["min_extent"] / 2, (u0 + u1) / 2 + cv["min_extent"] / 2
    if v1 - v0 < cv["min_extent"]:
        v0, v1 = (v0 + v1) / 2 - cv["min_extent"] / 2, (v0 + v1) / 2 + cv["min_extent"] / 2
    corners = [tuple(a * u + b * v) for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    obb = Polygon(corners)
    pad = obb.buffer(cfg["opening_search_pad"], join_style=2)
    idx = ep_tree.query(pad, predicate="contains")
    jambs = np.unique(np.round(endpoints[idx], 1), axis=0) if len(idx) else np.zeros((0, 2))
    arcs = (Pw["ptype"][mem] == 1) & (Pw["sweep"][mem] >= cv["swing_span_rad"][0]) & (Pw["sweep"][mem] <= cv["swing_span_rad"][1])
    radii = sorted(round(float(r), 1) for r in Pw["radius"][mem][arcs])
    block_w = u1 - u0
    leaf_w = max(radii) if radii else None
    big = [r for r in radii if r >= 0.8 * max(radii)] if radii else []
    geometric_double = len(big) >= 2 and abs(2 * big[-1] - block_w) <= 0.15 * block_w
    if geometric_double:
        leaf_w = 2 * big[-1]
    double_leaf = geometric_double or ins["cls"] == "door_double"
    clear_w, plug, centre = None, None, None
    if len(jambs) >= 2:
        proj, off = jambs @ u, jambs @ v
        split = (proj.max() + proj.min()) / 2
        left, right = proj <= split, proj > split
        if left.any() and right.any():
            clear_w = float(proj[right].min() - proj[left].max())
            bridges = []
            for li in np.where(left)[0]:
                ri = np.where(right)[0][np.argmin(np.abs(off[right] - off[li]))]
                bridges.append(LineString([tuple(jambs[li]), tuple(jambs[ri])]))
            if clear_w > 0.4 * block_w:
                plug = shapely.unary_union(bridges).buffer(cfg["snap_tol"], cap_style="flat")
                centre = shapely.MultiPoint([tuple(p) for p in jambs]).centroid
    ref_w = leaf_w or block_w
    if clear_w is not None and not (0.5 * ref_w <= clear_w <= 1.6 * ref_w):
        clear_w = None
    if plug is None:
        vw = wall_offset(par, v, v0, v1)
        edge = LineString([tuple(u0 * u + vw * v), tuple(u1 * u + vw * v)])
        plug = edge.buffer(cfg["snap_tol"], cap_style=2)
        centre = edge.interpolate(0.5, normalized=True)
    width = clear_w or leaf_w or block_w
    method = "jamb_clear_opening" if clear_w else ("leaf_arc_radius" if leaf_w else "block_extent")
    return {
        "kind": ins["kind"], "block": ins["cls"], "layer": None, "subtype": ins["cls"],
        "position": tuple(map(float, centre.coords[0])), "insert": tuple(map(float, centre.coords[0])),
        "rotation": float(math.degrees(theta) % 360.0), "mirrored": False, "axis": u.tolist(), "normal": v.tolist(),
        "width_mm": width, "width_method": method, "width_clear_mm": clear_w, "width_leaf_mm": leaf_w,
        "width_block_mm": block_w, "width_nominal_mm": None, "height_nominal_mm": None, "double_leaf": bool(double_leaf),
        "jamb_points": len(jambs), "_plug": plug, "_obb": obb, "cv_class": ins["cls"], "cv_confidence": ins["conf"],
        "cv_instance": ins["id"],
    }


def build_openings_cv(instances, Pw, bnd_segs, cfg, cv):
    if not len(bnd_segs):
        return []
    endpoints = np.vstack([bnd_segs[:, :2], bnd_segs[:, 2:]])
    ep_tree = STRtree(shapely.points(endpoints))
    seg_tree = STRtree(shapely.linestrings(bnd_segs.reshape(-1, 2, 2)))
    out = []
    for ins in instances:
        if ins["accepted"]:
            rec = opening_from_instance(ins, Pw, bnd_segs, endpoints, ep_tree, seg_tree, cfg, cv)
            if rec:
                out.append(rec)
    return out

def read_svg_texts(svg_path, origin):
    x0, y1 = origin
    raw = Path(svg_path).read_text(encoding="utf-8", errors="replace")
    out = []
    for m in re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>(.*?)</text>', raw, re.S):
        s = html.unescape(m.group(3)).strip()
        if s:
            out.append({"text": s, "layer": "", "owner_layer": "", "insert": (float(m.group(1)) + x0, y1 - float(m.group(2)))})
    return out


def cv_tags(texts, cfg, cv):
    ann = re.compile(cfg["annotation_label_regex"], re.I)
    code = re.compile(cv["door_code_regex"])
    labels, sizes = [], []
    for t in texts:
        s = " ".join(t["text"].split())
        if not s:
            continue
        size = core.parse_size(s)
        if size is not None:
            sizes.append((t["insert"], s, size))
            continue
        if sum(ch.isalpha() for ch in s) < cv["min_room_letters"] or len(s) > cv["room_text_max_len"] or code.match(s.upper()):
            continue
        labels.append({"raw_label": s, "raw_size": "", "nominal": None, "position": t["insert"], "annotation": bool(ann.search(s))})
    real = [k for k, lab in enumerate(labels) if not lab["annotation"]]
    if real and sizes:
        D = cdist(np.array([labels[k]["position"] for k in real]), np.array([s[0] for s in sizes]))
        pairs = sorted(zip(*np.where(D <= cv["size_attach_radius"])), key=lambda ij: D[ij])
        used_l, used_s = set(), set()
        for i, j in pairs:
            if i in used_l or j in used_s:
                continue
            used_l.add(i)
            used_s.add(j)
            labels[real[i]]["raw_size"], labels[real[i]]["nominal"] = sizes[j][1], sizes[j][2]
    return labels

def cv_scale(info):
    return {"insunits": 4, "insunits_source": "svg_export", "unit_name": "mm", "to_metre": 0.001, "floor_insert_scale": [1.0, 1.0],
            "block_units": 0, "dimlfac_header": 1.0, "issues": [], "cross_checks": {},
            "normalised_to_mm_factor": float(info["mm_factor"]), "unit_inference": info["units"]}


def min_prob_in(geoms, tree, probs, pad):
    out = []
    for g in geoms:
        hit = tree.query(g.buffer(pad), predicate="intersects")
        out.append((float(probs[hit].min()), float(probs[hit].mean())) if len(hit) else (None, None))
    return out


def run_cv(svg_path, info, Pw, scene, top=None, cfg=None, cv=None):
    cfg = cfg or core.CONFIG
    cv = {**CV_CONFIG, **(cv or {})}
    t0 = time.time()
    texts = read_svg_texts(svg_path, info["origin_mm"])
    tags = cv_tags(texts, cfg, cv)
    instances = build_instances(Pw, scene, cv)
    leaves, boundary_idx = cv_leaves(Pw, scene, instances, cv)
    ex = core.Exploded(leaves=leaves, texts=texts)
    scale = cv_scale(info)
    floor = top if top is not None else SimpleNamespace(dxf=SimpleNamespace(name=info.get("floor_block") or Path(svg_path).stem, insert=(0.0, 0.0, 0.0)))
    with patched(load_floor=lambda p, c: (None, floor, ex, scale),
                 build_openings=lambda doc, e, segs, c: build_openings_cv(instances, Pw, segs, c, cv),
                 room_tags=lambda e, c: tags,
                 check_dimensions=lambda e, s, c: []):
        r = core.run(svg_path, cfg)
    manifest = core.build_manifest(info.get("source_file") or Path(svg_path).name, r, cfg)
    annotate(manifest, r, Pw, scene, instances, boundary_idx, tags, cv)
    manifest["summary"]["elapsed_s"] = round(r["elapsed"], 1)
    manifest["summary"]["cv"]["adapter_elapsed_s"] = round(time.time() - t0, 1)
    core.ManifestM.model_validate(manifest)
    return manifest, r


def annotate(manifest, r, Pw, scene, instances, boundary_idx, tags, cv):
    prob = scene["prob"]
    b_probs = prob[boundary_idx]
    b_tree = STRtree(shapely.linestrings(Pw["pts"][boundary_idx])) if len(boundary_idx) else None
    geoms = {s["id"]: s["geom"] for s in r["spaces"]}
    rooms = manifest["rooms"]
    conf = min_prob_in([geoms[x["id"]] for x in rooms], b_tree, b_probs, cv["boundary_conf_pad"]) if b_tree is not None else [(None, None)] * len(rooms)
    for rm, (lo, mean) in zip(rooms, conf):
        rm["source"] = "cv_model"
        rm["cv_boundary_conf_min"], rm["cv_boundary_conf_mean"] = lo, mean
        if lo is not None:
            rm["confidence"] = round(min(rm["confidence"], lo), 4)
    walls = manifest["walls"]
    boxes = [LineString([w["start"], w["end"]]).buffer(w["thickness_mm"] / 2 + cv["wall_conf_pad"], cap_style=2) for w in walls]
    wconf = min_prob_in(boxes, b_tree, b_probs, 0.0) if b_tree is not None else [(None, None)] * len(walls)
    for w, (lo, mean) in zip(walls, wconf):
        w["source"] = "cv_model"
        w["cv_conf_min"], w["cv_conf_mean"] = lo, mean
        w["confidence"] = round(lo if lo is not None else 0.5, 4)
    by_id = {o["id"]: o for o in r["openings"]}
    for o in manifest["openings"]:
        src = by_id[o["id"]]
        o["source"] = "cv_model"
        o["cv_class"], o["cv_confidence"], o["cv_instance"] = src["cv_class"], round(src["cv_confidence"], 4), src["cv_instance"]
        o["confidence"] = round(min(o["confidence"], src["cv_confidence"]), 4)
    names = np.array(TARGET_CLASSES)
    symbols = []
    for s in range(len(scene["seg_cls"])):
        if names[scene["seg_cls"][s]] in ("stairs", "elevator"):
            m = np.where(scene["seg"] == s)[0]
            xy = Pw["pts"][m].reshape(-1, 2)
            symbols.append({"class": str(names[scene["seg_cls"][s]]), "bbox": [round(float(v), 1) for v in (*xy.min(0), *xy.max(0))],
                            "confidence": round(float(prob[m].min()), 4), "n_prims": int(len(m))})
    manifest["summary"]["cv"] = {
        "head2_threshold": scene["thr"],
        "primitives": int(len(prob)),
        "uncovered_primitives": int((~scene["covered"]).sum()),
        "class_counts": {n: int(c) for n, c in zip(TARGET_CLASSES, np.bincount(scene["cls"], minlength=len(TARGET_CLASSES)))},
        "opening_instances": {"accepted": sum(i["accepted"] for i in instances), "rejected_size": sum(not i["accepted"] for i in instances)},
        "openings_in_plan": len(r["openings"]),
        "text_items": len(tags),
        "symbols": symbols,
    }