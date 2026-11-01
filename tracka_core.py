import re
import json
import math
import time
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import ezdxf
from ezdxf import path as ezpath
from ezdxf.math import Matrix44, Vec3
from shapely import STRtree
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import shapely

CONFIG = {
    "floor_block": None,
    "boundary_layers": {"A-WALL", "A-COL", "A-CW", "A-LOW WALL", "A-RET-WALL", "A-BALCONY", "A-RAIL"},
    "wall_layers": {"A-WALL"},
    "door_layers": {"A-DOOR", "A-DOOR-INT", "N-DOOR"},
    "window_layers": {"A-WIN"},
    "room_tag_layers": {"A-RD"},
    "room_tag_name_attr": "RT",
    "room_tag_size_attr": "SZ",
    "room_text_layers": set(),
    "opening_tag_layers": {"A-TAG-DOOR"},
    "level_tag_attrs": {("LVLNAME", "LVLVALUE"), ("LVL", "%%P0'0\"")},
    "unit_label_layers": {"A-UNIT NO"},
    "annotation_label_regex": r"^(UP|DN|DOWN|RAMP( UP| DN| DOWN)?|\d+\s*PASS\.?)$|CUT-OUT|DOUBLE HT|SLAB|RCC ROOF|^@|JALI|BOXING|^(FHC|ELE|COM)-?\d+",
    "open_space_closing": 2000.0,
    "split_gap_range": (450.0, 2600.0),
    "split_angle_tol_deg": 5.0,
    "corridor_label_regex": r"CORRIDOR|PASSAGE|LOBBY",
    "circulation_label_regex": r"STAIR\w*|FIRE TOWER|FIRE LIFT|LIFT|RAMP",
    "default_insunits": 4,
    "snap_tol": 15.0,
    "flatten_tol": 5.0,
    "min_hole_area": 50_000.0,
    "void_area": 1_000_000.0,
    "void_width": 300.0,
    "wall_thickness_range": (50.0, 450.0),
    "wall_min_overlap": 100.0,
    "opening_search_pad": 60.0,
    "door_probe_offsets": (250.0, 450.0, 700.0, 1000.0),
    "adjacency_gap": 450.0,
    "adjacency_min_shared": 300.0,
    "opening_tag_radius": 1500.0,
    "dimension_tol": 25.0,
    "size_check_tol_pct": 5.0,
}

UNIT_TO_METRE = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}
UNIT_NAME = {0: "unitless", 1: "inch", 2: "foot", 4: "mm", 5: "cm", 6: "m", 14: "dm"}

def norm_layer(name):
    return name.split("$0$")[-1].split("|")[-1].strip().upper()


def pick_floor_insert(doc, block_name=None):
    inserts = list(doc.modelspace().query("INSERT"))
    if block_name:
        return next(i for i in inserts if i.dxf.name == block_name)
    return max(inserts, key=lambda i: len(doc.blocks[i.dxf.name]))


def short_block(name):
    return name.split("$0$")[-1]


@dataclass
class Leaf:
    dxftype: str
    layer: str
    owner_layer: str
    owner_block: str
    points: list
    closed: bool
    radius: float = 0.0


@dataclass
class InsertRec:
    name: str
    layer: str
    owner_layer: str
    matrix: Matrix44
    insert: tuple
    scale: tuple
    rotation: float
    attribs: dict
    depth: int


@dataclass
class Exploded:
    leaves: list = field(default_factory=list)
    inserts: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)


def to_points(entity, tol):
    try:
        p = ezpath.make_path(entity)
    except Exception:
        return None
    pts = [(v.x, v.y) for v in p.flattening(tol)]
    return pts if len(pts) >= 2 else None


def explode(entities, out, cfg, parent_layer="0", owner_layer="", owner_block="", depth=0):
    for e in entities:
        t = e.dxftype()
        layer = norm_layer(e.dxf.layer)
        if layer == "0":
            layer = parent_layer
        if t == "INSERT":
            attribs = {a.dxf.tag: a.dxf.text for a in e.attribs}
            out.inserts.append(InsertRec(
                short_block(e.dxf.name), layer, owner_layer, e.matrix44(),
                tuple(e.dxf.insert)[:2], (e.dxf.xscale, e.dxf.yscale), e.dxf.rotation, attribs, depth))
            try:
                children = list(e.virtual_entities())
            except Exception:
                continue
            explode(children, out, cfg, layer, owner_layer or layer, owner_block or short_block(e.dxf.name), depth + 1)
        elif t in ("TEXT", "MTEXT"):
            text = e.plain_text() if t == "MTEXT" else e.dxf.text
            out.texts.append({"text": text.strip(), "layer": layer, "owner_layer": owner_layer,
                              "insert": tuple(e.dxf.insert)[:2]})
        elif t == "DIMENSION":
            out.dimensions.append((e, layer))
        elif t in ("LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"):
            pts = to_points(e, cfg["flatten_tol"])
            if pts is None:
                continue
            closed = t == "CIRCLE" or (t in ("LWPOLYLINE", "POLYLINE") and e.closed)
            radius = e.dxf.radius if t in ("ARC", "CIRCLE") else 0.0
            out.leaves.append(Leaf(t, layer, owner_layer, owner_block, pts, closed, radius))


def load_floor(path, cfg):
    doc = ezdxf.readfile(path)
    top = pick_floor_insert(doc, cfg["floor_block"])
    scale = resolve_scale(doc, top, cfg)
    k = scale["to_metre"] * 1000.0
    entities = list(top.virtual_entities())
    if not math.isclose(k, 1.0):
        m = Matrix44.scale(k, k, k)
        for e in entities:
            try:
                e.transform(m)
            except Exception:
                scale["issues"].append(f"could not rescale {e.dxftype()}")
    scale["normalised_to_mm_factor"] = k
    out = Exploded()
    explode(entities, out, cfg)
    return doc, top, out, scale

def parse_nominal_width(name):
    m = re.search(r"-\s*(\d{3,4})(?:\s*X\s*(\d{3,4}))?", name.upper())
    if not m:
        return None, None
    w = float(m.group(1))
    h = float(m.group(2)) if m.group(2) else None
    return (w if 400 <= w <= 6000 else None), h


def parse_size(sz):
    s = (sz or "").upper()
    m = re.search(r"(\d{3,5})\s*[X\*]\s*(\d{3,5})", s)
    if m:
        return {"dims": sorted([float(m.group(1)), float(m.group(2))])}
    m = re.search(r"(\d{3,5})\s*M{0,2}\s*WIDE", s)
    if m:
        return {"width": float(m.group(1))}
    return None


def resolve_scale(doc, top, cfg):
    insunits = doc.header.get("$INSUNITS", 0) or cfg["default_insunits"]
    block_units = doc.blocks[top.dxf.name].block_record.dxf.get("units", 0)
    sx, sy = top.dxf.xscale, top.dxf.yscale
    issues = []
    if not math.isclose(abs(sx), abs(sy), rel_tol=1e-6):
        issues.append(f"non-uniform floor insert scale {sx}x{sy}")
    if block_units and block_units != insunits:
        issues.append(f"block units {block_units} differ from drawing units {insunits}")
    return {
        "insunits": insunits,
        "insunits_source": "header" if doc.header.get("$INSUNITS", 0) else "config_default",
        "unit_name": UNIT_NAME.get(insunits, str(insunits)),
        "to_metre": UNIT_TO_METRE.get(insunits, 1.0),
        "floor_insert_scale": [sx, sy],
        "block_units": block_units,
        "dimlfac_header": doc.header.get("$DIMLFAC", 1.0),
        "issues": issues,
        "cross_checks": {},
    }

def leaf_line(leaf):
    pts = leaf.points + ([leaf.points[0]] if leaf.closed else [])
    return LineString(pts)


def classify_leaves(ex, cfg):
    opening_layers = cfg["door_layers"] | cfg["window_layers"]
    walls, boundary, windows = [], [], []
    for lf in ex.leaves:
        if lf.owner_layer in opening_layers:
            if lf.owner_layer in cfg["window_layers"]:
                windows.append(lf)
            continue
        if lf.layer in cfg["boundary_layers"] or lf.owner_layer in cfg["boundary_layers"]:
            boundary.append(lf)
            if lf.layer in cfg["wall_layers"]:
                walls.append(lf)
    return walls, boundary, windows


def segments_of(leaves):
    segs = []
    for lf in leaves:
        pts = lf.points + ([lf.points[0]] if lf.closed else [])
        for a, b in zip(pts[:-1], pts[1:]):
            if math.dist(a, b) > 1e-6:
                segs.append((a[0], a[1], b[0], b[1]))
    return np.array(segs, dtype=float).reshape(-1, 4)

def block_info(doc, name, cache):
    if name in cache:
        return cache[name]
    blk = next((b for b in doc.blocks if short_block(b.name) == name), None)
    xs, ys, radii = [], [], []
    if blk is not None:
        for e in blk:
            if e.dxftype() == "INSERT":
                try:
                    subs = list(e.virtual_entities())
                except Exception:
                    subs = []
            else:
                subs = [e]
            for s in subs:
                if s.dxftype() == "ARC":
                    radii.append(s.dxf.radius)
                pts = to_points(s, 5.0) if s.dxftype() not in ("ATTDEF", "TEXT", "MTEXT", "HATCH", "INSERT") else None
                if pts:
                    xs += [p[0] for p in pts]
                    ys += [p[1] for p in pts]
    info = {"bbox": (min(xs), min(ys), max(xs), max(ys)) if xs else None, "arc_radii": sorted(round(r, 1) for r in radii)}
    cache[name] = info
    return info


def opening_kind(rec, cfg):
    if rec.layer in cfg["door_layers"]:
        return "door"
    if rec.layer in cfg["window_layers"]:
        return "window"
    return None


def opening_subtype(name):
    m = re.match(r"\s*([A-Z]+\d?)", name.upper())
    return m.group(1) if m else name


def transform_pts(matrix, pts):
    return [(v.x, v.y) for v in matrix.transform_vertices([Vec3(x, y, 0) for x, y in pts])]


def build_openings(doc, ex, wall_segs, cfg):
    cache = {}
    endpoints = np.vstack([wall_segs[:, :2], wall_segs[:, 2:]]) if len(wall_segs) else np.zeros((0, 2))
    ep_tree = STRtree([Point(p) for p in endpoints])
    openings = []
    for rec in ex.inserts:
        kind = opening_kind(rec, cfg)
        if kind is None or rec.owner_layer:
            continue
        info = block_info(doc, rec.name, cache)
        if info["bbox"] is None:
            continue
        x0, y0, x1, y1 = info["bbox"]
        corners = transform_pts(rec.matrix, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        origin, ux_pt, uy_pt = transform_pts(rec.matrix, [(0, 0), (1, 0), (0, 1)])
        u = np.subtract(ux_pt, origin)
        sx = float(np.linalg.norm(u))
        u = u / sx
        v = np.subtract(uy_pt, origin)
        v = v / np.linalg.norm(v)
        obb = Polygon(corners)
        pad = obb.buffer(cfg["opening_search_pad"], join_style=2)
        idx = ep_tree.query(pad, predicate="contains")
        jambs = np.unique(np.round(endpoints[idx], 1), axis=0) if len(idx) else np.zeros((0, 2))
        nominal_w, nominal_h = parse_nominal_width(rec.name)
        block_w = (x1 - x0) * sx
        leaf_w = max(info["arc_radii"]) * sx if info["arc_radii"] else None
        big = [r for r in info["arc_radii"] if r >= 0.8 * max(info["arc_radii"])] if info["arc_radii"] else []
        double_leaf = len(big) >= 2 and abs(2 * big[-1] * sx - block_w) <= 0.15 * block_w
        if double_leaf:
            leaf_w = 2 * big[-1] * sx
        clear_w, plug, centre = None, None, None
        if len(jambs) >= 2:
            proj = jambs @ u
            off = jambs @ v
            split = (proj.max() + proj.min()) / 2
            left, right = proj <= split, proj > split
            if left.any() and right.any():
                clear_w = float(proj[right].min() - proj[left].max())
                bridges = []
                for li in np.where(left)[0]:
                    ri = np.where(right)[0][np.argmin(np.abs(off[right] - off[li]))]
                    bridges.append(LineString([tuple(jambs[li]), tuple(jambs[ri])]))
                if clear_w > 0.4 * block_w:
                    plug = unary_union(bridges).buffer(cfg["snap_tol"], cap_style="flat")
                    centre = shapely.MultiPoint([tuple(p) for p in jambs]).centroid
        ref_w = nominal_w or leaf_w or block_w
        if clear_w is not None and not (0.5 * ref_w <= clear_w <= 1.6 * ref_w):
            clear_w = None
        if plug is None:
            edge = LineString(corners[:2])
            plug = edge.buffer(cfg["snap_tol"], cap_style=2)
            centre = edge.interpolate(0.5, normalized=True)
        width = clear_w or leaf_w or nominal_w or block_w
        method = "jamb_clear_opening" if clear_w else ("leaf_arc_radius" if leaf_w else ("block_name" if nominal_w else "block_extent"))
        openings.append({
            "kind": kind, "block": rec.name, "layer": rec.layer, "subtype": opening_subtype(rec.name),
            "position": tuple(map(float, centre.coords[0])), "insert": rec.insert, "rotation": rec.rotation,
            "mirrored": rec.scale[0] * rec.scale[1] < 0, "axis": u.tolist(), "normal": v.tolist(),
            "width_mm": width, "width_method": method, "width_clear_mm": clear_w, "width_leaf_mm": leaf_w,
            "width_block_mm": block_w, "width_nominal_mm": nominal_w, "height_nominal_mm": nominal_h,
            "double_leaf": double_leaf,
            "jamb_points": len(jambs), "_plug": plug, "_obb": obb,
        })
    return openings

def room_tags(ex, cfg):
    tags = []
    ann = re.compile(cfg["annotation_label_regex"], re.I)
    for rec in ex.inserts:
        if rec.layer not in cfg["room_tag_layers"] or not rec.attribs:
            continue
        rt = rec.attribs.get(cfg["room_tag_name_attr"], "").strip()
        sz = rec.attribs.get(cfg["room_tag_size_attr"], "").strip()
        size = parse_size(sz)
        label = rt or (sz if size is None else "")
        if not label:
            continue
        tags.append({"raw_label": label, "raw_size": sz, "nominal": size, "position": rec.insert,
                     "annotation": bool(ann.search(label))})
    for t in ex.texts:
        if t["layer"] in cfg["room_text_layers"] and t["text"]:
            tags.append({"raw_label": t["text"], "raw_size": "", "nominal": None, "position": t["insert"],
                         "annotation": bool(ann.search(t["text"]))})
    return tags

def build_solid(boundary, windows, openings, cfg):
    tol = cfg["snap_tol"]
    lines = [leaf_line(lf) for lf in boundary + windows]
    pieces = shapely.buffer(np.array(lines, dtype=object), tol, quad_segs=1, cap_style="square")
    plugs = [o["_plug"] for o in openings]
    return unary_union(list(pieces) + plugs)


def as_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


def extract_spaces(solid, cfg):
    comps = as_polygons(solid)
    comp_tree = STRtree(comps)
    spaces = []
    for ci, comp in enumerate(comps):
        for ring in comp.interiors:
            hole = Polygon(ring)
            if hole.area < cfg["min_hole_area"]:
                continue
            inner = [comps[j] for j in comp_tree.query(hole, predicate="contains")]
            shape = hole.difference(unary_union(inner)) if inner else hole
            shape = shape.buffer(cfg["snap_tol"], join_style=2).intersection(hole.buffer(cfg["snap_tol"], join_style=2))
            for poly in as_polygons(shape):
                if poly.area >= cfg["min_hole_area"]:
                    spaces.append({"component": ci, "geom": poly})
    return comps, spaces

def clear_width(poly, iters=14):
    lo, hi = 0.0, inscribed_width(poly) + 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        g = poly.buffer(-mid / 2, join_style=2)
        if not g.is_empty and g.geom_type == "Polygon":
            lo = mid
        else:
            hi = mid
    return lo


def inscribed_width(poly):
    if hasattr(shapely, "maximum_inscribed_circle"):
        return 2 * shapely.maximum_inscribed_circle(poly, 1.0).length
    return 2 * shapely.polylabel(poly, 1.0).distance(poly.exterior)


def mrr_dims(poly):
    r = poly.minimum_rotated_rectangle
    c = list(r.exterior.coords)
    return sorted([math.dist(c[0], c[1]), math.dist(c[1], c[2])])

def assign_tags(spaces, tags, comps):
    tree = STRtree([s["geom"] for s in spaces])
    for s in spaces:
        s["tags"], s["notes"] = [], []
    unassigned = []
    for t in tags:
        p = Point(t["position"])
        hit = [i for i in tree.query(p, predicate="within")]
        if not hit:
            unassigned.append(t)
            continue
        i = min(hit, key=lambda k: spaces[k]["geom"].area)
        (spaces[i]["notes"] if t["annotation"] else spaces[i]["tags"]).append(t)
    return unassigned


def size_check(space, dims, area_m2, to_m, cfg):
    noms = [t["nominal"] for t in space["tags"] if t["nominal"]]
    if len(space["tags"]) == 1 and noms and "dims" in noms[0]:
        nd = noms[0]["dims"]
        delta = [100 * (m - n) / n for m, n in zip(dims, nd)]
        nominal_area = nd[0] * nd[1] * to_m * to_m
        return {"nominal_dims_mm": nd, "measured_dims_mm": [round(d) for d in dims],
                "delta_pct": [round(d, 2) for d in delta],
                "nominal_area_m2": round(nominal_area, 3),
                "area_delta_pct": round(100 * (area_m2 - nominal_area) / nominal_area, 2),
                "within_tol": all(abs(d) <= cfg["size_check_tol_pct"] for d in delta)}
    if noms and all("dims" in n for n in noms):
        nominal_area = sum(n["dims"][0] * n["dims"][1] for n in noms) * to_m * to_m
        return {"nominal_area_m2": round(nominal_area, 3),
                "area_delta_pct": round(100 * (area_m2 - nominal_area) / nominal_area, 2),
                "merged_tags": len(space["tags"]), "within_tol": None}
    return None


def describe_spaces(spaces, to_m, cfg):
    corr = re.compile(cfg["corridor_label_regex"], re.I)
    for i, s in enumerate(spaces):
        g = s["geom"]
        dims = mrr_dims(g)
        area_m2 = g.area * to_m * to_m
        cw, iw = clear_width(g), inscribed_width(g)
        labels = [t["raw_label"] for t in s["tags"]]
        if cw < cfg["void_width"] and not labels:
            status = "void"
        elif labels:
            status = "room"
        elif g.area < cfg["void_area"]:
            status = "void"
        else:
            status = "unlabelled"
        widths = [t["nominal"]["width"] for t in s["tags"] if t["nominal"] and "width" in t["nominal"]]
        s.update({
            "id": f"R{i:03d}", "status": status, "labels": labels,
            "raw_label": " / ".join(labels) if labels else None,
            "is_corridor": any(corr.search(l) for l in labels),
            "area_m2": round(area_m2, 3), "perimeter_m": round(g.length * to_m, 3),
            "mrr_dims_mm": [round(d) for d in dims], "inscribed_width_mm": round(iw), "clear_width_mm": round(cw),
            "nominal_width_mm": widths[0] if widths else None,
            "size_check": size_check(s, dims, area_m2, to_m, cfg),
        })
        if s["nominal_width_mm"]:
            s["width_check"] = {"nominal_mm": s["nominal_width_mm"], "clear_mm": round(cw), "inscribed_mm": round(iw),
                                "delta_pct": round(100 * (cw - s["nominal_width_mm"]) / s["nominal_width_mm"], 2)}
        s.setdefault("enclosure", "closed")
        conf = 1.0
        if status != "room":
            conf = 0.6
        elif s["enclosure"] == "open":
            conf = 0.7
        elif s["enclosure"] == "split":
            conf = 0.85
        elif s["size_check"] and s["size_check"].get("within_tol") is False:
            conf = 0.85
        s["confidence"] = conf

def open_spaces(comps, plan_comps, tags, cfg):
    g = cfg["open_space_closing"]
    filled = unary_union([Polygon(comps[c].exterior) for c in plan_comps])
    closed = filled.buffer(g, join_style=2).buffer(-g, join_style=2)
    tol = cfg["snap_tol"]
    pockets = as_polygons(closed.difference(filled).buffer(-1).buffer(1))
    found = []
    for p in pockets:
        p = p.buffer(tol, join_style=2).intersection(closed)
        p = max(as_polygons(p), key=lambda q: q.area) if as_polygons(p) else p
        if p.area >= cfg["min_hole_area"] and any(p.contains(Point(t["position"])) for t in tags if not t["annotation"]):
            found.append({"component": -1, "geom": p, "enclosure": "open"})
    return found


def endpoint_dirs(segs):
    a, b = segs[:, :2], segs[:, 2:]
    d = b - a
    n = np.hypot(d[:, 0], d[:, 1])[:, None]
    d = d / n
    return np.vstack([a, b]), np.vstack([-d, d])


def gap_cuts(geom, pts, dirs, cfg):
    lo, hi = cfg["split_gap_range"]
    cos_tol = math.cos(math.radians(cfg["split_angle_tol_deg"]))
    x0, y0, x1, y1 = geom.bounds
    m = (pts[:, 0] >= x0 - 50) & (pts[:, 0] <= x1 + 50) & (pts[:, 1] >= y0 - 50) & (pts[:, 1] <= y1 + 50)
    P, D = pts[m], dirs[m]
    if len(P) < 2:
        return []
    near = shapely.distance(geom.boundary, shapely.points(P)) <= 2 * cfg["snap_tol"] + 1
    P, D = P[near], D[near]
    PD, keep = np.unique(np.round(np.hstack([P, D * 1000]), 1), axis=0, return_index=True)
    P, D = P[keep], D[keep]
    cuts = []
    for i in range(len(P)):
        v = P - P[i]
        L = np.hypot(v[:, 0], v[:, 1])
        ok = (L >= lo) & (L <= hi)
        for j in np.where(ok)[0]:
            if j <= i or np.allclose(P[i], P[j]):
                continue
            w = v[j] / L[j]
            if D[i] @ w >= cos_tol and D[j] @ -w >= cos_tol:
                line = LineString([tuple(P[i]), tuple(P[j])])
                if geom.contains(line.interpolate(0.5, normalized=True)):
                    cuts.append(line)
    return sorted(cuts, key=lambda c: c.length)


def extend(line, d):
    (x0, y0), (x1, y1) = line.coords
    L = line.length
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    return LineString([(x0 - ux * d, y0 - uy * d), (x1 + ux * d, y1 + uy * d)])


def tag_count(poly, tags):
    return sum(poly.contains(Point(t["position"])) for t in tags)


def split_space(space, pts, dirs, cfg):
    from shapely.ops import split as shp_split
    tags = space["tags"]
    if len({tuple(t["position"]) for t in tags}) < 2:
        return [space]
    faces, used = [space["geom"]], []
    for cut in gap_cuts(space["geom"], pts, dirs, cfg):
        mid = cut.interpolate(0.5, normalized=True)
        for fi, f in enumerate(faces):
            if not f.contains(mid):
                continue
            parts = as_polygons(shp_split(f, extend(cut, 3 * cfg["snap_tol"])))
            if len(parts) >= 2 and sum(tag_count(p, tags) > 0 for p in parts) >= 2:
                faces = faces[:fi] + parts + faces[fi + 1:]
                used.append(cut)
            break
    if not used:
        return [space]
    tagged = [f for f in faces if tag_count(f, tags)]
    for f in faces:
        if not tag_count(f, tags):
            k = max(range(len(tagged)), key=lambda i: tagged[i].boundary.intersection(f.buffer(1)).length)
            tagged[k] = tagged[k].union(f)
    out = []
    for f in tagged:
        sub = dict(space)
        sub.update({"geom": f, "enclosure": "split" if space.get("enclosure", "closed") == "closed" else space["enclosure"],
                    "tags": [t for t in tags if f.buffer(1).contains(Point(t["position"]))],
                    "notes": [t for t in space["notes"] if f.buffer(1).contains(Point(t["position"]))],
                    "split_lines": [[[round(x, 1), round(y, 1)] for x, y in c.coords] for c in used if c.buffer(5).intersects(f.boundary)]})
        out.append(sub)
    return out

def connect_openings(openings, spaces, comps, cfg):
    spaces = [s for s in spaces if s["status"] != "void"]
    tree = STRtree([s["geom"] for s in spaces])
    comp_tree = STRtree([Polygon(c.exterior) for c in comps])

    def side(c, n, sign):
        for d in cfg["door_probe_offsets"]:
            p = Point(c[0] + sign * n[0] * d, c[1] + sign * n[1] * d)
            hit = list(tree.query(p, predicate="within"))
            if hit:
                return spaces[min(hit, key=lambda k: spaces[k]["geom"].area)]["id"]
        p = Point(c[0] + sign * n[0] * cfg["door_probe_offsets"][-1], c[1] + sign * n[1] * cfg["door_probe_offsets"][-1])
        return "EXTERIOR" if not len(comp_tree.query(p, predicate="within")) else None

    for o in openings:
        a = side(o["position"], o["normal"], +1)
        b = side(o["position"], o["normal"], -1)
        o["connects"] = [a, b]
        o["self_loop"] = a is not None and a == b
        o["is_external"] = "EXTERIOR" in (a, b)
        o["in_plan"] = len(comp_tree.query(Point(o["position"]).buffer(cfg["opening_search_pad"]), predicate="intersects")) > 0


def attach_opening_tags(openings, ex, cfg):
    tags = [r for r in ex.inserts if r.layer in cfg["opening_tag_layers"] and r.attribs]
    if not openings or not tags:
        return
    pts = np.array([o["position"] for o in openings])
    best = {}
    for r in tags:
        d = np.hypot(*(pts - np.array(r.insert)).T)
        k = int(d.argmin())
        if d[k] <= cfg["opening_tag_radius"] and (k not in best or d[k] < best[k][0]):
            best[k] = (d[k], next(iter(r.attribs.values())))
    for k, (d, code) in best.items():
        openings[k]["type_code"] = code


def build_adjacency(spaces, openings, cfg):
    rooms = [s for s in spaces if s["status"] != "void"]
    geoms = [s["geom"] for s in rooms]
    tree = STRtree(geoms)
    edges = {}
    gap = cfg["adjacency_gap"]
    for i, s in enumerate(rooms):
        grown = s["geom"].buffer(gap, join_style=2)
        for j in tree.query(grown, predicate="intersects"):
            if j <= i:
                continue
            shared = rooms[j]["geom"].boundary.intersection(grown).length
            if shared >= cfg["adjacency_min_shared"]:
                edges[(s["id"], rooms[j]["id"])] = {"a": s["id"], "b": rooms[j]["id"], "shared_wall_mm": round(shared), "doors": [], "windows": []}
    for o in openings:
        a, b = o["connects"]
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        e = edges.setdefault(key, {"a": key[0], "b": key[1], "shared_wall_mm": None, "doors": [], "windows": []})
        e["doors" if o["kind"] == "door" else "windows"].append(o["id"])
    tree_all = STRtree(geoms)
    seen = set()
    for s in rooms:
        for line in s.get("split_lines", []):
            key = tuple(map(tuple, line))
            if key in seen:
                continue
            seen.add(key)
            ls = LineString(line)
            (x0, y0), (x1, y1) = line
            n = np.array([y0 - y1, x1 - x0]) / max(ls.length, 1)
            mid = np.array(ls.interpolate(0.5, normalized=True).coords[0])
            side = []
            for sign in (1, -1):
                p = Point(*(mid + sign * n * 80))
                hit = list(tree_all.query(p, predicate="within"))
                side.append(rooms[hit[0]]["id"] if hit else None)
            if None in side or side[0] == side[1]:
                continue
            k2 = tuple(sorted(side))
            e = edges.setdefault(k2, {"a": k2[0], "b": k2[1], "shared_wall_mm": None, "doors": [], "windows": []})
            e.setdefault("passages", []).append({"width_mm": round(ls.length), "line": line})
    for e in edges.values():
        e.setdefault("passages", [])
    for s in spaces:
        s["adjacent"] = sorted({e["b"] if e["a"] == s["id"] else e["a"] for e in edges.values() if s["id"] in (e["a"], e["b"])})
    return list(edges.values())

def pair_walls(segs, cfg):
    tmin, tmax = cfg["wall_thickness_range"]
    if not len(segs):
        return []
    d = segs[:, 2:] - segs[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > cfg["wall_min_overlap"]
    segs, d, L = segs[keep], d[keep], L[keep]
    u = d / L[:, None]
    ang = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 180
    bucket = np.round(ang).astype(int) % 180
    walls, used = [], set()
    for b in np.unique(bucket):
        idx = np.where((bucket == b) | (bucket == (b + 1) % 180) | (bucket == (b - 1) % 180))[0]
        own = idx[bucket[idx] == b]
        for i in own:
            ui, ni = u[i], np.array([-u[i][1], u[i][0]])
            p0 = segs[i, :2]
            off_a = (segs[idx, :2] - p0) @ ni
            off_b = (segs[idx, 2:] - p0) @ ni
            off = (off_a + off_b) / 2
            t0 = (segs[idx, :2] - p0) @ ui
            t1 = (segs[idx, 2:] - p0) @ ui
            lo = np.maximum(0, np.minimum(t0, t1))
            hi = np.minimum(L[i], np.maximum(t0, t1))
            ov = hi - lo
            ok = (np.abs(off) >= tmin) & (np.abs(off) <= tmax) & (ov >= cfg["wall_min_overlap"]) & (np.abs(off_a - off_b) < 5)
            if not ok.any():
                continue
            cand = np.where(ok)[0]
            k = cand[np.argmin(np.abs(off[cand]))]
            j = idx[k]
            key = (min(i, j), max(i, j))
            if key in used:
                continue
            used.add(key)
            t = abs(off[k])
            a = p0 + ui * lo[k] + ni * off[k] / 2
            c = p0 + ui * hi[k] + ni * off[k] / 2
            walls.append({"start": [round(a[0], 1), round(a[1], 1)], "end": [round(c[0], 1), round(c[1], 1)],
                          "thickness_mm": round(t, 1), "length_mm": round(float(ov[k]), 1)})
    return walls

def check_dimensions(ex, spaces, cfg):
    tree = STRtree([s["geom"] for s in spaces])
    out = []
    for e, layer in ex.dimensions:
        rec = {"layer": layer, "dimtype": e.dimtype}
        try:
            m = float(e.get_measurement())
        except Exception:
            continue
        if not isinstance(m, float):
            continue
        txt = (e.dxf.get("text", "") or "").strip()
        try:
            lfac = float(e.override().get("dimlfac", 1.0))
        except Exception:
            lfac = 1.0
        rec.update({"measurement_mm": round(m, 1), "dimlfac": lfac, "text_override": txt or None})
        num = re.findall(r"[\d.]+", txt.replace("<>", ""))
        if txt and txt not in ("<>",) and num:
            try:
                rec["override_delta_mm"] = round(float(num[0]) - m * lfac, 1)
            except ValueError:
                pass
        if e.dimtype in (0, 1) and m > 0:
            p2 = np.array(e.dxf.defpoint2)[:2]
            p3 = np.array(e.dxf.defpoint3)[:2]
            if e.dimtype == 0:
                a = math.radians(e.dxf.get("angle", 0.0))
                u = np.array([math.cos(a), math.sin(a)])
            else:
                u = (p3 - p2) / (np.linalg.norm(p3 - p2) or 1)
            mid = (p2 + p3) / 2
            t2, t3 = sorted([float(p2 @ u), float(p3 @ u)])
            for k in tree.query(Point(mid), predicate="within"):
                g = spaces[k]["geom"]
                o2, o3 = g.boundary.distance(Point(p2)), g.boundary.distance(Point(p3))
                if max(o2, o3) > cfg["dimension_tol"]:
                    continue
                ray = LineString([mid - u * 1e6, mid + u * 1e6]).intersection(g)
                pieces = [ray] if ray.geom_type == "LineString" else list(getattr(ray, "geoms", []))
                pieces = [q for q in pieces if q.geom_type == "LineString" and not q.is_empty]
                if not pieces:
                    continue
                piece = min(pieces, key=lambda q: q.distance(Point(mid)))
                ends = sorted(float(np.array(c) @ u) for c in piece.coords)
                spans_room = abs(ends[0] - t2) <= cfg["dimension_tol"] and abs(ends[-1] - t3) <= cfg["dimension_tol"]
                rec.update({"room": spaces[k]["id"], "endpoint_offsets_mm": [round(o2, 1), round(o3, 1)],
                            "spans_room": spans_room})
                if spans_room:
                    rec.update({"computed_span_mm": round(piece.length, 1),
                                "delta_mm": round(piece.length - m * lfac, 1)})
                break
        out.append(rec)
    return out

def scale_cross_checks(scale, spaces, openings, dims, cfg):
    tol = cfg["size_check_tol_pct"]
    door = [100 * (o["width_block_mm"] - o["width_nominal_mm"]) / o["width_nominal_mm"]
            for o in openings if o["width_nominal_mm"]]
    rooms = [s["size_check"]["area_delta_pct"] for s in spaces if s.get("size_check") and s["size_check"].get("within_tol") is not None]
    dm = [d["delta_mm"] for d in dims if d.get("spans_room")]
    overrides = [d for d in dims if d.get("override_delta_mm") not in (None, 0)]
    lfacs = sorted({d["dimlfac"] for d in dims})
    checks = {
        "opening_block_vs_name": {"n": len(door), "median_delta_pct": round(float(np.median(door)), 2) if door else None,
                                  "within_tol": int(sum(abs(d) <= tol for d in door))},
        "room_area_vs_tag_size": {"n": len(rooms), "median_delta_pct": round(float(np.median(rooms)), 2) if rooms else None,
                                  "within_tol": int(sum(abs(d) <= tol for d in rooms))},
        "dimension_vs_geometry": {"n": len(dm), "max_abs_delta_mm": round(max(map(abs, dm)), 1) if dm else None,
                                  "within_tol": int(sum(abs(d) <= cfg["dimension_tol"] for d in dm))},
        "dimension_text_overrides": len(overrides),
        "dimlfac_values": lfacs,
    }
    medians = [c["median_delta_pct"] for c in (checks["opening_block_vs_name"], checks["room_area_vs_tag_size"]) if c["median_delta_pct"] is not None]
    ok = all(abs(m) <= tol for m in medians) and (not dm or checks["dimension_vs_geometry"]["within_tol"] == len(dm)) and lfacs in ([], [1.0])
    if any(abs(m) > 50 for m in medians):
        scale["issues"].append("median size deltas exceed 50%: probable unit mismatch")
    scale["cross_checks"] = checks
    scale["status"] = "pass" if ok and not scale["issues"] else "check"
    return scale

def levels_of(ex, cfg):
    lv = []
    for r in ex.inserts:
        for name_k, val_k in cfg["level_tag_attrs"]:
            if name_k in r.attribs and val_k in r.attribs:
                lv.append({"name": r.attribs[name_k], "value": r.attribs[val_k], "position": [round(c) for c in r.insert]})
    return lv


def unit_labels(ex, spaces, cfg):
    tree = STRtree([s["geom"] for s in spaces])
    out = []
    for t in ex.texts:
        if t["layer"] in cfg["unit_label_layers"] and t["text"]:
            hit = list(tree.query(Point(t["insert"]), predicate="within"))
            out.append({"label": t["text"], "position": [round(c) for c in t["insert"]],
                        "room": spaces[hit[0]]["id"] if hit else None})
    return out


def clean(poly):
    return shapely.remove_repeated_points(poly.simplify(0.5), 0.5)


def rnd_coords(poly):
    return [[round(x, 1), round(y, 1)] for x, y in clean(poly).exterior.coords]


def room_record(s, openings, edges):
    rec = {k: s[k] for k in ("id", "status", "enclosure", "raw_label", "labels", "is_corridor", "area_m2", "perimeter_m",
                             "mrr_dims_mm", "inscribed_width_mm", "clear_width_mm", "nominal_width_mm",
                             "size_check", "adjacent", "confidence")}
    rec["width_check"] = s.get("width_check")
    rec["building_component"] = s["component"]
    rec["centroid"] = [round(c, 1) for c in s["geom"].centroid.coords[0]]
    rec["bbox"] = [round(c, 1) for c in s["geom"].bounds]
    rec["polygon"] = rnd_coords(s["geom"])
    rec["holes"] = [[[round(x, 1), round(y, 1)] for x, y in r.coords] for r in clean(s["geom"]).interiors]
    rec["notes"] = [t["raw_label"] for t in s["notes"]]
    rec["raw_sizes"] = [t["raw_size"] for t in s["tags"] if t["raw_size"]]
    rec["doors"] = [o["id"] for o in openings if o["kind"] == "door" and s["id"] in o["connects"]]
    rec["open_passages"] = sorted({e["b"] if e["a"] == s["id"] else e["a"] for e in edges if e.get("passages") and s["id"] in (e["a"], e["b"])})
    rec["windows"] = [o["id"] for o in openings if o["kind"] == "window" and s["id"] in o["connects"]]
    rec["split_lines"] = s.get("split_lines", [])
    rec["source"] = "deterministic"
    return rec


def opening_record(o):
    keys = ("id", "kind", "subtype", "type_code", "block", "layer", "position", "rotation", "mirrored", "axis", "double_leaf", "self_loop",
            "width_mm", "width_method", "width_clear_mm", "width_leaf_mm", "width_block_mm", "width_nominal_mm",
            "height_nominal_mm", "connects", "is_external", "jamb_points", "confidence")
    rec = {k: o.get(k) for k in keys}
    for k in ("position", "axis"):
        rec[k] = [round(c, 4 if k == "axis" else 1) for c in rec[k]]
    for k in ("width_mm", "width_clear_mm", "width_leaf_mm", "width_block_mm"):
        if rec[k] is not None:
            rec[k] = round(rec[k], 1)
    rec["source"] = "deterministic"
    return rec

def corridor_summary(spaces):
    out = []
    for s in spaces:
        if s["is_corridor"]:
            g = s["geom"]
            out.append({"room": s["id"], "label": s["raw_label"], "enclosure": s["enclosure"],
                        "clear_width_mm": s["clear_width_mm"], "inscribed_width_mm": s["inscribed_width_mm"],
                        "clear_width_reliable": s["enclosure"] != "open",
                        "nominal_width_mm": s["nominal_width_mm"], "approx_length_mm": round(g.area / max(s["clear_width_mm"], 1)),
                        "area_m2": s["area_m2"]})
    return out


def circulation_summary(spaces, cfg):
    rx = re.compile(cfg["circulation_label_regex"], re.I)
    return [{"room": s["id"], "label": s["raw_label"], "kind": rx.search(s["raw_label"]).group(0).upper(),
             "area_m2": s["area_m2"], "centroid": [round(c, 1) for c in s["geom"].centroid.coords[0]]}
            for s in spaces if s["raw_label"] and rx.search(s["raw_label"])]

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict


class Rec(BaseModel):
    model_config = ConfigDict(extra="allow")


class RoomM(Rec):
    id: str
    status: Literal["room", "unlabelled"]
    enclosure: Literal["closed", "open", "split"]
    raw_label: Optional[str]
    area_m2: float
    polygon: list[list[float]]
    doors: list[str]
    windows: list[str]
    adjacent: list[str]
    source: Literal["deterministic", "cv_model"]
    confidence: float


class OpeningM(Rec):
    id: str
    kind: Literal["door", "window"]
    width_mm: float
    width_method: str
    connects: list[Optional[str]]
    is_external: bool
    source: Literal["deterministic", "cv_model"]
    confidence: float


class WallM(Rec):
    id: str
    start: list[float]
    end: list[float]
    thickness_mm: float
    source: Literal["deterministic", "cv_model"]
    confidence: float


class EdgeM(Rec):
    a: str
    b: str
    shared_wall_mm: Optional[float]
    doors: list[str]
    windows: list[str]


class ManifestM(Rec):
    schema_version: str
    source_file: str
    floor: dict
    units: dict
    scale_resolution: dict
    rooms: list[RoomM]
    openings: list[OpeningM]
    walls: list[WallM]
    adjacency: list[EdgeM]
    corridors: list[dict]
    vertical_circulation: list[dict]
    dimensions: list[dict]
    unassigned_labels: list[dict]
    summary: dict


def build_manifest(path, r, cfg):
    rooms = [s for s in r["spaces"] if s["status"] != "void"]
    room_ids = {s["id"] for s in rooms}
    for e in r["edges"]:
        e["doors"] = sorted(set(e["doors"]))
        e["windows"] = sorted(set(e["windows"]))
    manifest = {
        "schema_version": "0.2",
        "source_file": Path(path).name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "floor": {"block": r["top"].dxf.name, "insert": [round(c, 1) for c in tuple(r["top"].dxf.insert)[:2]],
                  "levels": levels_of(r["ex"], cfg), "unit_labels": unit_labels(r["ex"], rooms, cfg),
                  "building_components": len({s["component"] for s in rooms})},
        "units": {"length": "mm", "area": "m2", "coordinates": "WCS, normalised to mm"},
        "scale_resolution": r["scale"],
        "rooms": [room_record(s, r["openings"], r["edges"]) for s in rooms],
        "openings": [opening_record(o) for o in r["openings"]],
        "walls": r["walls"],
        "adjacency": [e for e in r["edges"] if e["a"] in room_ids | {"EXTERIOR"} and e["b"] in room_ids | {"EXTERIOR"}],
        "corridors": corridor_summary(rooms),
        "vertical_circulation": circulation_summary(rooms, cfg),
        "dimensions": r["dims"],
        "unassigned_labels": [{"raw_label": t["raw_label"], "raw_size": t["raw_size"], "position": [round(c, 1) for c in t["position"]]}
                              for t in r["unassigned"] if not t["annotation"]],
        "summary": {
            "rooms": sum(s["status"] == "room" for s in rooms),
            "unlabelled_spaces": sum(s["status"] == "unlabelled" for s in rooms),
            "voids_dropped": sum(s["status"] == "void" for s in r["spaces"]),
            "enclosure": dict(Counter(s["enclosure"] for s in rooms)),
            "doors": sum(o["kind"] == "door" for o in r["openings"]),
            "windows": sum(o["kind"] == "window" for o in r["openings"]),
            "external_doors": sum(o["kind"] == "door" and o["is_external"] for o in r["openings"]),
            "walls": len(r["walls"]),
            "wall_thickness_mm": dict(Counter(round(w["thickness_mm"]) for w in r["walls"]).most_common(6)),
            "adjacency_edges": len(r["edges"]),
            "unassigned_labels": sum(not t["annotation"] for t in r["unassigned"]),
            "elapsed_s": round(r["elapsed"], 1),
        },
    }
    ManifestM.model_validate(manifest)
    return manifest

def run(path, cfg=CONFIG, out_path=None):
    t0 = time.time()
    doc, top, ex, scale = load_floor(path, cfg)
    to_m = 0.001
    walls_l, boundary, windows = classify_leaves(ex, cfg)
    wall_segs = segments_of(walls_l)
    bnd_segs = segments_of(boundary)
    openings = build_openings(doc, ex, bnd_segs, cfg)
    solid = build_solid(boundary, windows, openings, cfg)
    comps, spaces = extract_spaces(solid, cfg)
    tags = room_tags(ex, cfg)
    unassigned = assign_tags(spaces, tags, comps)
    tagged_comps = {s["component"] for s in spaces if s["tags"]}
    plan_zone = unary_union([Polygon(comps[c].exterior) for c in tagged_comps]).buffer(5000)
    plan_comps = [i for i, c in enumerate(comps) if c.intersects(plan_zone)]
    pockets = open_spaces(comps, plan_comps, unassigned, cfg)
    spaces = [s for s in spaces if s["component"] in tagged_comps] + pockets
    unassigned = assign_tags(spaces, tags, comps)
    pts, dirs = endpoint_dirs(bnd_segs)
    spaces = [part for s in spaces for part in split_space(s, pts, dirs, cfg)]
    describe_spaces(spaces, to_m, cfg)
    connect_openings(openings, spaces, comps, cfg)
    openings = [o for o in openings if o["in_plan"]]
    for i, o in enumerate(openings):
        o["id"] = f"{o['kind'][0].upper()}{i:03d}"
        o["confidence"] = 0.5 if o["self_loop"] else (1.0 if o["width_method"] == "jamb_clear_opening" else 0.85)
    attach_opening_tags(openings, ex, cfg)
    edges = build_adjacency(spaces, openings, cfg)
    walls = [w for w in pair_walls(wall_segs, cfg) if plan_zone.contains(Point(w["start"]))]
    for i, w in enumerate(walls):
        w.update({"id": f"W{i:04d}", "source": "deterministic", "confidence": 1.0})
    dims = check_dimensions(ex, spaces, cfg)
    scale_cross_checks(scale, spaces, openings, dims, cfg)
    return dict(doc=doc, top=top, ex=ex, scale=scale, spaces=spaces, openings=openings, edges=edges,
                walls=walls, dims=dims, tags=tags, unassigned=unassigned, comps=comps, tagged_comps=tagged_comps,
                plan_comps=plan_comps, plan_zone=plan_zone,
                elapsed=time.time() - t0)