import re
import math
import json
import time
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from pathlib import Path as FsPath
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import ezdxf
from ezdxf import recover
from ezdxf import path as ezpath
from ezdxf.math import Vec3
import shapely
from shapely import STRtree
from shapely.geometry import LineString
from shapely.ops import unary_union

UNIT_TO_MM = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0, 14: 100.0}
UNIT_CANDIDATES = [1.0, 10.0, 25.4, 304.8, 1000.0]


@dataclass
class Rules:
    exclude: str = r"TEXT|TXT|DIM|TAG|ANNO|HATCH|\bHAT\b|PATT|SCHED|TITLE|GRID|AXIS|CENT(ER|RE)|LEADER|NOTE|SYMB|DEFPOINTS|VPORT|BORDER|FRAME|FURN|\bFUR\b|FUR-|FIXT|SANIT|LAND|TREE|PLANT|ELEC|PLUMB|DRAIN|HVAC|KEY\s*PLAN|LOCATION\s*PLAN|VICINITY"
    door: str = r"DOOR|(^|[-_ ])DR($|[-_ \d])|GATE|SHUTTER|^[A-Z]?D\d*\s*-\s*\d|^(MD|FD|SD|GD)\b|TOIDOR"
    window: str = r"WIN(DOW)?|VENT|LOUV|JALI|^[A-Z]?W\d*\s*-\s*\d|^(KW|BW|GW|SW|CW)\s*-?\s*\d|^V\s*-\s*\d"
    curtain: str = r"CURTAIN|GLAZ|(^|[-_ ])CW($|[-_ ])|GLASS"
    railing: str = r"RAIL|BALUSTR|GRILL"
    stair: str = r"STAIR|\bSTEPS?\b|STRCASE"
    lift: str = r"LIFT|ELEV|EVTR"
    column: str = r"(^|[^A-Z])COL(UMN|UMNS|S)?($|[^A-Z])|COLU"
    wall: str = r"WALL|BRICK|MASONRY|PARTITION|BLOCKWORK|MURO"
    bay: str = r"BAY|^BW"
    rolling: str = r"SHUTTER|ROLL"
    folding: str = r"FOLD"
    revolving: str = r"REVOLV"
    column_as: str = "wall"

    def compiled(self):
        return {k: re.compile(v, re.I) for k, v in asdict(self).items() if k != "column_as"}


CATEGORY_ORDER = ["door", "window", "curtain", "railing", "stair", "lift", "column", "wall"]
CARRIER_CATEGORIES = {"door", "window", "stair", "lift"}
STRONG_CARRIERS = {"door", "window"}
INSTANCE_SIZE_MM = {"door": (450.0, 6000.0), "window": (300.0, 10000.0), "stair": (1000.0, 40000.0), "lift": (900.0, 8000.0)}


def norm_name(name):
    return (name or "").split("$0$")[-1].split("|")[-1].strip().upper()


def categorize(name, rx):
    n = norm_name(name)
    if not n or n == "0" or rx["exclude"].search(n):
        return None
    for cat in CATEGORY_ORDER:
        if rx[cat].search(n):
            return cat
    return None


@dataclass
class Prim:
    t: str
    p: list
    layer: str
    cat: object = None
    carrier: int = -1
    carrier_name: str = ""
    r: float = 0.0
    c: tuple = None
    large: bool = False
    span: float = 0.0
    sid: int = 0
    iid: int = -1


@dataclass
class TextItem:
    text: str
    x: float
    y: float
    h: float

def _xy(v):
    return (float(v[0]), float(v[1]))


def _arc_prims(e, layer):
    ocs = e.ocs()
    c, r = e.dxf.center, float(e.dxf.radius)
    a0, a1 = float(e.dxf.start_angle), float(e.dxf.end_angle)
    span = (a1 - a0) % 360.0
    if r <= 0:
        return []
    if span < 1e-6 or span >= 359.999:
        return [Prim("circle", [_xy(ocs.to_wcs(c))], layer, r=r, c=_xy(ocs.to_wcs(c)))]
    pts = []
    for a in (a0, a0 + span / 2, a0 + span):
        q = Vec3(c.x + r * math.cos(math.radians(a)), c.y + r * math.sin(math.radians(a)), c.z)
        pts.append(_xy(ocs.to_wcs(q)))
    return [Prim("arc", pts, layer, r=r, c=_xy(ocs.to_wcs(c)), large=span > 180, span=span)]


def _circle_prims(e, layer):
    r = float(e.dxf.radius)
    if r <= 0:
        return []
    c = _xy(e.ocs().to_wcs(e.dxf.center))
    return [Prim("circle", [c], layer, r=r, c=c)]


def _bulge_segment(p1, p2, b, ocs, elev, layer):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return []
    w = lambda x, y: _xy(ocs.to_wcs(Vec3(x, y, elev)))
    if abs(b) < 1e-9:
        return [Prim("line", [w(x1, y1), w(x2, y2)], layer)]
    nx, ny = dy / L, -dx / L
    s = b * L / 2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    r = L * (1 + b * b) / (4 * abs(b))
    cx, cy = mx + nx * (s - math.copysign(r, s)), my + ny * (s - math.copysign(r, s))
    span = math.degrees(4 * math.atan(abs(b)))
    return [Prim("arc", [w(x1, y1), w(mx + nx * s, my + ny * s), w(x2, y2)], layer, r=r, c=w(cx, cy),
                 large=abs(b) > 1, span=span)]


def _lwpolyline_prims(e, layer):
    ocs, elev = e.ocs(), float(e.dxf.get("elevation", 0.0))
    pts = [(float(x), float(y), float(b)) for x, y, b in e.get_points("xyb")]
    if len(pts) < 2:
        return []
    if e.closed:
        pts.append(pts[0])
    out = []
    for (x1, y1, b), (x2, y2, _) in zip(pts[:-1], pts[1:]):
        out += _bulge_segment((x1, y1), (x2, y2), b, ocs, elev, layer)
    return out


def _polyline_prims(e, layer):
    verts = list(e.vertices)
    if len(verts) < 2:
        return []
    if e.is_2d_polyline:
        ocs, elev = e.ocs(), float(e.dxf.elevation.z) if e.dxf.hasattr("elevation") else 0.0
        pts = [(float(v.dxf.location.x), float(v.dxf.location.y), float(v.dxf.get("bulge", 0.0))) for v in verts]
        if e.is_closed:
            pts.append(pts[0])
        out = []
        for (x1, y1, b), (x2, y2, _) in zip(pts[:-1], pts[1:]):
            out += _bulge_segment((x1, y1), (x2, y2), b, ocs, elev, layer)
        return out
    if e.is_3d_polyline:
        pts = [_xy(v.dxf.location) for v in verts]
        if e.is_closed:
            pts.append(pts[0])
        return [Prim("line", [a, b], layer) for a, b in zip(pts[:-1], pts[1:]) if math.dist(a, b) > 1e-9]
    return []


def _curve_prims(e, layer):
    try:
        p = ezpath.make_path(e)
    except Exception:
        return []
    out, cur = [], _xy(p.start)
    for cmd in p.commands():
        end = _xy(cmd.end)
        kind = cmd.type.name
        if kind == "LINE_TO":
            if math.dist(cur, end) > 1e-9:
                out.append(Prim("line", [cur, end], layer))
        elif kind == "CURVE4_TO":
            out.append(Prim("cubic", [cur, _xy(cmd.ctrl1), _xy(cmd.ctrl2), end], layer))
        elif kind == "CURVE3_TO":
            q = _xy(cmd.ctrl)
            c1 = (cur[0] + 2 / 3 * (q[0] - cur[0]), cur[1] + 2 / 3 * (q[1] - cur[1]))
            c2 = (end[0] + 2 / 3 * (q[0] - end[0]), end[1] + 2 / 3 * (q[1] - end[1]))
            out.append(Prim("cubic", [cur, c1, c2, end], layer))
        cur = end
    return out


GEOMETRY = {"LINE": lambda e, l: [Prim("line", [_xy(e.dxf.start), _xy(e.dxf.end)], l)] if e.dxf.start != e.dxf.end else [],
            "ARC": _arc_prims, "CIRCLE": _circle_prims, "LWPOLYLINE": _lwpolyline_prims, "POLYLINE": _polyline_prims,
            "ELLIPSE": _curve_prims, "SPLINE": _curve_prims}
COMPOUND = {"DIMENSION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION", "LEADER", "MLEADER", "MULTILEADER", "ACAD_TABLE"}


@dataclass
class Exploded:
    prims: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    carriers: dict = field(default_factory=dict)
    errors: Counter = field(default_factory=Counter)


def _text(e):
    t = e.dxftype()
    try:
        if t == "MTEXT":
            return TextItem(e.plain_text().strip(), *_xy(e.dxf.insert), float(e.dxf.get("char_height", 250.0)))
        return TextItem((e.dxf.text or "").strip(), *_xy(e.dxf.insert), float(e.dxf.get("height", 250.0)))
    except Exception:
        return None


def explode(entities, out, rx, parent_layer="0", carrier=-1, carrier_cat=None, depth=0):
    for e in entities:
        t = e.dxftype()
        layer = norm_name(e.dxf.get("layer", "0"))
        if layer == "0":
            layer = parent_layer
        if t == "INSERT":
            for a in getattr(e, "attribs", []):
                ti = _text(a)
                if ti and ti.text:
                    out.texts.append(ti)
            c_id, c_cat = carrier, carrier_cat
            if carrier < 0 or carrier_cat not in STRONG_CARRIERS:
                lc = categorize(layer, rx)
                bc = None if (rx["exclude"].search(layer) and layer != "0") else categorize(e.dxf.name, rx)
                cat = lc if lc in CARRIER_CATEGORIES else (bc if bc in CARRIER_CATEGORIES else None)
                if carrier >= 0 and cat not in STRONG_CARRIERS:
                    cat = None
                if cat:
                    c_id, c_cat = len(out.carriers), cat
                    out.carriers[c_id] = {"cat": cat, "names": f"{norm_name(e.dxf.name)} {layer}"}
            if depth < 12:
                try:
                    explode(e.virtual_entities(), out, rx, layer, c_id, c_cat, depth + 1)
                except Exception as ex:
                    out.errors[f"INSERT {type(ex).__name__}"] += 1
        elif t in ("TEXT", "MTEXT", "ATTRIB"):
            ti = _text(e)
            if ti and ti.text:
                out.texts.append(ti)
        elif t in COMPOUND:
            if depth < 12:
                try:
                    explode(e.virtual_entities(), out, rx, layer, carrier, carrier_cat, depth + 1)
                except Exception as ex:
                    out.errors[f"{t} {type(ex).__name__}"] += 1
        elif t in GEOMETRY:
            try:
                prims = GEOMETRY[t](e, layer)
            except Exception as ex:
                out.errors[f"{t} {type(ex).__name__}"] += 1
                continue
            own = categorize(layer, rx)
            if carrier >= 0 and (carrier_cat in STRONG_CARRIERS or own is None or own == carrier_cat):
                cat, car = carrier_cat, carrier
            else:
                cat, car = own, -1
            for p in prims:
                p.cat, p.carrier = cat, car
            out.prims.extend(prims)


def read_dxf(path):
    try:
        return ezdxf.readfile(str(path)), "ok"
    except Exception:
        doc, auditor = recover.readfile(str(path))
        return doc, f"recovered ({len(auditor.errors)} errors)"


def explode_file(path, rules):
    doc, status = read_dxf(path)
    out = Exploded()
    explode(doc.modelspace(), out, rules.compiled())
    return doc, out, status

def wall_thickness_mode(prims, max_segments=2500):
    segs = np.array([p.p[0] + p.p[1] for p in prims if p.t == "line" and p.cat == "wall"], dtype=float).reshape(-1, 4)
    if len(segs) < 4:
        return None
    d = segs[:, 2:] - segs[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    keep = np.argsort(-L)[:max_segments]
    segs, d, L = segs[keep], d[keep], L[keep]
    ok = L > 0
    segs, d, L = segs[ok], d[ok], L[ok]
    u = d / L[:, None]
    nearest = []
    for s in range(0, len(segs), 500):
        ui, ai, Li = u[s:s + 500], segs[s:s + 500, :2], L[s:s + 500]
        par = np.abs(ui[:, None, 0] * u[None, :, 1] - ui[:, None, 1] * u[None, :, 0]) < 0.035
        rel = segs[None, :, :2] - ai[:, None, :]
        rel2 = segs[None, :, 2:] - ai[:, None, :]
        off = ui[:, None, 0] * rel[..., 1] - ui[:, None, 1] * rel[..., 0]
        t0 = (ui[:, None, :] * rel).sum(-1)
        t1 = (ui[:, None, :] * rel2).sum(-1)
        ov = np.minimum(Li[:, None], np.maximum(t0, t1)) - np.maximum(0, np.minimum(t0, t1))
        cand = par & (ov > 0.3 * np.minimum(Li[:, None], L[None, :])) & (np.abs(off) > 1e-6)
        dist = np.where(cand, np.abs(off), np.inf)
        best = dist.min(1)
        nearest.extend(best[np.isfinite(best)].tolist())
    if len(nearest) < 4:
        return None
    lg = np.log(np.array(nearest))
    hist, edges = np.histogram(lg, bins=np.arange(lg.min(), lg.max() + 0.1, 0.05))
    k = int(hist.argmax())
    sel = (lg >= edges[k]) & (lg < edges[k + 1])
    return float(np.exp(np.median(lg[sel])))


def infer_units(doc, prims):
    header = int(doc.header.get("$INSUNITS", 0) or 0)
    swings = [p for p in prims if p.t == "arc" and p.cat == "door" and 20 <= p.span <= 180]
    per_carrier = defaultdict(float)
    for p in swings:
        if p.carrier >= 0:
            per_carrier[p.carrier] = max(per_carrier[p.carrier], p.r)
    radii = list(per_carrier.values()) + [p.r for p in swings if p.carrier < 0]
    if radii:
        top = float(np.quantile(radii, 0.9))
        radii = [r for r in radii if r >= 0.2 * top]
    r_med = float(np.median(radii)) if radii else None
    t_mode = wall_thickness_mode(prims)
    cands = list(dict.fromkeys(([UNIT_TO_MM[header]] if header in UNIT_TO_MM else []) + UNIT_CANDIDATES))

    def score(f):
        s = 0
        if r_med:
            s += 600 <= f * r_med <= 1300
        if t_mode:
            s += 75 <= f * t_mode <= 450
        return s

    best = max(cands, key=lambda f: (score(f), f == UNIT_TO_MM.get(header), f == 1.0))
    sc = score(best)
    evidence = (r_med is not None) + (t_mode is not None)
    status = "confirmed" if evidence and sc == evidence else ("partial" if sc else "unverified")
    return {"insunits": header, "header_mm": UNIT_TO_MM.get(header), "mm_factor": best, "status": status,
            "door_leaf_mm": round(best * r_med, 1) if r_med else None, "n_door_arcs": len(radii),
            "wall_thickness_mm": round(best * t_mode, 1) if t_mode else None,
            "header_overridden": header in UNIT_TO_MM and UNIT_TO_MM[header] != best}


def scale_to_mm(ex, f):
    for p in ex.prims:
        p.p = [(x * f, y * f) for x, y in p.p]
        p.r *= f
        if p.c is not None:
            p.c = (p.c[0] * f, p.c[1] * f)
    for t in ex.texts:
        t.x, t.y, t.h = t.x * f, t.y * f, t.h * f


def _geom(p):
    if p.t == "circle":
        return shapely.Point(p.c).buffer(p.r, quad_segs=4).exterior
    return LineString(p.p)


CLUSTER_TOL = {"door": 50.0, "window": 50.0, "stair": 300.0, "lift": 300.0}


def assign_instances(ex):
    for p in ex.prims:
        if p.carrier >= 0:
            p.iid = p.carrier
            p.carrier_name = ex.carriers[p.carrier]["names"]
    next_id = len(ex.carriers)
    loose_names = {}
    for cat, tol in CLUSTER_TOL.items():
        idx = [i for i, p in enumerate(ex.prims) if p.cat == cat and p.carrier < 0]
        if not idx:
            continue
        geoms = [_geom(ex.prims[i]) for i in idx]
        merged = unary_union([g.buffer(tol / 2) for g in geoms])
        blobs = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        tree = STRtree(blobs)
        base = next_id
        for i, g in zip(idx, geoms):
            hit = tree.query(g, predicate="intersects")
            k = int(hit[0]) if len(hit) else 0
            ex.prims[i].iid = base + k
            loose_names.setdefault(base + k, set()).add(ex.prims[i].layer)
        next_id = base + len(blobs)
        for k in range(base, next_id):
            ex.carriers[k] = {"cat": cat, "names": " ".join(sorted(loose_names.get(k, []))), "loose": True}
    for p in ex.prims:
        if p.iid >= 0 and not p.carrier_name:
            p.carrier_name = ex.carriers[p.iid]["names"]


def door_subtype(members, names, rx):
    pts = np.array([q for p in members for q in (p.p if p.t != "circle" else [p.c])], dtype=float)
    W = float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0
    leaves = [p for p in members if p.t == "arc" and 20 <= p.span <= 180 and max(250.0, 0.2 * W) <= p.r <= min(1600.0, 1.15 * W)]
    centers = {(round(p.c[0] / 30), round(p.c[1] / 30)) for p in leaves}
    if len(centers) >= 2:
        return "double door", W, [p.r for p in leaves]
    if len(centers) == 1:
        return "single door", W, [p.r for p in leaves]
    if rx["rolling"].search(names):
        return "rolling door", W, []
    if rx["folding"].search(names):
        return "folding door", W, []
    if rx["revolving"].search(names):
        return "revolving door", W, []
    return "sliding door", W, []


STUFF_NAME = {"wall": "wall", "curtain": "curtain wall", "railing": "railing"}


def label_prims(ex, rules, raw_names):
    rx = rules.compiled()
    sid = {n: i + 1 for i, n in enumerate(raw_names)}
    stuff = dict(STUFF_NAME)
    if rules.column_as in sid:
        stuff["column"] = rules.column_as
    members = defaultdict(list)
    for p in ex.prims:
        if p.iid >= 0:
            members[p.iid].append(p)
    inst_class, door_info, implausible = {}, [], Counter()
    for iid, mem in members.items():
        cat = ex.carriers[iid]["cat"]
        names = ex.carriers[iid]["names"]
        pts = np.array([q for p in mem for q in (p.p if p.t != "circle" else [p.c])], dtype=float)
        size = float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0
        lo, hi = INSTANCE_SIZE_MM.get(cat, (0.0, float("inf")))
        if not lo <= size <= hi:
            inst_class[iid] = None
            implausible[cat] += 1
            continue
        if cat == "door":
            name, W, leaf = door_subtype(mem, names, rx)
            door_info.append({"iid": iid, "subtype": name, "width_mm": round(W, 1), "leaf_radii_mm": [round(r, 1) for r in leaf],
                              "names": names, "loose": bool(ex.carriers[iid].get("loose"))})
        elif cat == "window":
            name = "bay window" if rx["bay"].search(names) else "window"
        elif cat == "stair":
            name = "stairs"
        elif cat == "lift":
            name = "elevator"
        else:
            name = None
        inst_class[iid] = name
    dense = {iid: k for k, iid in enumerate(sorted(i for i, n in inst_class.items() if n))}
    for p in ex.prims:
        if p.iid >= 0:
            name = inst_class.get(p.iid)
            p.sid, p.iid = (sid[name], dense[p.iid]) if name in sid else (0, -1)
        elif p.cat in stuff:
            p.sid, p.iid = sid[stuff[p.cat]], -1
        else:
            p.sid, p.iid = 0, -1
    ex.implausible = dict(implausible)
    return door_info

def _bounds(ex):
    xs, ys = [], []
    for p in ex.prims:
        if p.t == "circle":
            xs += [p.c[0] - p.r, p.c[0] + p.r]
            ys += [p.c[1] - p.r, p.c[1] + p.r]
        else:
            xs += [q[0] for q in p.p]
            ys += [q[1] for q in p.p]
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def write_svg(ex, path, margin=500.0):
    x0, y0, x1, y1 = _bounds(ex)
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    fx = lambda x: x - x0
    fy = lambda y: y1 - y
    P = lambda q: f"{fx(q[0]):.2f} {fy(q[1]):.2f}"
    out = ['<?xml version="1.0" encoding="utf-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 {x1 - x0:.2f} {y1 - y0:.2f}" data-units="mm">']
    for p in ex.prims:
        attrs = f' semantic-id="{p.sid}" instance-id="{p.iid}"' if p.sid else ""
        style = ' fill="none" stroke="rgb(0,0,0)" stroke-width="1"'
        if p.t == "line":
            out.append(f'<path d="M {P(p.p[0])} L {P(p.p[1])}"{style}{attrs}/>')
        elif p.t == "arc":
            s, m, e = [(fx(q[0]), fy(q[1])) for q in p.p]
            cross = (m[0] - s[0]) * (e[1] - m[1]) - (m[1] - s[1]) * (e[0] - m[0])
            sweep = 1 if cross > 0 else 0
            out.append(f'<path d="M {s[0]:.2f} {s[1]:.2f} A {p.r:.2f} {p.r:.2f} 0 {int(p.large)} {sweep} {e[0]:.2f} {e[1]:.2f}"{style}{attrs}/>')
        elif p.t == "circle":
            out.append(f'<circle cx="{fx(p.c[0]):.2f}" cy="{fy(p.c[1]):.2f}" r="{p.r:.2f}"{style}{attrs}/>')
        elif p.t == "cubic":
            a, b, c, d = p.p
            out.append(f'<path d="M {P(a)} C {P(b)} {P(c)} {P(d)}"{style}{attrs}/>')
    for t in ex.texts:
        out.append(f'<text x="{fx(t.x):.2f}" y="{fy(t.y):.2f}" font-size="{max(t.h, 1.0):.1f}">{escape(t.text)}</text>')
    out.append("</svg>")
    FsPath(path).parent.mkdir(parents=True, exist_ok=True)
    FsPath(path).write_text("\n".join(out), encoding="utf-8")
    return {"origin_mm": [round(x0, 1), round(y1, 1)], "size_mm": [round(x1 - x0, 1), round(y1 - y0, 1)]}


PREVIEW_COLORS = {"wall": "#1f4e9c", "curtain wall": "#17becf", "railing": "#8c564b", "single door": "#d62728",
                  "double door": "#ff7f0e", "sliding door": "#e377c2", "folding door": "#bcbd22", "rolling door": "#bcbd22",
                  "revolving door": "#bcbd22", "window": "#2ca02c", "bay window": "#98df8a", "stairs": "#9467bd",
                  "elevator": "#7f7f7f"}


def _poly(p):
    if p.t == "circle":
        a = np.linspace(0, 2 * np.pi, 13)
        return np.stack([p.c[0] + p.r * np.cos(a), p.c[1] + p.r * np.sin(a)], 1)
    return np.array(p.p)


def preview(ex, raw_names, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(14, 14))
    groups = defaultdict(list)
    for p in ex.prims:
        groups[raw_names[p.sid - 1] if p.sid else None].append(_poly(p))
    if groups.get(None):
        ax.add_collection(LineCollection(groups.pop(None), colors="#d0d0d0", linewidths=0.3))
    for name, segs in groups.items():
        ax.add_collection(LineCollection(segs, colors=PREVIEW_COLORS.get(name, "#000000"), linewidths=0.8, label=f"{name} ({len(segs)})"))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])
    FsPath(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path

def find_drawings(root):
    root = FsPath(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".dxf", ".dwg"))


def source_group(path, root):
    rel = FsPath(path).relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def convert_dwgs(src_root, dxf_root, oda_path="ODAFileConverter", version="ACAD2018"):
    src_root, dxf_root = FsPath(src_root), FsPath(dxf_root)
    dwgs = [p for p in find_drawings(src_root) if p.suffix.lower() == ".dwg"]
    if not dwgs:
        return {"dwg": 0, "converted": 0}
    exe = shutil.which(oda_path) or (oda_path if FsPath(oda_path).exists() else None)
    if exe is None:
        print("ODA File Converter not found: convert DWGs locally, or install it (see Stage 1). Skipping", len(dwgs), "DWG files.")
        return {"dwg": len(dwgs), "converted": 0}
    cmd = [exe, str(src_root), str(dxf_root), version, "DXF", "1", "1", "*.DWG"]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    done = sum(1 for d in dwgs if (dxf_root / d.relative_to(src_root)).with_suffix(".dxf").exists())
    return {"dwg": len(dwgs), "converted": done}



def process_file(path, group, rules, raw_names, svg_dir, preview_dir):
    stem = f"{group}__{FsPath(path).stem}"
    rep = {"file": str(path), "group": group, "id": stem, "flags": []}
    try:
        doc, ex, status = explode_file(path, rules)
        rep["read_status"] = status
        units = infer_units(doc, ex.prims)
        scale_to_mm(ex, units["mm_factor"])
        assign_instances(ex)
        doors = label_prims(ex, rules, raw_names)
        geom = write_svg(ex, FsPath(svg_dir) / f"{stem}.svg")
        preview(ex, raw_names, FsPath(preview_dir) / f"{stem}.png", title=f"{stem}  ({units['mm_factor']} mm/unit, {units['status']})")
    except Exception as e:
        rep.update({"flags": ["error"], "error": repr(e)[:300]})
        return rep
    n = len(ex.prims)
    sid_counts = Counter(p.sid for p in ex.prims)
    inst = defaultdict(set)
    for p in ex.prims:
        if p.iid >= 0:
            inst[p.sid].add(p.iid)
    unl = Counter(p.layer for p in ex.prims if p.sid == 0)
    widths = [d["width_mm"] for d in doors]
    rep.update({
        "svg": str(FsPath(svg_dir) / f"{stem}.svg"), "preview": str(FsPath(preview_dir) / f"{stem}.png"),
        "n_prims": n, "n_text": len(ex.texts), "labelled_frac": round(1 - sid_counts.get(0, 0) / max(n, 1), 4),
        "mm_factor": units["mm_factor"], "unit_status": units["status"], "insunits": units["insunits"],
        "header_overridden": units["header_overridden"], "door_leaf_mm": units["door_leaf_mm"],
        "wall_thickness_mm": units["wall_thickness_mm"],
        "n_doors": len(doors), "door_width_median_mm": float(np.median(widths)) if widths else None,
        "door_subtypes": dict(Counter(d["subtype"] for d in doors)),
        "class_prims": {raw_names[k - 1]: v for k, v in sid_counts.items() if k},
        "class_instances": {raw_names[k - 1]: len(v) for k, v in inst.items()},
        "unlabelled_layers": dict(unl.most_common(25)), "explode_errors": dict(ex.errors),
        "implausible_instances": getattr(ex, "implausible", {}),
        "size_mm": geom["size_mm"],
    })
    wall_prims = sum(v for k, v in rep["class_prims"].items() if k == "wall")
    if wall_prims < 20:
        rep["flags"].append("no_walls")
    if not doors:
        rep["flags"].append("no_doors")
    elif not 600 <= rep["door_width_median_mm"] <= 1400:
        rep["flags"].append("door_size_off")
    if units["status"] == "unverified":
        rep["flags"].append("units_unverified")
    if rep["labelled_frac"] < 0.03:
        rep["flags"].append("low_coverage")
    return rep


def _process_job(args):
    return process_file(*args)


def process_all(items, rules, raw_names, svg_dir, preview_dir, workers=2):
    jobs = [(str(p), g, rules, raw_names, str(svg_dir), str(preview_dir)) for p, g in items]
    t0 = time.time()
    if workers <= 1:
        reps = [_process_job(j) for j in jobs]
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            reps = list(pool.imap_unordered(_process_job, jobs, chunksize=1))
    print(f"processed {len(jobs)} drawings in {time.time() - t0:.0f}s")
    return pd.DataFrame(reps).sort_values("id").reset_index(drop=True)


def unmatched_layers(report, top=40):
    tot, files = Counter(), Counter()
    for d in report["unlabelled_layers"].dropna():
        tot.update(d)
        files.update(d.keys())
    return pd.DataFrame([{"layer": k, "unlabelled_prims": v, "files": files[k]} for k, v in tot.most_common(top)])

def split_floors(dxf_dir, plan, log=print):
    dxf_dir = FsPath(dxf_dir)
    jobs = [(role, tower, block, tag) for role, floors in plan.items() for tower, block, tag in floors]
    for role, tower, block, tag in jobs:
        dest = dxf_dir / role / f"{tower}_{tag}.dxf"
        if dest.exists():
            continue
        src = next((dxf_dir / tower).glob("*.dxf"))
        t0 = time.time()
        doc, _ = recover.readfile(str(src))
        msp = doc.modelspace()
        kept = 0
        for e in list(msp):
            if e.dxftype() == "INSERT" and e.dxf.name == block:
                kept += 1
            else:
                msp.delete_entity(e)
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(str(dest))
        log(f"{role}/{dest.name}: kept {kept} insert, {dest.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s")
    masters = dxf_dir.parent / "dxf_master"
    for tower in {t for _, t, _, _ in jobs}:
        if (dxf_dir / tower).exists():
            masters.mkdir(exist_ok=True)
            shutil.move(str(dxf_dir / tower), str(masters / tower))

def assign_split_groups(report, test_groups=(), val_groups=(), val_frac=0.15, test_frac=0.15, seed=0):
    groups = report.groupby("group").size().to_dict()
    split = {g: "test" for g in test_groups if g in groups}
    split.update({g: "val" for g in val_groups if g in groups})
    free = sorted(g for g in groups if g not in split)
    rnd = np.random.default_rng(seed)
    rnd.shuffle(free)
    total = sum(groups.values())
    need = {"test": test_frac * total - sum(groups[g] for g, s in split.items() if s == "test"),
            "val": val_frac * total - sum(groups[g] for g, s in split.items() if s == "val")}
    for g in free:
        left = [x for x in free if x not in split]
        if len(left) <= 1 and "train" not in split.values():
            target = "train"
        elif need["test"] > 0:
            target = "test"
        elif need["val"] > 0:
            target = "val"
        else:
            target = "train"
        split[g] = target
        if target != "train":
            need[target] -= groups[g]
    if len(groups) < 3:
        print("WARNING: fewer than 3 source groups; splits cannot be source-disjoint")
    return split


def _graph_job(args):
    svg, cfg_dict, lut, stuff, min_wall, min_frac, wall_id, other_id, fpgraph_dir = args
    import sys
    if fpgraph_dir not in sys.path:
        sys.path.insert(0, fpgraph_dir)
    import fpgraph
    try:
        cfg = fpgraph.GraphConfig(**cfg_dict)
        P = fpgraph.split_long_lines(fpgraph.parse_svg(svg, cfg.mm_per_unit), cfg.max_prim_len_mm)
        y_all = lut[P["raw"]].astype(np.int64)
        windows = fpgraph.tile_windows(fpgraph.midpoints(P), cfg)
        kept = []
        for t, (win, idx) in enumerate(windows):
            y = y_all[idx]
            if int((y == wall_id).sum()) < min_wall or float((y != other_id).mean()) < min_frac:
                continue
            g = fpgraph.build_graph(P, idx, win, cfg, lut, stuff)
            g["meta"] = {"drawing": FsPath(svg).stem, "tile": t, "window": [round(v, 1) for v in win],
                         "n_nodes": int(len(idx)), "n_edges": int(g["edge_index"].shape[1]), "n_text": int(P["n_text"])}
            kept.append(g)
        return svg, kept, len(windows), ""
    except Exception as e:
        return svg, [], 0, repr(e)


def build_split(svgs, split, out_dir, fpgraph, cfg, lut, stuff, targets, min_wall_nodes=20, min_labelled_frac=0.03,
                workers=2, tiles_per_shard=256):
    writer = fpgraph.ShardWriter(out_dir, split, tiles_per_shard)
    fp_dir = str(FsPath(fpgraph.__file__).resolve().parent)
    jobs = [(str(s), asdict(cfg), lut, stuff, min_wall_nodes, min_labelled_frac, targets.index("wall"), targets.index("other"), fp_dir)
            for s in svgs]
    if workers <= 1:
        results = map(_graph_job, jobs)
        pool = None
    else:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap_unordered(_graph_job, jobs, chunksize=1)
    errors, dropped, t0 = [], 0, time.time()
    for svg, kept, n_all, err in results:
        if err:
            errors.append({"file": svg, "error": err})
        dropped += n_all - len(kept)
        for t in kept:
            writer.add(t)
    if pool is not None:
        pool.close()
        pool.join()
    writer.flush()
    index = pd.DataFrame(writer.index)
    index.to_parquet(FsPath(out_dir) / f"index_{split}.parquet", index=False)
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(FsPath(out_dir) / f"errors_{split}.csv", index=False)
    print(f"{split}: {len(svgs)} drawings -> {len(index)} tiles kept, {dropped} dropped, {len(errors)} errors, {time.time() - t0:.0f}s")
    return index


def write_card(out_dir, info):
    rows = "\n".join(f"| {s['split']} | {s['drawings']} | {s['tiles']} | {s['nodes']:,} | {s['edges']:,} | {s['edge_pos_weight']} |"
                     for s in info["stats"])
    groups = "\n".join(f"| {g} | {s} | {n} |" for g, s, n in info["groups"])
    card = f"""# Indian CAD fine-tuning set (Phase B)

Auto-labelled Indian floor plans for fine-tuning the Phase A GNN. Same tile, graph and label format as the FloorPlanCAD
shards in this repository, built with the same `fpgraph.py` and `config.json`.

**Labels** come from layer and block-name rules (`label_rules.json`) plus geometry (leaf-arc count decides single, double
or sliding doors; door swings and wall thickness confirm the units). They are automatic, so treat them as silver
labels: every drawing was checked on a preview before inclusion (`review.csv`).

**Provenance.** Drawings come from the sources listed below. Raw drawings are not redistributed. Respect each source's
terms; this derivative is for non-commercial research.

## Splits (source-disjoint)

| split | drawings | tiles | nodes | edges | edge pos_weight |
|---|---|---|---|---|---|
{rows}

| source group | split | drawings |
|---|---|---|
{groups}

Tiles were kept only if they have at least {info['min_wall_nodes']} wall nodes and at least {info['min_labelled_frac']:.0%} labelled nodes, which drops
elevations, sections, title blocks and empty margins.
"""
    (FsPath(out_dir) / "README.md").write_text(card)


def push_folder(out_dir, repo_id, path_in_repo="indian", token=None, private=True):
    from huggingface_hub import HfApi, get_token
    token = token or get_token()
    if not token:
        print("No Hugging Face token found; set HF_TOKEN (Colab secret) and re-run.")
        return None
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    out_dir = FsPath(out_dir)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir), path_in_repo=path_in_repo,
                      ignore_patterns=["graphs/**"], commit_message="Phase B metadata")
    for sub in sorted(p for p in (out_dir / "graphs").iterdir() if p.is_dir()):
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(sub), path_in_repo=f"{path_in_repo}/graphs/{sub.name}",
                          commit_message=f"Phase B shards: {sub.name}")
        print("uploaded", sub.name)
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"
    print("pushed to", url)
    return url
