import math
from pathlib import Path

import numpy as np

import fpgraph

SEM = {"wall": 1, "single door": 3, "double door": 4, "sliding door": 5, "window": 9, "stairs": 30, "bed": 14}


def _path(d, sem, inst=-1):
    return f'<path d="{d}" fill="none" stroke="#000" stroke-width="0.1" semantic-id="{sem}" instance-id="{inst}"/>'


def _wall(x0, y0, x1, y1, t=1.0):
    horiz = abs(y1 - y0) < abs(x1 - x0)
    o = (0, t) if horiz else (t, 0)
    return [_path(f"M {x0 - o[0]:.3f} {y0 - o[1]:.3f} L {x1 - o[0]:.3f} {y1 - o[1]:.3f}", SEM["wall"]),
            _path(f"M {x0 + o[0]:.3f} {y0 + o[1]:.3f} L {x1 + o[0]:.3f} {y1 + o[1]:.3f}", SEM["wall"])]


def _door(x, y, horiz, inst, double, r=8.0):
    if horiz:
        pts = [(x, y, x + r, y + r, x + r, y)] if not double else [(x, y, x + r, y + r, x + r, y), (x + 2 * r, y, x + r, y + r, x + r, y)]
        out = []
        for ax, ay, bx, by, cx, cy in pts:
            out.append(_path(f"M {ax:.3f} {ay:.3f} A {r} {r} 0 0 1 {bx:.3f} {by:.3f}", SEM["double door" if double else "single door"], inst))
            out.append(_path(f"M {ax:.3f} {ay:.3f} L {ax:.3f} {ay + r:.3f}", SEM["double door" if double else "single door"], inst))
        return out
    return [_path(f"M {x:.3f} {y:.3f} A {r} {r} 0 0 0 {x + r:.3f} {y + r:.3f}", SEM["single door"], inst),
            _path(f"M {x:.3f} {y:.3f} L {x + r:.3f} {y:.3f}", SEM["single door"], inst)]


def _window(x, y, horiz, inst, w=12.0):
    out = []
    for off in (-0.6, 0.0, 0.6):
        d = f"M {x:.3f} {y + off:.3f} L {x + w:.3f} {y + off:.3f}" if horiz else f"M {x + off:.3f} {y:.3f} L {x + off:.3f} {y + w:.3f}"
        out.append(_path(d, SEM["window"], inst))
    return out


def _stairs(x, y, inst, n=9, w=10.0, step=2.5):
    out = [_path(f"M {x:.3f} {y + i * step:.3f} L {x + w:.3f} {y + i * step:.3f}", SEM["stairs"], inst) for i in range(n)]
    out.append(_path(f"M {x:.3f} {y:.3f} L {x:.3f} {y + (n - 1) * step:.3f}", SEM["stairs"], inst))
    out.append(_path(f"M {x + w:.3f} {y:.3f} L {x + w:.3f} {y + (n - 1) * step:.3f}", SEM["stairs"], inst))
    return out


def synth_svg(rng):
    cut_x = float(rng.uniform(35, 65))
    cut_y = float(rng.uniform(35, 65))
    els, inst = [], 0
    for seg in [(5, 5, 95, 5), (5, 95, 95, 95), (5, 5, 5, 95), (95, 5, 95, 95), (cut_x, 5, cut_x, 95), (5, cut_y, 95, cut_y)]:
        els += _wall(*seg)
    for _ in range(int(rng.integers(3, 6))):
        horiz = bool(rng.integers(2))
        if horiz:
            x, y = float(rng.uniform(8, 80)), cut_y
        else:
            x, y = cut_x, float(rng.uniform(8, 80))
        els += _door(x, y, horiz, inst, double=bool(rng.random() < 0.3)) if horiz else _door(x, y, False, inst, False)
        inst += 1
    for _ in range(int(rng.integers(2, 5))):
        side = int(rng.integers(4))
        t = float(rng.uniform(10, 75))
        x, y, horiz = [(t, 5, True), (t, 95, True), (5, t, False), (95, t, False)][side]
        els += _window(x, y, horiz, inst)
        inst += 1
    els += _stairs(float(rng.uniform(10, 25)), float(rng.uniform(10, 25)), inst)
    inst += 1
    bx, by = float(rng.uniform(55, 75)), float(rng.uniform(55, 75))
    els.append(_path(f"M {bx} {by} L {bx + 10} {by} L {bx + 10} {by + 7} L {bx} {by + 7} Z", SEM["bed"]))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(els) + "</svg>")


def build_synthetic(out_dir, cfg_json, n_train=24, n_val=6, n_test=6, seed=0):
    import json
    out_dir = Path(out_dir)
    svg_dir = out_dir / "_svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    conf = json.loads(Path(cfg_json).read_text())
    cfg = fpgraph.GraphConfig(**conf["graph"])
    lut = fpgraph.build_label_map(conf["raw_class_names"], conf["harmonization"], conf["targets"])
    stuff = fpgraph.stuff_mask(conf["targets"])
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        paths = []
        for i in range(n):
            p = svg_dir / f"{split}_{i:03d}.svg"
            p.write_text(synth_svg(rng))
            paths.append(p)
        fpgraph.process_split(paths, split, out_dir, cfg, lut, stuff, workers=1, tiles_per_shard=16, log_every=10 ** 9)
    return out_dir
