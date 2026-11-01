import json
import time
from pathlib import Path

import cvexport
import cvadapter
import gnninfer


def run_floor(dxf_path, model, phase_a, lut, stuff, device, work_dir="cv_work", thr=0.5, floor_only=True, floor_block=None, mm_factor=None,
              node_budget=12000, max_tiles=8, amp=False, out_path=None):
    t0 = time.time()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    svg = work / (Path(dxf_path).stem.replace(" ", "_") + ".svg")
    doc, top, ex, info = cvexport.export_svg(dxf_path, svg, floor_only=floor_only, floor_block=floor_block, mm_factor=mm_factor)
    cfg = gnninfer.graph_config(phase_a)
    P = gnninfer.read_primitives(svg, cfg)
    tiles = gnninfer.build_tiles(P, cfg, lut, stuff)
    outs = gnninfer.predict_tiles(model, tiles, device, node_budget, max_tiles, amp)
    merged = gnninfer.merge_tiles(len(P["ptype"]), tiles, outs)
    scene = gnninfer.decode_scene(P, merged, thr)
    Pw = cvadapter.to_wcs(P, info["origin_mm"])
    manifest, r = cvadapter.run_cv(svg, info, Pw, scene, top=top)
    manifest["summary"]["cv"]["total_elapsed_s"] = round(time.time() - t0, 1)
    if out_path:
        Path(out_path).write_text(json.dumps(manifest, indent=1, default=float))
    return manifest