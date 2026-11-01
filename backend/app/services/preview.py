import cvexport
import indlabel

from .. import store
from .common import COMPUTE_LOCK

THIN_STROKE = 'stroke="rgb(0,0,0)" stroke-width="1"'
SCREEN_STROKE = 'stroke="rgb(0,0,0)" stroke-width="1.3" vector-effect="non-scaling-stroke"'


def render(dxf, run_path, floor_block=None):
    doc, _ = indlabel.read_dxf(dxf)
    floors = cvexport.list_floor_blocks(doc)
    with COMPUTE_LOCK:
        _, _, _, info = cvexport.export_svg(dxf, run_path / "preview.svg", floor_only=True, floor_block=floor_block, doc=doc)
    path = run_path / "preview.svg"
    path.write_text(path.read_text(encoding="utf-8").replace(THIN_STROKE, SCREEN_STROKE), encoding="utf-8")
    return floors, info


def describe(run_id):
    path = store.run_dir(run_id) / "preview.svg"
    info = store.read_json(path.with_suffix(".json"))
    return {"url": f"/runs/{run_id}/preview.svg", "version": int(path.stat().st_mtime),
            "origin_mm": info["origin_mm"], "size_mm": info["size_mm"], "mm_factor": info["mm_factor"],
            "units_status": info["units"]["status"], "n_prims": info["n_prims"], "floor_block": info["floor_block"]}
