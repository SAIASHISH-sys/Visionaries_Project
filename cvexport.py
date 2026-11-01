import json
from pathlib import Path

import indlabel
import tracka_core


def list_floor_blocks(doc, top=15):
    rows = [{"block": i.dxf.name, "entities": len(doc.blocks[i.dxf.name])} for i in doc.modelspace().query("INSERT")]
    return sorted(rows, key=lambda r: -r["entities"])[:top]


def floor_entities(doc, floor_only, floor_block=None):
    if floor_only:
        try:
            top = tracka_core.pick_floor_insert(doc, floor_block)
            return top, list(top.virtual_entities())
        except (ValueError, StopIteration):
            if floor_block:
                raise ValueError(f"no INSERT of block {floor_block!r} in modelspace; see list_floor_blocks")
    return None, list(doc.modelspace())


def export_svg(dxf_path, svg_path, floor_only=True, floor_block=None, mm_factor=None, rules=None, doc=None):
    rules = rules or indlabel.Rules()
    status = "preloaded"
    if doc is None:
        doc, status = indlabel.read_dxf(dxf_path)
    top, entities = floor_entities(doc, floor_only, floor_block)
    ex = indlabel.Exploded()
    indlabel.explode(entities, ex, rules.compiled())
    units = indlabel.infer_units(doc, ex.prims)
    factor = float(mm_factor or units["mm_factor"])
    indlabel.scale_to_mm(ex, factor)
    geom = indlabel.write_svg(ex, svg_path)
    info = {**geom, "units": units, "mm_factor": factor, "read_status": status, "n_prims": len(ex.prims),
            "n_text": len(ex.texts), "floor_block": top.dxf.name if top is not None else None,
            "source_file": Path(dxf_path).name, "explode_errors": dict(ex.errors)}
    Path(svg_path).with_suffix(".json").write_text(json.dumps(info, default=float, indent=1))
    return doc, top, ex, info


def load_info(svg_path):
    return json.loads(Path(svg_path).with_suffix(".json").read_text())

def export_silver_svg(ex, svg_path, raw_names, rules=None):
    rules = rules or indlabel.Rules()
    indlabel.assign_instances(ex)
    indlabel.label_prims(ex, rules, raw_names)
    return indlabel.write_svg(ex, svg_path)