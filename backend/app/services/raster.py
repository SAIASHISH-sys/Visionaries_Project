import os
from functools import lru_cache

from fastapi import HTTPException

from .. import settings
from .common import stamp

_CPU_THREADS = max(1, (os.cpu_count() or 2) // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_THREADS))


@lru_cache(maxsize=1)
def load_bundle():
    if not settings.RASTER_CKPT.exists():
        raise HTTPException(503, f"raster model missing: {settings.RASTER_CKPT}")
    import torch

    torch.set_num_threads(_CPU_THREADS)

    import rastercv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(settings.RASTER_CKPT, map_location="cpu", weights_only=False)
    model = rastercv.make_model(settings.RASTER_ENCODER, None)
    model.load_state_dict(ck["model"])
    return {"model": model.to(device).eval(), "device": device}


def to_manifest(struct, region, mm_per_px, ckpt_name):
    adjacent = {}
    for e in struct["adjacency"]:
        adjacent.setdefault(e["a"], set()).add(e["b"])
        adjacent.setdefault(e["b"], set()).add(e["a"])
    rooms = [{"id": r["id"], "status": "room", "enclosure": r["enclosure"], "raw_label": None, "labels": [],
              "area_m2": r["area_m2"], "polygon": r["polygon"], "adjacent": sorted(adjacent.get(r["id"], [])),
              "source": "cv_model"} for r in struct["rooms"]]
    openings = [{"id": f"O{i + 1:03d}", "source": "cv_model", **o} for i, o in enumerate(struct["openings"])]
    return {
        "schema_version": "raster-0.1",
        "units": {"length": "mm", "area": "m2", "coordinates": "WCS, mm"},
        "rooms": rooms,
        "openings": openings,
        "walls": [],
        "adjacency": struct["adjacency"],
        "corridors": [],
        "vertical_circulation": [],
        "unassigned_labels": [],
        "summary": {"rooms": len(rooms), "doors": sum(o["kind"] == "door" for o in openings),
                    "windows": sum(o["kind"] == "window" for o in openings),
                    "adjacency_edges": len(struct["adjacency"]), "elapsed_s": struct["seconds"],
                    "raster": {"region_mm": list(map(float, region)), "mm_per_px": mm_per_px, "checkpoint": ckpt_name}},
    }


def run(dxf, classical_manifest, run_id, source_file):
    b = load_bundle()
    import rastercv

    s = settings.RASTER_MM_PER_PX
    region = rastercv.manifest_region(classical_manifest)
    img, _ = rastercv.render_dxf(dxf, region, s)
    pred, conf = rastercv.predict_image(b["model"], img, b["device"], batch=8 if b["device"] == "cuda" else 4)
    struct = rastercv.extract_structure(pred, conf, region, s)
    return stamp(to_manifest(struct, region, s, settings.RASTER_CKPT.name), "raster", run_id, source_file)
