import json
import os
from functools import lru_cache

from fastapi import HTTPException

from .. import settings
from .common import stamp

_CPU_THREADS = max(1, (os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU_THREADS))


@lru_cache(maxsize=1)
def load_bundle():
    missing = [p for p in (settings.GNN_CKPT, settings.PHASE_A_CONFIG) if not p.exists()]
    if missing:
        raise HTTPException(503, f"GNN model files missing: {', '.join(map(str, missing))}")
    import torch

    torch.set_num_threads(_CPU_THREADS)

    import fpgraph
    import gnninfer

    device = torch.device(settings.GNN_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    phase_a = json.loads(settings.PHASE_A_CONFIG.read_text())
    model, _ = gnninfer.load_model(settings.GNN_CKPT, device)
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    return {
        "model": model,
        "device": device,
        "phase_a": phase_a,
        "lut": fpgraph.build_label_map(phase_a["raw_class_names"], phase_a["harmonization"], phase_a["targets"]),
        "stuff": fpgraph.stuff_mask(phase_a["targets"]),
        "thr": gnninfer.read_threshold(settings.GNN_THRESHOLD),
    }


def run(dxf, run_path, run_id, source_file, floor_block=None):
    b = load_bundle()
    import cvpipeline

    manifest = cvpipeline.run_floor(dxf, b["model"], b["phase_a"], b["lut"], b["stuff"], b["device"],
                                        work_dir=run_path / "gnn_work", thr=b["thr"], floor_block=floor_block,
                                        amp=b["device"].type == "cuda",
                                        node_budget=12000, max_tiles=8)
    return stamp(manifest, "gnn", run_id, source_file)
