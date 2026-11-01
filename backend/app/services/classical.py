import tracka_core

from .common import COMPUTE_LOCK, stamp


def run(dxf, run_id, source_file, floor_block=None):
    cfg = {**tracka_core.CONFIG, "floor_block": floor_block}
    with COMPUTE_LOCK:
        result = tracka_core.run(str(dxf), cfg)
        manifest = tracka_core.build_manifest(dxf, result, cfg)
    return stamp(manifest, "classical", run_id, source_file)
