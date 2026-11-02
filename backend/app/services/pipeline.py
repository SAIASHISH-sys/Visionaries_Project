from .. import store
from . import classical, gnn, raster


def run_track(run_id, track):
    existing = store.manifest_path(run_id, track)
    if existing.exists():
        return store.read_json(existing)
    meta = store.read_meta(run_id)
    dxf, path = store.dxf_path(run_id), store.run_dir(run_id)
    if track == "classical":
        manifest = classical.run(dxf, run_id, meta["source_file"], meta["floor_block"])
    else:
        reference = store.load_manifest(run_id, "classical")
        if track == "gnn":
            manifest = gnn.run(dxf, path, run_id, meta["source_file"], floor_block=reference["floor"]["block"])
        else:
            manifest = raster.run(dxf, reference, run_id, meta["source_file"])
    store.save_manifest(run_id, track, manifest)
    return manifest
