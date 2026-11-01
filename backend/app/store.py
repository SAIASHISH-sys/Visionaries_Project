import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException

from . import settings

_status_lock = threading.RLock()


def new_run():
    run_id = uuid.uuid4().hex[:12]
    path = settings.RUNS_DIR / run_id
    path.mkdir(parents=True)
    return run_id, path


def run_dir(run_id):
    path = settings.RUNS_DIR / run_id
    if not path.is_dir():
        raise HTTPException(404, f"unknown run {run_id}")
    return path


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


def read_json(path):
    return json.loads(Path(path).read_text())


def read_meta(run_id):
    return read_json(run_dir(run_id) / "meta.json")


def write_meta(run_id, meta):
    write_json(run_dir(run_id) / "meta.json", meta)


def manifest_path(run_id, track):
    return run_dir(run_id) / f"manifest_{track}.json"


def save_manifest(run_id, track, manifest):
    write_json(manifest_path(run_id, track), manifest)


def load_manifest(run_id, track):
    path = manifest_path(run_id, track)
    if not path.exists():
        raise HTTPException(409, f"run {run_id} has no {track} manifest yet")
    return read_json(path)


def available_tracks(run_id):
    return [t for t in settings.TRACKS if manifest_path(run_id, t).exists()]


def dxf_path(run_id):
    return run_dir(run_id) / "floor.dxf"


def idle_status():
    return {"classical_mm_factor": None,
            "tracks": {t: {"state": "idle", "seconds": None, "error": None, "summary": None} for t in settings.TRACKS}}


def read_status(run_id):
    path = run_dir(run_id) / "status.json"
    with _status_lock:
        return read_json(path) if path.exists() else idle_status()


def update_status(run_id, track=None, **fields):
    path = run_dir(run_id) / "status.json"
    with _status_lock:
        status = read_status(run_id)
        (status["tracks"][track] if track else status).update(fields)
        tmp = path.with_suffix(".tmp")
        write_json(tmp, status)
        tmp.replace(path)
        return status


def clear_results(run_id):
    path = run_dir(run_id)
    for pattern in ("manifest_*.json", "comparison.json", "compliance_*.json", "explanations_*.json", "status.json"):
        for f in path.glob(pattern):
            f.unlink()
    shutil.rmtree(path / "gnn_work", ignore_errors=True)
