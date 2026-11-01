import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

from . import settings, store
from .services import pipeline

_pool = ThreadPoolExecutor(max_workers=1)
ACTIVE = {"queued", "running"}


def start(run_id):
    status = store.read_status(run_id)
    if any(t["state"] in ACTIVE for t in status["tracks"].values()):
        raise HTTPException(409, "generation is already running for this run")
    store.clear_results(run_id)
    for track in settings.TRACKS:
        store.update_status(run_id, track, state="queued")
    _pool.submit(_run, run_id)


def _summary(manifest):
    return {k: v for k, v in manifest["summary"].items() if isinstance(v, (int, float))}


def _run_one(run_id, track):
    store.update_status(run_id, track, state="running")
    t0 = time.time()
    try:
        manifest = pipeline.run_track(run_id, track)
    except HTTPException as e:
        store.update_status(run_id, track, state="failed", error=str(e.detail), seconds=round(time.time() - t0, 1))
        return False
    except Exception as e:
        store.update_status(run_id, track, state="failed", error=repr(e), seconds=round(time.time() - t0, 1))
        return False
    store.update_status(run_id, track, state="done", error=None, seconds=round(time.time() - t0, 1), summary=_summary(manifest))
    if track == "classical":
        store.update_status(run_id, classical_mm_factor=manifest["scale_resolution"].get("normalised_to_mm_factor"))
    return True


def _run(run_id):
    classical_ok = _run_one(run_id, "classical")
    for track in settings.TRACKS:
        if track == "classical":
            continue
        if classical_ok:
            _run_one(run_id, track)
        else:
            store.update_status(run_id, track, state="failed", error="classical extraction failed; this track depends on it")
