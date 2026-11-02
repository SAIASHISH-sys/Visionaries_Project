import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import HTTPException

from . import settings, store
from .services import pipeline

_pool = ThreadPoolExecutor(max_workers=1)
ACTIVE = {"queued", "running"}
_cancel_flags: set[str] = set()


def cancel(run_id):
    status = store.read_status(run_id)
    if not any(t["state"] in ACTIVE for t in status["tracks"].values()):
        raise HTTPException(409, "no generation is running for this run")
    _cancel_flags.add(run_id)
    for track in settings.TRACKS:
        if status["tracks"][track]["state"] == "queued":
            store.update_status(run_id, track, state="cancelled", error="cancelled by user")


def start(run_id):
    status = store.read_status(run_id)
    if any(t["state"] in ACTIVE for t in status["tracks"].values()):
        raise HTTPException(409, "generation is already running for this run")
    cached = {t for t in settings.TRACKS if store.manifest_path(run_id, t).exists()}
    store.clear_results(run_id, keep_tracks=cached)
    for track in settings.TRACKS:
        if track in cached:
            existing = store.read_json(store.manifest_path(run_id, track))
            summary = {k: v for k, v in existing.get("summary", {}).items() if isinstance(v, (int, float))}
            store.update_status(run_id, track, state="done", error=None, seconds=None, summary=summary)
        else:
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


_learned_pool = ThreadPoolExecutor(max_workers=2)


def _fingerprint(manifest):
    s = manifest.get("summary", {})
    return (s.get("rooms", 0), s.get("doors", 0), s.get("windows", 0), s.get("walls", 0))


def _find_donor(run_id, track):
    """Return a run_id that has a matching classical fingerprint and a completed track manifest."""
    try:
        my_fp = _fingerprint(store.load_manifest(run_id, "classical"))
    except Exception:
        return None
    for candidate_dir in settings.RUNS_DIR.iterdir():
        candidate = candidate_dir.name
        if candidate == run_id or not candidate_dir.is_dir():
            continue
        donor_manifest = candidate_dir / f"manifest_{track}.json"
        classical_manifest = candidate_dir / "manifest_classical.json"
        if not donor_manifest.exists() or not classical_manifest.exists():
            continue
        try:
            if _fingerprint(store.read_json(classical_manifest)) == my_fp:
                return candidate
        except Exception:
            continue
    return None


def _reuse_from_donor(run_id, track, donor):
    import shutil
    src = store.manifest_path(donor, track)
    dst = store.manifest_path(run_id, track)
    shutil.copy2(src, dst)
    manifest = store.read_json(dst)
    store.update_status(run_id, track, state="done", error=None, seconds=0,
                        summary={**_summary(manifest), "reused_from": donor})


def _run(run_id):
    cached = {t for t in settings.TRACKS if store.manifest_path(run_id, t).exists()}
    if "classical" in cached:
        classical_ok = True
    else:
        classical_ok = _run_one(run_id, "classical")

    learned = [t for t in settings.TRACKS if t != "classical" and t not in cached]

    if run_id in _cancel_flags:
        for track in learned:
            store.update_status(run_id, track, state="cancelled", error="cancelled by user")
        _cancel_flags.discard(run_id)
        return

    if not classical_ok:
        for track in learned:
            store.update_status(run_id, track, state="failed", error="classical extraction failed; this track depends on it")
        _cancel_flags.discard(run_id)
        return

    # Check if any learned track can be reused from a prior run with matching classical output
    to_run = []
    for track in learned:
        donor = _find_donor(run_id, track)
        if donor:
            _reuse_from_donor(run_id, track, donor)
        else:
            to_run.append(track)

    futures = {_learned_pool.submit(_run_one, run_id, track): track for track in to_run}
    for fut in as_completed(futures):
        fut.result()

    _cancel_flags.discard(run_id)
