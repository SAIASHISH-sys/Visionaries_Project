import threading

COMPUTE_LOCK = threading.Lock()


def stamp(manifest, track, run_id, source_file):
    manifest["source_file"] = source_file
    manifest["pipeline"] = {"track": track, "run_id": run_id}
    return manifest
