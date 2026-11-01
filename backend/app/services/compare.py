import cvcompare

from .. import store


def compare(run_id):
    reference = store.load_manifest(run_id, "classical")
    out = []
    for track in store.available_tracks(run_id):
        if track == "classical":
            continue
        other = store.load_manifest(run_id, track)
        sim = cvcompare.similarity(other, reference)
        out.append({"track": track, **sim, "matches": cvcompare.matches(other, reference), "details": cvcompare.compare_manifests(other, reference).to_dict("records")})
    return {"run_id": run_id, "reference": "classical", "similarity": out}
