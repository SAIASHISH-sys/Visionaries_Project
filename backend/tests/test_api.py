import copy
import time

from backend.app import store
from backend.app.services import explain, raster


def wait_for_generation(client, run_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        tracks = client.get(f"/runs/{run_id}").json()["tracks"]
        if all(t["state"] in ("done", "failed") for t in tracks.values()):
            return tracks
        time.sleep(0.5)
    raise AssertionError("generation did not finish")


def test_upload_returns_preview_without_extraction(client, run_id):
    info = client.get(f"/runs/{run_id}").json()
    assert info["floors"][0]["block"] == "GROUND FLOOR PLAN"
    assert all(t["state"] == "idle" for t in info["tracks"].values())
    svg = client.get(info["preview"]["url"])
    assert svg.status_code == 200 and svg.headers["content-type"].startswith("image/svg+xml")
    assert len(info["preview"]["origin_mm"]) == 2


def test_rejects_other_file_types(client):
    assert client.post("/runs", files={"file": ("x.txt", b"hi")}).status_code == 415
    assert client.get("/runs/missing").status_code == 404


def test_floor_choice_validates_block(client, run_id):
    assert client.put(f"/runs/{run_id}/floor", json={"floor_block": "NOPE"}).status_code == 422
    ok = client.put(f"/runs/{run_id}/floor", json={"floor_block": "GROUND FLOOR PLAN"})
    assert ok.status_code == 200


def test_generate_runs_all_tracks(client, run_id):
    assert client.post(f"/runs/{run_id}/generate").status_code == 202
    tracks = wait_for_generation(client, run_id)
    assert tracks["classical"]["state"] == "done" and tracks["classical"]["summary"]["rooms"] == 2
    for name in ("gnn", "raster"):
        assert tracks[name]["state"] in ("done", "failed")
    manifest = client.get(f"/runs/{run_id}/manifest/classical").json()
    assert manifest["pipeline"]["track"] == "classical"
    assert client.get(f"/runs/{run_id}").json()["unit_warning"] is None


def test_compliance_fields_and_not_checked(client, run_id):
    client.post(f"/runs/{run_id}/generate")
    wait_for_generation(client, run_id)
    manifest = store.load_manifest(run_id, "classical")
    manifest["rooms"][0]["area_m2"] = 3.0
    manifest["rooms"][0]["raw_label"] = "M.BEDROOM"
    manifest["rooms"][1]["confidence"] = 0.4
    store.save_manifest(run_id, "classical", manifest)
    report = client.post(f"/runs/{run_id}/compliance").json()
    fail = next(v for v in report["violations"] if v["rule_id"] == "min_area")
    assert fail["severity"] == "major" and fail["margin_pct"] < 0 and fail["clause"] is None
    assert report["by_severity"]["major"] >= 1
    assert report["not_checked"] == [{"room_id": manifest["rooms"][1]["id"], "room": manifest["rooms"][1]["raw_label"], "reason": "low confidence (0.40)"}]


def test_compare_returns_matches(client, run_id):
    client.post(f"/runs/{run_id}/generate")
    wait_for_generation(client, run_id)
    reference = store.load_manifest(run_id, "classical")
    model = copy.deepcopy(reference)
    model["rooms"] = model["rooms"][:1]
    store.save_manifest(run_id, "gnn", model)
    result = client.post(f"/runs/{run_id}/compare").json()
    gnn = next(s for s in result["similarity"] if s["track"] == "gnn")
    assert len(gnn["matches"]["rooms"]["pairs"]) == 1
    assert gnn["matches"]["rooms"]["ref_only"] == [reference["rooms"][1]["id"]]
    assert 0 < gnn["score"] < 1


def test_explain_requires_report_and_key(client, run_id, monkeypatch):
    assert client.post(f"/runs/{run_id}/compliance/explain").status_code == 409
    client.post(f"/runs/{run_id}/generate")
    wait_for_generation(client, run_id)
    manifest = store.load_manifest(run_id, "classical")
    manifest["rooms"][0]["area_m2"] = 3.0
    manifest["rooms"][0]["raw_label"] = "M.BEDROOM"
    store.save_manifest(run_id, "classical", manifest)
    client.post(f"/runs/{run_id}/compliance")
    monkeypatch.setattr(explain.settings, "GEMINI_API_KEY", None)
    assert client.post(f"/runs/{run_id}/compliance/explain").status_code == 503
    monkeypatch.setattr(explain.settings, "GEMINI_API_KEY", "test")
    key = next(v["key"] for v in client.post(f"/runs/{run_id}/compliance").json()["violations"])
    monkeypatch.setattr(explain, "call_model", lambda prompt: '{"%s": {"explanation": "too small", "fix": "enlarge"}, "invented": {"explanation": "x", "fix": "y"}}' % key)
    out = client.post(f"/runs/{run_id}/compliance/explain").json()
    assert out["advisory"] is True and list(out["explanations"]) == [key]


def test_raster_manifest_is_comparable_shape():
    struct = {"rooms": [{"id": "P000", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "area_m2": 1.0, "enclosure": "closed"}],
              "openings": [], "adjacency": [], "seconds": 0.1}
    m = raster.to_manifest(struct, (0, 0, 10, 10), 10.0, "best.pt")
    assert {"rooms", "openings", "walls", "adjacency", "corridors", "vertical_circulation", "unassigned_labels"} <= set(m)
