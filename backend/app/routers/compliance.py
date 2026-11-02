from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from .. import settings, store
from ..schemas import ComplianceReport, ExplanationSet
from ..services import compliance, explain

router = APIRouter(tags=["compliance"])


def check_track(track):
    if track not in settings.TRACKS:
        raise HTTPException(404, f"track must be one of {settings.TRACKS}")
    return track


def check(manifest):
    if not isinstance(manifest.get("rooms"), list):
        raise HTTPException(422, "manifest has no rooms list")
    return compliance.check_manifest(manifest)


@router.post("/runs/{run_id}/compliance", response_model=ComplianceReport)
def check_run(run_id: str, track: str = "classical"):
    report = {"run_id": run_id, "track": check_track(track), **check(store.load_manifest(run_id, track))}
    store.write_json(store.run_dir(run_id) / f"compliance_{track}.json", report)
    return report


@router.post("/runs/{run_id}/compliance/explain", response_model=ExplanationSet)
def explain_run(run_id: str, track: str = "classical"):
    path = store.run_dir(run_id) / f"compliance_{check_track(track)}.json"
    if not path.exists():
        raise HTTPException(409, f"run the {track} compliance check first")
    result = explain.explain(store.read_json(path))
    store.write_json(store.run_dir(run_id) / f"explanations_{track}.json", result)
    return result


@router.post("/compliance", response_model=ComplianceReport)
def check_manifest(manifest: dict = Body(...)):
    return check(manifest)


class QueryRequest(BaseModel):
    question: str
    track: str = "classical"


@router.post("/runs/{run_id}/query")
def query_run(run_id: str, body: QueryRequest):
    if not settings.GEMINI_API_KEY:
        raise HTTPException(503, "GEMINI_API_KEY is not set")
    track = check_track(body.track)
    manifest = store.load_manifest(run_id, track)
    compliance_path = store.run_dir(run_id) / f"compliance_{track}.json"
    compliance_data = store.read_json(compliance_path) if compliance_path.exists() else None
    answer = explain.query(body.question, manifest, compliance_data)
    return {"answer": answer}
