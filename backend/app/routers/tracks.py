from fastapi import APIRouter

from ..schemas import TrackResult
from ..services import pipeline

router = APIRouter(prefix="/runs/{run_id}", tags=["tracks"])


def respond(run_id, track):
    manifest = pipeline.run_track(run_id, track)
    return TrackResult(run_id=run_id, track=track, manifest_url=f"/runs/{run_id}/manifest/{track}",
                       summary=manifest["summary"])


@router.post("/gnn", response_model=TrackResult)
def run_gnn(run_id: str):
    return respond(run_id, "gnn")


@router.post("/raster", response_model=TrackResult)
def run_raster(run_id: str):
    return respond(run_id, "raster")
