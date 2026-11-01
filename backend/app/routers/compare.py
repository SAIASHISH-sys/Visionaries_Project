from fastapi import APIRouter, HTTPException

from .. import store
from ..schemas import ComparisonResult
from ..services import compare

router = APIRouter(prefix="/runs/{run_id}", tags=["compare"])


@router.post("/compare", response_model=ComparisonResult)
def compare_tracks(run_id: str):
    result = compare.compare(run_id)
    if not result["similarity"]:
        raise HTTPException(409, "run a learned track (gnn or raster) before comparing against classical")
    store.write_json(store.run_dir(run_id) / "comparison.json", result)
    return result
