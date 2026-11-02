import math
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import jobs, settings, store
from ..schemas import FloorChoice, RunInfo
from ..services import ingest, preview

router = APIRouter(prefix="/runs", tags=["runs"])


def unit_warning(status, info):
    classical = status["classical_mm_factor"]
    if classical and not math.isclose(classical, info["mm_factor"], rel_tol=1e-6):
        return (f"preview scale ({info['mm_factor']} mm/unit) differs from the classical extraction ({classical} mm/unit); "
                "overlays on the plan may be misaligned")
    return None


def run_info(run_id):
    meta, status, info = store.read_meta(run_id), store.read_status(run_id), preview.describe(run_id)
    return RunInfo(run_id=run_id, source_file=meta["source_file"], floors=meta["floors"], floor_block=meta["floor_block"],
                   preview=info, tracks=status["tracks"], unit_warning=unit_warning(status, info))


@router.post("", response_model=RunInfo)
def create_run(file: UploadFile = File(...)):
    run_id, path = store.new_run()
    try:
        dxf = ingest.to_dxf(ingest.save_upload(file, path), path)
        floors, info = preview.render(dxf, path)
    except HTTPException:
        shutil.rmtree(path, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(path, ignore_errors=True)
        raise HTTPException(422, f"could not read the drawing: {e!r}")
    store.write_meta(run_id, {"source_file": file.filename, "floors": floors, "floor_block": info["floor_block"]})
    return run_info(run_id)


@router.get("/{run_id}", response_model=RunInfo)
def get_run(run_id: str):
    return run_info(run_id)


@router.put("/{run_id}/floor", response_model=RunInfo)
def choose_floor(run_id: str, choice: FloorChoice):
    meta = store.read_meta(run_id)
    if choice.floor_block not in {f["block"] for f in meta["floors"]}:
        raise HTTPException(422, "unknown floor block")
    try:
        preview.render(store.dxf_path(run_id), store.run_dir(run_id), choice.floor_block)
    except Exception as e:
        raise HTTPException(422, f"could not render that floor: {e!r}")
    store.write_meta(run_id, {**meta, "floor_block": choice.floor_block})
    store.clear_results(run_id)
    return run_info(run_id)


@router.post("/{run_id}/generate", response_model=RunInfo, status_code=202)
def generate(run_id: str):
    store.run_dir(run_id)
    jobs.start(run_id)
    return run_info(run_id)


@router.post("/{run_id}/cancel", response_model=RunInfo)
def cancel(run_id: str):
    store.run_dir(run_id)
    jobs.cancel(run_id)
    return run_info(run_id)


@router.get("/{run_id}/preview.svg")
def preview_svg(run_id: str):
    return FileResponse(store.run_dir(run_id) / "preview.svg", media_type="image/svg+xml")


@router.get("/{run_id}/manifest/{track}")
def get_manifest(run_id: str, track: str):
    if track not in settings.TRACKS:
        raise HTTPException(404, f"track must be one of {settings.TRACKS}")
    return store.load_manifest(run_id, track)
