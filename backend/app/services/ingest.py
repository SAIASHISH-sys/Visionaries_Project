import shutil
from pathlib import Path

from fastapi import HTTPException

from .. import settings

SUFFIXES = {".dxf", ".dwg"}


def save_upload(upload, run_path):
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUFFIXES:
        raise HTTPException(415, "upload a .dxf or .dwg file")
    src = run_path / f"input{suffix}"
    with src.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return src


def to_dxf(src, run_path):
    dxf = run_path / "floor.dxf"
    if src.suffix == ".dxf":
        src.replace(dxf)
        return dxf
    import indlabel

    stage, out = run_path / "dwg_in", run_path / "dxf_out"
    stage.mkdir()
    shutil.move(str(src), stage / src.name)
    indlabel.convert_dwgs(stage, out, settings.ODA_PATH)
    converted = out / "input.dxf"
    if not converted.exists():
        raise HTTPException(422, "DWG to DXF conversion failed: check ODA_PATH and that the DWG opens in ODA")
    converted.replace(dxf)
    shutil.rmtree(stage)
    shutil.rmtree(out)
    return dxf
