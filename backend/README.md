# Backend

FastAPI service over the three extraction tracks. Run from the repository root, because the research modules (`tracka_core`, `gnninfer`, `rastercv`, ...) live there:

    python -m venv .venv
    .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/Scripts/python -m pip install -r backend/requirements.txt python-dotenv huggingface_hub
    .venv/Scripts/python backend/fetch_models.py
    .venv/Scripts/python -m uvicorn backend.app.main:app

`fetch_models.py` reads `HF_TOKEN` from `.env` and pulls the GNN checkpoint from `saiashish29/GNN-CV` and the raster checkpoint from the `saiashish29/FloorPlanCAD` dataset repo (`models/raster/best.pt`).

## Flow

| Call | Does |
|---|---|
| `POST /runs` (multipart `file`) | Saves the upload, converts DWG to DXF if needed, renders the preview SVG, lists floor blocks. No extraction yet. |
| `PUT /runs/{id}/floor` (`{"floor_block": ...}`) | Chooses which floor block to preview and extract. |
| `POST /runs/{id}/generate` | Background job: classical, then GNN, then raster. Poll `GET /runs/{id}` for per-track state. |
| `GET /runs/{id}`, `GET /runs/{id}/preview.svg`, `GET /runs/{id}/manifest/{track}` | Run status, drawing preview, stored manifests. |
| `POST /runs/{id}/gnn`, `POST /runs/{id}/raster` | Run a single learned track synchronously (needs classical done). |
| `POST /runs/{id}/compare` | Similarity of each learned manifest to classical, plus the room and opening matches. |
| `POST /runs/{id}/compliance?track=classical` | Rule checks on a stored manifest: severity, margin, confidence, not-checked rooms. |
| `POST /runs/{id}/compliance/explain?track=` | Gemini-written explanation per failed check (needs `GEMINI_API_KEY`; advisory). |
| `POST /compliance` (manifest JSON body) | Rule checks on any manifest. |

Every manifest carries `pipeline.track` (`classical`, `gnn` or `raster`).

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `DATA_DIR` | `backend/data` (runs are stored under `runs/<id>/`) |
| `ODA_PATH` | `ODAFileConverter` (needed for DWG input) |
| `GNN_DIR` | `backend/models/gnn` with `best.pt` and `head2_threshold.json` |
| `PHASE_A_CONFIG` | `config.json` in the repository root |
| `RASTER_CKPT` | `backend/models/raster/best.pt` |
| `RASTER_ENCODER`, `RASTER_MM_PER_PX`, `GNN_DEVICE` | `resnet34`, `10.0`, auto |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | unset, `gemini-3.5-flash-lite` (read from `.env`; confirm the model name) |

A missing checkpoint gives `503` on that track only.

## Rules

Thresholds live in `app/services/rules.json`, carried over unchanged from the compliance notebook. Severities, rationales and remedies there are first drafts, and every `clause` is `null` until it is tied to the code text.

## Tests

    .venv/Scripts/python -m pip install -r backend/requirements-dev.txt
    .venv/Scripts/python -m pytest backend/tests
