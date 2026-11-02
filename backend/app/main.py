import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from . import settings
from .routers import compare, compliance, runs, tracks

settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)

# On startup, any run left in queued/running from a dead process gets reset to idle
_ACTIVE = {"queued", "running"}
for _status_file in settings.RUNS_DIR.glob("*/status.json"):
    try:
        _s = json.loads(_status_file.read_text())
        _changed = False
        for _t in _s.get("tracks", {}).values():
            if _t.get("state") in _ACTIVE:
                _t["state"] = "idle"
                _t["error"] = None
                _changed = True
        if _changed:
            _status_file.write_text(json.dumps(_s, indent=1))
    except Exception:
        pass

app = FastAPI(title="Floor plan compliance API")
app.add_middleware(GZipMiddleware, minimum_size=1024)
for module in (runs, tracks, compare, compliance):
    app.include_router(module.router)


@app.get("/health")
def health():
    return {"status": "ok"}
