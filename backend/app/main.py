from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from . import settings
from .routers import compare, compliance, runs, tracks

settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Floor plan compliance API")
app.add_middleware(GZipMiddleware, minimum_size=1024)
for module in (runs, tracks, compare, compliance):
    app.include_router(module.router)


@app.get("/health")
def health():
    return {"status": "ok"}
