import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
BACKEND = ROOT / "backend"

DATA_DIR = Path(os.environ.get("DATA_DIR", BACKEND / "data"))
RUNS_DIR = DATA_DIR / "runs"

ODA_PATH = os.environ.get("ODA_PATH", "ODAFileConverter")

PHASE_A_CONFIG = Path(os.environ.get("PHASE_A_CONFIG", ROOT / "config.json"))
GNN_DIR = Path(os.environ.get("GNN_DIR", BACKEND / "models" / "gnn"))
GNN_CKPT = GNN_DIR / "best.pt"
GNN_THRESHOLD = GNN_DIR / "head2_threshold.json"
GNN_DEVICE = os.environ.get("GNN_DEVICE")

RASTER_CKPT = Path(os.environ.get("RASTER_CKPT", BACKEND / "models" / "raster" / "best.pt"))
RASTER_ENCODER = os.environ.get("RASTER_ENCODER", "resnet34")
RASTER_MM_PER_PX = float(os.environ.get("RASTER_MM_PER_PX", "10.0"))

RULES_PATH = Path(__file__).parent / "services" / "rules.json"
MIN_CONFIDENCE = 0.7

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
EXPLAIN_MAX_ISSUES = 80

TRACKS = ("classical", "gnn", "raster")
