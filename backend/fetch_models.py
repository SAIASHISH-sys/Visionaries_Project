import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "backend" / "models"

GNN_REPO = ("saiashish29/GNN-CV", "model")
RASTER_REPO = ("saiashish29/FloorPlanCAD", "dataset")

FILES = [
    (GNN_REPO, "finetune/best.pt", MODELS / "gnn" / "best.pt"),
    (GNN_REPO, "finetune/head2_threshold.json", MODELS / "gnn" / "head2_threshold.json"),
    (RASTER_REPO, "models/raster/best.pt", MODELS / "raster" / "best.pt"),
]


def main():
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN")
    for (repo, repo_type), name, dest in FILES:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo, name, repo_type=repo_type, token=token)
        shutil.copy(cached, dest)
        
        print(dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
