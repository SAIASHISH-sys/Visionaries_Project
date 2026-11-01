import os
import sys
import tempfile
from pathlib import Path

import ezdxf
import pytest

os.environ["DATA_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from backend.app.main import app


def build_dxf(path):
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    for name in ("A-WALL", "A-DOOR", "A-RD"):
        doc.layers.add(name)
    door = doc.blocks.new("D1-900")
    door.add_line((0, 0), (900, 0), dxfattribs={"layer": "A-DOOR"})
    door.add_arc((0, 0), 900, 0, 90, dxfattribs={"layer": "A-DOOR"})
    tag = doc.blocks.new("RD")
    tag.add_attdef("RT", (0, 0), height=150, dxfattribs={"layer": "A-RD"})
    tag.add_attdef("SZ", (0, -200), height=100, dxfattribs={"layer": "A-RD"})
    floor = doc.blocks.new("GROUND FLOOR PLAN")
    lines = [((0, 0), (7600, 0)), ((7600, 0), (7600, 3400)), ((7600, 3400), (0, 3400)), ((0, 3400), (0, 0)),
             ((200, 200), (4000, 200)), ((4000, 3200), (200, 3200)), ((200, 3200), (200, 200)),
             ((4400, 200), (7400, 200)), ((7400, 200), (7400, 3200)), ((7400, 3200), (4400, 3200)),
             ((4000, 200), (4000, 1000)), ((4000, 1900), (4000, 3200)),
             ((4400, 200), (4400, 1000)), ((4400, 1900), (4400, 3200))]
    for a, b in lines:
        floor.add_line(a, b, dxfattribs={"layer": "A-WALL"})
    floor.add_blockref("D1-900", (4000, 1000), dxfattribs={"layer": "A-DOOR", "rotation": 90})
    for name, size, pos in (("M.BEDROOM", "3800X3000", (800, 1700)), ("TOILET", "3000X3000", (5000, 1700))):
        floor.add_blockref("RD", pos, dxfattribs={"layer": "A-RD"}).add_auto_attribs({"RT": name, "SZ": size})
    doc.modelspace().add_blockref("GROUND FLOOR PLAN", (0, 0))
    doc.saveas(path)


@pytest.fixture(scope="session")
def dxf_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("dxf") / "tiny.dxf"
    build_dxf(path)
    return path


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def run_id(client, dxf_file):
    r = client.post("/runs", files={"file": ("tiny.dxf", dxf_file.read_bytes())})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]
