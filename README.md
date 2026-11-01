# Phase B: Indian CAD floor plans → auto-labelled GNN fine-tuning set

CV5570 Computer Vision in Civil Engineering, IIT Madras. Team Visionaries (Group 8), Track B.

Builds an Indian-drafting fine-tuning set for the Phase A floor-plan GNN (FloorPlanCAD pre-training) from DWG/DXF drawings, with no hand annotation. Output is FloorPlanCAD-compatible graph shards that load with the same `fpgraph` helpers.

## Contents

| File | Purpose |
|---|---|
| `phaseB_indian_autolabel.ipynb` | End-to-end notebook: DWG → DXF → per-floor split → rule-based labels → SVG → tiles → shards → Hugging Face. Runs on Colab (CPU only). |
| `indlabel.py` | The labeller, generated from the notebook's `%%writefile` cells. Re-generated on every notebook run; do not edit by hand. |
| `fpgraph.py`, `config.json` | Phase A parser, graph builder and settings (12-class taxonomy, 10 m tiles). Copied from the Phase A dataset repo; the notebook re-downloads them so Indian tiles are always built identically to the FloorPlanCAD tiles. |
| `phaseCD_gnn_training.ipynb` | Phase C/D notebook: model, training, fine-tuning, and the verification steps. Runs on Colab (GPU). |
| `gnnmodel.py`, `gnndata.py`, `gnneval.py`, `train_gnn.py`, `gnncheck.py`, `gnnsynth.py` | Phase C/D modules, generated from the notebook's `%%writefile` cells: network and loss, data and augmentation, metrics and instance decoding, the training script (single GPU or DDP), the checks, and a synthetic dataset for the smoke test. |
| `phaseE_gnn_inference.ipynb` | Phase E notebook, self-contained: DWG (file picker) + trained GNN from Hugging Face → manifest JSON with the Track A schema (`source: "cv_model"`). Carries the Track A core and the Phase B exporter inside as `%%writefile` cells. Optional comparison with a classical manifest. Runs on Colab (GPU optional). |
| `tracka_core.py` | The Track A geometry core: the definition-only cells of `stage2a_dxf_extraction.ipynb`, copied unchanged into the Phase E notebook. Regenerated on every Phase E run; do not edit by hand. |
| `cvexport.py`, `gnninfer.py`, `cvadapter.py`, `cvcompare.py`, `cvpipeline.py` | Phase E modules, generated from the notebook's `%%writefile` cells: DXF to label-free SVG, tiling and model inference with overlap merging, the adapter into the Track A core, the manifest comparison, and a one-call entry point. |
| `stage2a_dxf_extraction.ipynb`, `manifest_TOWER-B_GF.json` | Track A notebook and its manifest for the TOWER-B ground floor (the classical reference). |
| `requirements.txt` | Package versions used for the reported run (`fpgraph.to_pyg` additionally needs `torch` and `torch_geometric`). Phase E also needs `pydantic` (v2) for the Track A schema. |

## How it works

1. **Convert** DWG to DXF with ODA File Converter (headless via `xvfb`).
2. **Split each master drawing into unique floors.** Each DWG holds every floor as a block insert. Each floor becomes its own DXF, and its role folder (`tower_test`, `tower_val`, `tower_train`) is its split group.
3. **Label** primitives from layer and block names (regex rules). Block inserts become instances. Door subtype comes from leaf-arc geometry (two hinge centres: double, one: single, none: sliding). Units are inferred from door swings and modal wall thickness, not trusted from the header.
4. **Export** a FloorPlanCAD-style SVG per floor (`semantic-id` / `instance-id`, true arcs, mm coordinates, no layer names).
5. **Tile and shard** with `fpgraph` (10 m tiles, 300 mm edges, 12-class taxonomy). Tiles need at least 20 wall nodes and 3% labelled nodes.
6. **Push** derived graphs and metadata to the dataset repo under `indian/`.

## Running it

Provide `HF_TOKEN` as a Colab secret (or environment variable) with write access to the dataset repo set in `HF_REPO_ID` in the config cell. Put the master drawings under `phaseB_input/towera/` and `phaseB_input/towerb/` and run the notebook top to bottom.

## Data policy

Raw DWG/DXF files are not committed or uploaded. `.gitignore` excludes them, `.env`, and all working directories. Only derived tile graphs, labels and statistics go to Hugging Face.

## Split and its limits

TOWER-A and TOWER-B share byte-identical geometry for the ground-floor site plan, the 1st and 2nd floors and the typical 3rd–8th floor, so the 12 plans are 8 unique floors. The split is therefore by unique floor:

| split | floors |
|---|---|
| test | TOWER-B ground floor (whole-site plan; the held-out floor for the classical-vs-GNN comparison) |
| val | TOWER-B 9th floor |
| train | 1st, 2nd, typical 3rd–8th, typical 10th–14th, terrace, upper terrace |

All floors come from one office and share a base block, so val and test are **same-drafter**, not source-disjoint. Out-of-distribution claims must rest on the FloorPlanCAD test split.

Last run produced 319 train, 58 val and 75 test tiles, with zero invariant problems from `fpgraph.compute_stats`.

## Known limitations

- Labels are silver (automatic). They were not hand-verified beyond preview inspection.
- Lifts are drawn without lift-named blocks or layers, so they are labelled `other`. Mask the `elevator` class in the Indian loss during fine-tuning.
- `curtain_wall`, `door_other` and `opening` have no examples in this set.
- `A-BALCONY` and `A-THIN` layers are unlabelled.
- Door and window counts for the ground floor (181 and 115) differ from the Track A manifest (164 and 101) by 10–14%. The files differ and this has not been reconciled.
- Site-level clutter on the ground floor (compound walls, gate cabin, tanks) is labelled as wall.

## Phase C/D: model and training

`phaseCD_gnn_training.ipynb` builds a sparse GAT (8 stages, 8 heads; semantic head plus same-instance edge head), pre-trains it on the FloorPlanCAD shards, fine-tunes on the Indian shards and writes the metrics.

Before running, set the two placeholders in the config cell: `HF_DATASET_REPO` (dataset repo with `graphs/`, `indian/graphs/`, `fpgraph.py`, `config.json`) and `HF_OUTPUT_REPO` (model repo that step 9 uploads checkpoints, logs and configs to). `SMOKE_TEST = True` runs the whole pipeline on a tiny synthetic dataset first.

Verification built into the notebook:

- **Before training (step 4):** tile invariants, augmentation consistency with the Phase A features, model sanity (initial loss, node-order equivariance, no leakage between tiles in a batch, gradients), an overfit-four-tiles test, a learning-rate sweep, and a go/no-go summary.
- **D1 dry run (step 5):** measured peak GPU memory and seconds per step, with a projected total time.
- **After training (steps 6.1 and 7.1):** an automatic diagnosis of the run logs (numerics, learning, gradient spikes, overfitting, per-class F1, Head 2 precision and recall, threshold) with suggested changes, loss and validation curves, and a ground-truth versus prediction picture.

The model is plain PyTorch and does not need PyTorch Geometric.

## Phase E: model output to manifest

`phaseE_gnn_inference.ipynb` runs the fine-tuned GNN on a per-floor DXF and writes a manifest with the same schema as Stage 2a (`schema_version` 0.2, same top-level keys, validated by the same Pydantic `ManifestM`), with `source: "cv_model"` and a `confidence` on every room, opening and wall.

Inputs: a DWG (a file picker opens in step 4; a DXF also works) and the id of the model repo (`HF_OUTPUT_REPO`). Nothing else is uploaded: `fpgraph.py` and `config.json` come from the dataset repo, the checkpoint, the Head 2 threshold and the model modules from the model repo, the DWG is converted with ODA File Converter inside the notebook, and the Track A core and `indlabel.py` are carried in the notebook. A DWG that holds several floors needs `FLOOR_BLOCK` set to the floor to process. A classical manifest (`REFERENCE_MANIFEST`) is optional and only enables the comparison.

How the CV manifest is produced:

- **Same geometry core.** Rooms, door sides, adjacency, corridors and wall thickness come from the unchanged Track A `run()` and `build_manifest()`. The adapter swaps four of its inputs for the model's: the geometry loader, the opening builder, the room tags and the dimension check (`cvadapter.patched`).
- **Model to core inputs.** Primitives predicted `wall`, `curtain_wall`, `railing` become boundary leaves (touching collinear pieces are re-joined after the graph builder's 2 m split); accepted `window` instances become glazing leaves; accepted `door_*` and `window` instances become openings whose footprint, wall axis, jambs and clear width are rebuilt from the instance and the nearby wall segments.
- **Tiles merged.** Class logits and edge logits are averaged over overlapping tiles, then instances are decoded once on the merged graph.
- **Room names** come from `<text>` items in the SVG (a size-looking text is attached to the nearest name). `dimensions` is empty because the SVG has no dimension entities.
- **Confidence** is the minimum softmax probability over the contributing primitives, capped by Track A's own confidence.

Verification done offline on CPU (modules only; the notebook and the trained checkpoint have not been run):

- The adapter fed silver labels as perfect predictions reproduced the classical manifest on a synthetic flat (upright and rotated 30 degrees): 3 of 3 rooms at IoU 1.0, all 5 openings with identical widths, adjacency F1 1.0, wall count 11 of 11.
- Tiling, merging and decoding reproduced the labels exactly on a 12-tile synthetic grid (PQ 1.0, 24 of 24 instances).
- A random-weight model ran through the whole chain.

The notebook adds an **oracle** run (silver labels through the same adapter) so that a gap to the classical manifest can be split into adapter and label limits versus model errors.
