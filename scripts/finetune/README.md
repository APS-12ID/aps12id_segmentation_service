# Finetune pipeline

Semi-supervised SAM3 finetune workflow for aps12id sample-plate imagery.
Everything downstream (training, evaluation, deployment) reads a single
canonical dataset shape: **COCO instance segmentation with compressed-RLE
masks**. Every producer here emits that shape.

Two plate types, four categories, one dataset:

| id | name          | supercategory | source                                |
|----|---------------|---------------|---------------------------------------|
| 1  | hole          | gcp           | Label Studio project #1               |
| 2  | sample        | gcp           | Label Studio project #1               |
| 3  | slit          | capillary     | Label Studio project #1               |
| 4  | capillary tube| capillary     | Label Studio project #1               |

## Strategy

Semi-supervised loop, four steps:

1. **Auto-generate labels.** For GCP plates, prompt vanilla SAM3 with
   `text "hole"` and coordinate prompts inside each hole for the sample.
   For capillary plates, vanilla SAM3 is weak so bootstrap only gives a
   starting point; expect more manual work.
2. **Correct the failures in Label Studio.** The LS project on
   `sentosa.xray.aps.anl.gov:8080/projects/1` is the annotation UI. All
   four categories are configured in one project. Bootstrap output is
   loaded as pre-annotations so annotators correct rather than draw.
3. **Migrate LS state to a canonical COCO** via `ls_to_coco.py`.
4. **Fine-tune SAM3** on the COCO. Freeze both encoders, train the mask
   decoder with dice + focal mask loss. Focal loss handles class
   imbalance across the four categories.

All categories are treated the same at both training and inference time:
text prompt per category, no interactive point prompts, no
per-category prompting tricks. Keeping the recipe uniform avoids
overfitting to any one workflow.

## Modules

- `coco_schema.py` — category IDs, deterministic image/annotation IDs, and
  polygon↔RLE helpers. Frozen schema so partial COCOs can be merged without
  ID churn.
- `ls_to_coco.py` — one-shot migration of the Label Studio project. Reads
  the LS sqlite directly, rasterizes percent-coord polygons AND rectangles
  (converted to 4-corner polygons) to masks, emits RLE-encoded COCO. Uses
  `LS_LABEL_TO_CATEGORY_ID` in `coco_schema.py` to collapse the shape
  suffix on the LS class value into the canonical COCO category. Preserves
  both hand-drawn labels and SAM3 pre-annotations that were edited.
- `sam3_bootstrap.py` — vanilla SAM3 pseudo-labels for either plate type.
  `--plate-type gcp` reproduces the samH/V hole→sample pipeline.
  `--plate-type capillary` uses text-only prompts for slit + capillary tube.
- `dataset.py` — `GcpCocoDataset`. Reads a canonical COCO and yields one
  `Sample` per (image, category) pair with ≥1 instance. Deterministic
  train/val split by sorted image_id. Consumed by `train.py`.
- `train.py` — decoder-only fine-tune of SAM3. Freezes
  `backbone.vision_backbone.*` and `backbone.language_backbone.*`, trains
  the segmentation head + transformer decoder with dice + focal mask loss
  (weights from Meta's `roboflow_v100_full_ft_100_images.yaml`:
  `loss_mask=200.0`, `loss_dice=10.0`). Runs a per-epoch val loop
  computing per-category IoU; saves `best.pt` on improvement. Reads
  hyperparameters from a YAML config; logs to MLflow when
  `MLFLOW_TRACKING_URI` is set.

## Annotation (Label Studio)

Live project: **http://sentosa.xray.aps.anl.gov:8080/projects/1/data?tab=1**
(intranet only; requires ANL network / VPN).

Class values in LS are shape-suffixed so annotators can see at a glance
which drawing tool a label belongs to. `ls_to_coco.py` collapses the
suffix on ingest — training and downstream evaluation see one canonical
category per physical class.

| LS class value              | drawing tool | COCO category id / name |
|-----------------------------|--------------|-------------------------|
| `hole_polygon`              | polygon      | 1 / hole                |
| `sample_polygon`            | polygon      | 2 / sample              |
| `slit_polygon`              | polygon      | 3 / slit                |
| `slit_rectangle`            | rectangle    | 3 / slit                |
| `capillary_tube_polygon`    | polygon      | 4 / capillary tube      |
| `capillary_tube_rectangle`  | rectangle    | 4 / capillary tube      |

`BrushLabels` is retired. Any residual brush regions are dropped by
`ls_to_coco.py`. New annotations should use polygon; the rectangle tool
is kept for the existing rect-drawn slit annotations that render for
review.

Regenerate the canonical COCO from the current LS state whenever a training
run is about to start (see the operator quick-start below).

### Dataset state

Task ID ranges (project #1):

| range     | tasks | notes                                                 |
|-----------|-------|-------------------------------------------------------|
| 1–100     | 100   | GCP plates (hole + sample). Sample masks under review.|
| 463–554   | 92    | Mixed GCP + capillary (all four categories).          |

Instance counts across the current LS state (post-purge, single brush
region removed):

| range   | hole | sample | slit (poly / rect) | capillary tube |
|---------|------|--------|--------------------|----------------|
| 1–100   | 688  | 220    | —                  | —              |
| 463–554 | 398  | 172    | 231 / 101          | 137            |

Task IDs 101–462 were purged on 2026-07-17 — stale annotations from an
earlier label schema that predate the current 4-category setup. Backup
of the pre-purge sqlite is at
`~/aps12id_seg_finetune/label_studio/data/label_studio.sqlite3.bak.20260717_161213`
on sentosa.

## Round 1: GCP decoder fine-tune

Round 1 validated the full pipeline end-to-end on the GCP hole/sample
annotations before touching capillary. Recipe: freeze both encoders, train
the mask decoder, dice + focal losses. Runs on sentosa's H200 idx 1
(`CUDA_VISIBLE_DEVICES=1`, see `~/.claude/projects/-Users-haskels/memory/reference_sentosa.md`).

```bash
cd ~/aps12id_seg_finetune/repo
git checkout finetune-pipeline && git pull
uv sync

# 1. Refresh the GCP COCO from the current LS state
uv run python -m scripts.finetune.ls_to_coco \
    --db ~/aps12id_seg_finetune/label_studio/data/label_studio.sqlite3 \
    --project-id 1 \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --out ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json

# 2. Baseline: eval vanilla SAM3 on the val split to establish a floor
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.finetune.train \
    --config     scripts/finetune/configs/gcp_r1.yaml \
    --base-checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out-dir    ~/aps12id_seg_finetune/runs/gcp_r1_baseline \
    --eval-only

# 3. Fine-tune (50 epochs, ~100 min on H200)
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.finetune.train \
    --config     scripts/finetune/configs/gcp_r1.yaml \
    --base-checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out-dir    ~/aps12id_seg_finetune/runs/gcp_r1 \
    --epochs 50

# Watch per-epoch val IoU
tail -f ~/aps12id_seg_finetune/runs/gcp_r1/metrics.jsonl
```

Fine-tuned `best.pt` / `last.pt` are not committed (too large) — they stay
under `~/aps12id_seg_finetune/runs/gcp_r1/` on sentosa; the branch push
message links to them with the val numbers.

### Round 1 results (2026-07-16) — DONE

Split: 526 images (473 train / 53 val, 4975 annotations) from the
pre-purge dataset. Ran 50 epochs on sentosa H200 idx 1 at bs=1, bf16
autocast, ~130 s/epoch (~108 min total). Trained 32.7M / 840.5M params
(3.9%).

| checkpoint          | hole IoU | sample IoU | val_loss                       |
|---------------------|----------|------------|--------------------------------|
| baseline (vanilla)  | 0.583    | 0.000      | 295.12                         |
| fine-tuned best.pt  | 0.926    | 0.536      | 150.15 (train) / 161.58 (rerun)|
| fine-tuned last.pt  | 0.923    | **0.606**  | 181.83                         |

Vanilla SAM3 already sees "hole" from the text prompt but cannot locate
"sample" at all (0.0% IoU). Fine-tuning learns the sample geometry from
scratch and lifts hole segmentation by 34 percentage points.

`best.pt` was selected by val_loss (epoch 31) but `last.pt` (epoch 50) gives
noticeably higher sample IoU with almost identical hole IoU — val_loss and
per-category IoU disagree because the loss mixes bbox / ce / mask / dice
components and sample instances are heavily outnumbered by holes (1265 vs
3710 annotations). Recommend `last.pt` for deployment unless downstream
evaluation prefers the best-loss checkpoint.

Both live at `~/aps12id_seg_finetune/runs/gcp_r1/{best,last}.pt` on sentosa
(3.2 GB each). `metrics.jsonl` in the same directory has the per-epoch
train/val trace.

Note: the round 1 train/val split was derived from the pre-purge task
list. The current LS state (192 tasks) does not reproduce that split
exactly — for an apples-to-apples r1 vs r2 comparison, re-eval both
checkpoints against the current val split rather than trusting the
round 1 numbers above.

## Round 2: joint GCP + capillary

Round 2 is a single joint model over all four categories, trained on
one merged COCO derived from the current LS state. It re-fine-tunes
from the SAM3 **base** checkpoint (not round 1's `best.pt`) so the
strong hole/sample prior from round 1 does not suppress the capillary
categories at init.

Preconditions before kicking off:

- Sample masks in tasks 1–100 have been reviewed.
- Annotations in tasks 463–554 are final.

Trainer changes:

- Expand `GCP_CATEGORIES = ("hole", "sample")` in `train.py` to include
  `slit` and `capillary tube`.
- Load the merged COCO via `merge_cocos()` in `coco_schema.py`.
- Keep dice + focal losses at existing weights; focal loss carries the
  class imbalance (hole and sample dominate on GCP plates; slit and
  capillary tube are 100 % of capillary plates but appear on fewer
  images). If per-category val IoU still shows one category lagging by
  epoch 15, add per-category loss weighting or dataset-level oversampling
  of the minority plate type.
- Same freeze-encoders + decoder-only recipe as round 1.
- Same H200 idx 1, bs=1, bf16, ~50 epochs.

Log per-category train / val loss and per-category IoU to MLflow so the
per-category trend is visible in one place.

Config lives under `scripts/finetune/configs/r2_joint.yaml`. Commit the
config alongside any recipe change so runs are reproducible from history.

## Operator quick-start

Assumes the sentosa working tree at `~/aps12id_seg_finetune/` is populated
and the repo's uv env is synced.

```bash
cd ~/aps12id_seg_finetune/repo
uv sync

# 1. Migrate the current LS state to COCO
uv run python -m scripts.finetune.ls_to_coco \
    --db ~/aps12id_seg_finetune/label_studio/data/label_studio.sqlite3 \
    --project-id 1 \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --out ~/aps12id_seg_finetune/data/labels/canonical/joint.coco.json

# 2. (Optional, for capillary pre-annotations) bootstrap with vanilla SAM3
uv run python -m scripts.finetune.sam3_bootstrap \
    --plate-type capillary \
    --image-root ~/aps12id_seg_finetune/data/raw/capillary \
    --checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out ~/aps12id_seg_finetune/data/labels/canonical/capillary.coco.json \
    --summary ~/aps12id_seg_finetune/data/labels/canonical/capillary.summary.jsonl

# 3. Train — see the Round 1 block above for the full invocation shape.
```

## Reviewing results

Annotations are reviewed in the LS project at
`http://sentosa.xray.aps.anl.gov:8080/projects/1/data?tab=1`. Per-run
metrics are in MLflow (when configured) and in `metrics.jsonl` under the
run's output directory on sentosa. Checkpoints stay on sentosa; the
committed `data/labels/canonical/*.coco.json` files are the current
dataset snapshot.
