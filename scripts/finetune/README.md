# Finetune pipeline

Semi-supervised SAM3 finetune workflow for aps12id sample-plate imagery.
Everything downstream (training, evaluation, deployment) reads a single
canonical dataset shape: **COCO instance segmentation with compressed-RLE
masks**. Every producer here emits that shape.

Two plate types, four categories, one dataset:

| id | name          | supercategory | source                        |
|----|---------------|---------------|-------------------------------|
| 1  | hole          | gcp           | Label Studio (Sam + Ming)     |
| 2  | sample        | gcp           | Label Studio (Sam + Ming)     |
| 3  | slit          | capillary     | Label Studio (Sam + Ming)     |
| 4  | capillary tube| capillary     | Label Studio (Sam + Ming)     |

## Modules

- `coco_schema.py` — category IDs, deterministic image/annotation IDs, and
  polygon↔RLE helpers. Frozen schema so partial COCOs can be merged without
  ID churn.
- `ls_to_coco.py` — one-shot migration of the Label Studio project. Reads
  the LS sqlite directly, rasterizes percent-coord polygons to masks, emits
  RLE-encoded COCO. Preserves both hand-drawn labels and SAM3
  pre-annotations that the annotator edited.
- `sam3_bootstrap.py` — vanilla SAM3 pseudo-labels for either plate type.
  `--plate-type gcp` reproduces the existing samH/V hole→sample pipeline.
  `--plate-type capillary` uses text-only prompts for slit + capillary tube.
  Bootstrap output is imported into LS so the annotator only has to correct,
  not draw from scratch.
- `dataset.py` — `GcpCocoDataset`. Reads a canonical COCO and yields one
  `Sample` per (image, category) pair with ≥1 instance. Deterministic
  train/val split by sorted image_id. Consumed by `train.py`.
- `train.py` — decoder-only fine-tune of SAM3. Freezes
  `backbone.vision_backbone.*` and `backbone.language_backbone.*`, trains
  the segmentation head + transformer decoder with dice+focal mask loss
  (weights from Meta's `roboflow_v100_full_ft_100_images.yaml` commented
  block: `loss_mask=200.0`, `loss_dice=10.0`). Runs a per-epoch val loop
  computing per-category IoU; saves `best.pt` on improvement.

## Annotation (Label Studio)

Live project: **http://sentosa.xray.aps.anl.gov:8080/projects/1/data?tab=1**
(intranet only; requires ANL network / VPN).

All four categories are configured in the same LS project. GCP hole/sample
annotations are complete for round 1 (526 images, 4975 instances). Capillary
slit + tube annotation is in progress — Sam and Ming co-annotate; SAM3
bootstrap output for the 56 capillary images is loaded as pre-annotations to
review-and-correct rather than draw from scratch.

Regenerate the canonical COCO from the current LS state whenever a training
run is about to start (see the Round 1 quick-start below).

## Round 1: GCP decoder fine-tune

Round 1 validates the full pipeline end-to-end on the GCP hole/sample
annotations before touching capillary. Ming approved the recipe: freeze both
encoders, train the mask decoder, standard losses. Runs on sentosa's H200
idx 1 (`CUDA_VISIBLE_DEVICES=1`, ~26 GB free per
`~/.claude/projects/-Users-haskels/memory/reference_sentosa.md`).

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
    --coco       ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --base-checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out-dir    ~/aps12id_seg_finetune/runs/gcp_r1_baseline \
    --eval-only

# 3. Fine-tune (50 epochs, ~100 min on H200)
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.finetune.train \
    --coco       ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --base-checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out-dir    ~/aps12id_seg_finetune/runs/gcp_r1 \
    --epochs 50

# Watch per-epoch val IoU
tail -f ~/aps12id_seg_finetune/runs/gcp_r1/metrics.jsonl
```

Fine-tuned `best.pt` / `last.pt` are not committed (too large) — they stay
under `~/aps12id_seg_finetune/runs/gcp_r1/` on sentosa; the branch push
message links to them with the val numbers so Ming can pull them separately.

### Round 1 results (2026-07-16) — DONE

Split: 526 images (473 train / 53 val, 4975 annotations). Ran 50 epochs on
sentosa H200 idx 1 at bs=1, bf16 autocast, ~130 s/epoch (~108 min total).
Trained 32.7M / 840.5M params (3.9%).

| checkpoint          | hole IoU | sample IoU | val_loss |
|---------------------|----------|------------|----------|
| baseline (vanilla)  | 0.583    | 0.000      | 295.12   |
| fine-tuned best.pt  | 0.926    | 0.536      | 150.15 (train) / 161.58 (rerun) |
| fine-tuned last.pt  | 0.923    | **0.606**  | 181.83   |

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

### Known gap: metric ≠ Ming's real inference workflow

The 0.606 sample IoU above is measured with **text-only** prompts (the
trainer's find-mode: text "sample" → Hungarian-match to all instances). In
production Ming uses a **two-call** workflow: text "hole" to find holes,
then per hole a text "sample" + click at the hole center to segment the
sample inside it. The click branch was never re-trained (frozen encoders
preserve the vanilla point-feature machinery, but the decoder's response to
clicks on our sample geometry is unmeasured).

Before deploying, re-run the val split through Ming's actual 2-call workflow
and record sample IoU with the click — that's the number that matters for
his workflow. If it's significantly worse than 0.606, round 2 should add
interactive-mode training examples (image, text="sample", click_at_GT_center
→ one mask) alongside the current text-only training.

## Round 2: joint GCP + capillary — WAITING ON ANNOTATIONS

Round 2 is a single joint model over all four categories, trained on
merged GCP + capillary COCOs. It re-fine-tunes from the SAM3 **base**
checkpoint (not round 1's `best.pt`) to avoid GCP-specialization bias when
learning capillary geometry. Kick off once Sam + Ming finish the slit +
capillary tube annotations in LS.

Trainer edits needed at that point:
- Expand `GCP_CATEGORIES = ("hole", "sample")` in `train.py` to include
  `slit` and `capillary tube`.
- Load the merged COCO via `merge_cocos()` in `coco_schema.py`.
- If the known-gap eval (above) shows click-based sample detection is bad,
  add per-instance interactive-mode training examples.

## Operator quick-start

Assumes you're on sentosa with `~/aps12id_seg_finetune/` populated and the
repo's uv env synced.

```bash
cd ~/aps12id_seg_finetune/repo
uv sync

# 1. Migrate the current LS work to COCO
uv run python -m scripts.finetune.ls_to_coco \
    --db ~/aps12id_seg_finetune/label_studio/data/label_studio.sqlite3 \
    --project-id 1 \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --out ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json

# 2. (Optional, for capillary pre-annotations) bootstrap with vanilla SAM3
uv run python -m scripts.finetune.sam3_bootstrap \
    --plate-type capillary \
    --image-root ~/aps12id_seg_finetune/data/raw/capillary \
    --checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out ~/aps12id_seg_finetune/data/labels/canonical/capillary.coco.json \
    --summary ~/aps12id_seg_finetune/data/labels/canonical/capillary.summary.jsonl

# 3. Train — see the Round 1 block above for the full invocation.
```

## What Ming reviews

Ming reviews annotations in the shared LS project at
`http://sentosa.xray.aps.anl.gov:8080/projects/1/data?tab=1` and reads the
round 1 numbers above (or the raw `metrics.jsonl` on sentosa). Checkpoints
stay on sentosa; the committed `data/labels/canonical/*.coco.json` files
are the current dataset snapshot.
