# Finetune pipeline

Semi-supervised SAM3 finetune workflow for aps12id sample-plate imagery.
Everything downstream (training, evaluation, deployment) reads a single
canonical dataset shape: **COCO instance segmentation with compressed-RLE
masks**. Every producer here emits that shape.

Two plate types, four categories, one dataset:

| id | name          | supercategory | source                        |
|----|---------------|---------------|-------------------------------|
| 1  | hole          | gcp           | Label Studio migration        |
| 2  | sample        | gcp           | Label Studio migration        |
| 3  | slit          | capillary     | SAM3 pseudo-labels + human    |
| 4  | capillary tube| capillary     | SAM3 pseudo-labels + human    |

## Modules

- `coco_schema.py` — category IDs, deterministic image/annotation IDs, and
  polygon↔RLE helpers. Frozen schema so partial COCOs can be merged without
  ID churn.
- `ls_to_coco.py` — one-shot migration of Sam's Label Studio work. Reads
  the LS sqlite directly, rasterizes percent-coord polygons to masks,
  emits RLE-encoded COCO. Preserves both hand-drawn labels and SAM3
  pre-annotations that Sam edited.
- `sam3_bootstrap.py` — vanilla SAM3 pseudo-labels for either plate type.
  `--plate-type gcp` reproduces the existing samH/V hole→sample pipeline.
  `--plate-type capillary` uses text-only prompts for slit + capillary tube.
- `cvat_import.py` — creates a CVAT project + one task per plate type and
  uploads images + COCO annotations. Idempotent by name; `--replace` to
  recreate.

## Operator quick-start

Assumes you're on sentosa with `~/aps12id_seg_finetune/` populated and the
repo's uv env synced.

```bash
cd ~/aps12id_seg_finetune/repo
uv sync

# 1. Migrate Sam's LS work to COCO
uv run python -m scripts.finetune.ls_to_coco \
    --db ~/aps12id_seg_finetune/label_studio/data/label_studio.sqlite3 \
    --project-id 1 \
    --image-root ~/aps12id_seg_finetune/data/raw \
    --out ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json

# 2. Bootstrap capillary pseudo-labels
uv run python -m scripts.finetune.sam3_bootstrap \
    --plate-type capillary \
    --image-root ~/aps12id_seg_finetune/data/raw/capillary \
    --checkpoint ~/aps12id_seg_finetune/checkpoints/sam3.pt \
    --out ~/aps12id_seg_finetune/data/labels/canonical/capillary.coco.json \
    --summary ~/aps12id_seg_finetune/data/labels/canonical/capillary.summary.jsonl

# 3. Import both into CVAT (requires CVAT running — see below)
CVAT_PASSWORD='...' uv run python -m scripts.finetune.cvat_import \
    --gcp-coco ~/aps12id_seg_finetune/data/labels/canonical/gcp.coco.json \
    --gcp-image-root ~/aps12id_seg_finetune/data/raw \
    --capillary-coco ~/aps12id_seg_finetune/data/labels/canonical/capillary.coco.json \
    --capillary-image-root ~/aps12id_seg_finetune/data/raw/capillary
```

## CVAT deployment (sentosa) — BLOCKED on subuid delegation

**Status as of 2026-07-15:** CVAT images pulled, compose config staged
(ports remapped to 8180/8190 so LS keeps 8080), but the stack fails to
start under sentosa's rootless podman because `/etc/subuid` and
`/etc/subgid` have no delegated range for `haskels`. Symptoms:

- Image extraction: fails with `chown: Operation not permitted` on files
  owned by container-side uid 42/999/etc. Worked around in
  `~/.config/containers/storage.conf` via
  `[storage.options.overlay] ignore_chown_errors = "true"`.
- Runtime: `cvat_db` (postgres) crashes repeatedly with
  `chown: /var/lib/postgresql/data: Invalid argument` at entrypoint. This
  cannot be worked around from user-space — the postgres entrypoint runs
  its own `chown` regardless of pre-owned volume state. Every other
  service that switches uid at startup will hit the same wall.

### To unblock

Ask AES to run (on sentosa, as root):

```
usermod --add-subuids 100000-165535 --add-subgids 100000-165535 haskels
# then, from Sam's shell:
podman system migrate
```

Once subuid is delegated the stack should come up cleanly:

```bash
cd ~/cvat
podman-compose up -d
# wait for cvat_server healthy
podman-compose exec cvat_server python manage.py createsuperuser
# username: haskels, set a password, use it below

# Tunnel from your laptop:
#   ssh -N -L 8180:localhost:8180 haskels@sentosa.xray.aps.anl.gov
# then browse http://localhost:8180

CVAT_PASSWORD='...' uv run python -m scripts.finetune.cvat_import \
    --gcp-coco  data/labels/canonical/gcp.coco.json  --gcp-image-root  ...raw \
    --capillary-coco data/labels/canonical/capillary.coco.json \
    --capillary-image-root ...raw/capillary
```

### Meanwhile

- Label Studio at `http://localhost:8080` (via `ssh -N -L 8080:localhost:8080 …`)
  keeps the GCP labeling workflow live. Sam's in-progress annotations are
  preserved in the LS sqlite.
- `data/labels/canonical/gcp.coco.json` is a valid snapshot of that work;
  regenerate anytime with `ls_to_coco`.
- `data/labels/canonical/capillary.coco.json` is the vanilla-SAM3
  first-pass on the 56 capillary images (378 slit proposals, 0 capillary
  tube — Ming's "more manual labeling" warning bears out). Ready for CVAT
  the moment CVAT is up.

## What Ming reviews

Ming works from `data/labels/canonical/*.coco.json` in this branch or via
the CVAT UI. The scripts are the interchange; the checked-in COCO files
are the current dataset snapshot.
