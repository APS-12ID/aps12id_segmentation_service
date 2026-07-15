"""Vanilla SAM3 pseudo-labels for the aps12id finetune, emitted as COCO.

Two modes selected by --plate-type:

  gcp:
    HOLE  text="hole" @ 0.15 permissive, NMS-deduped, keep top-K by
          orientation (samH=3, samV=15), filter to score >= HOLE_KEEP_TH.
    SAMPLE per-kept-hole click at the hole centroid @ 0.35. Skip if the
          sample mask is essentially the hole itself or nearly fills it.
    Only samH/samV are processed; samX/samY skipped per Ming.

  capillary:
    SLIT and CAPILLARY_TUBE text prompts, permissive threshold, NMS across
    all detections. No per-object click pass — Ming: text-only global.
    All four orientations are processed; the vanilla model is expected to
    do poorly here (that's the whole point of the finetune).

Output is a COCO instances JSON. Confidence scores are stored in the
metadata sidecar on each annotation, not in the main mask fields.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# The sam3 submodule at repo/sam3/ shadows the venv-installed sam3 package
# (an editable install of repo/sam3/sam3/) when this script is invoked as
# `python -m scripts.finetune.sam3_bootstrap` from repo root. Drop the
# shadow so pkg_resources can locate sam3's bundled assets.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) in sys.path:
    sys.path = [p for p in sys.path if Path(p).resolve() != _repo_root]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

from .coco_schema import (
    CATEGORY_ID_BY_NAME,
    annotation_id_from,
    empty_coco,
    image_id_from_path,
    make_annotation,
    make_image,
    validate,
    write_coco,
)

# --- constants (tune here; no YAML) --------------------------------------

HOLE_THRESHOLD = 0.15        # permissive text-prompt cutoff (SAM3 raw)
HOLE_KEEP_THRESHOLD = 0.3    # min score to keep a hole after ranking
SAMPLE_THRESHOLD = 0.35
SAMPLE_MAX_HOLE_IOU = 0.7    # drop sample that just re-detected the hole
SAMPLE_MAX_AREA_RATIO = 0.85 # sample should be smaller than its hole
NMS_IOU = 0.5
POLY_MIN_AREA_PX = 200

HOLE_TOP_K = {"H": 3, "V": 15}

# Capillary prompts. Threshold is permissive on purpose — recall over
# precision; a human deletes false positives faster than they draw missed
# ones.
CAPILLARY_PROMPTS: dict[str, float] = {
    "slit": 0.15,
    "capillary tube": 0.15,
}
CAPILLARY_KEEP_THRESHOLD = 0.25
CAPILLARY_TOP_K = 30  # generous upper bound; NMS is the real filter

ORIENT_RE = re.compile(r"_sam([HVXY])_")
GCP_ORIENTATIONS = {"H", "V"}
CAPILLARY_ORIENTATIONS = {"H", "V", "X", "Y"}


# --- SAM3 wiring (kept identical to the working prototype) ---------------

def parse_orientation(path: Path) -> str | None:
    m = ORIENT_RE.search(path.stem)
    return m.group(1) if m else None


def load_processor(checkpoint: Path, device: str, threshold: float) -> Sam3Processor:
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint), device=device, load_from_HF=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=threshold)


def prime_visual_text(processor: Sam3Processor, state: dict) -> None:
    text_outputs = processor.model.backbone.forward_text(
        ["visual"], device=processor.device
    )
    state["backbone_out"].update(text_outputs)


def click_once(processor: Sam3Processor, state: dict, x: float, y: float) -> dict:
    state["geometric_prompt"] = processor.model._get_dummy_prompt()
    width, height = state["original_width"], state["original_height"]
    point = torch.tensor(
        [[[x / width, y / height]]], device=processor.device, dtype=torch.float32
    )
    label = torch.tensor([[1]], device=processor.device, dtype=torch.long)
    state["geometric_prompt"].append_points(point, label)
    return processor._forward_grounding(state)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def nms_by_mask(
    masks: list[np.ndarray],
    scores: np.ndarray,
    iou_thresh: float,
    keep_score: float,
    top_k: int,
) -> list[int]:
    order = list(np.argsort(-scores))
    kept: list[int] = []
    for i in order:
        if scores[i] < keep_score:
            continue
        if any(mask_iou(masks[i], masks[k]) > iou_thresh for k in kept):
            continue
        kept.append(int(i))
        if len(kept) >= top_k:
            break
    return kept


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# --- per-plate segmentation ---------------------------------------------

def _sanitize_mask(mask: np.ndarray) -> np.ndarray:
    """Drop tiny contour noise; return uint8 mask."""
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if mask_u8.sum() < POLY_MIN_AREA_PX:
        return np.zeros_like(mask_u8)
    return mask_u8


def segment_gcp(
    processor: Sam3Processor, image: Image.Image, orientation: str,
) -> list[tuple[int, np.ndarray, float]]:
    """Return list of (category_id, mask, score) for one GCP image."""
    top_k = HOLE_TOP_K.get(orientation, 3)
    out: list[tuple[int, np.ndarray, float]] = []

    processor.confidence_threshold = HOLE_THRESHOLD
    state_h = processor.set_image(image)
    hole_out = processor.set_text_prompt(state=state_h, prompt="hole")
    h_masks = [m.squeeze().astype(bool)
               for m in hole_out["masks"].detach().cpu().numpy()]
    h_scores = hole_out["scores"].detach().float().cpu().numpy()
    kept_hole_ix = nms_by_mask(h_masks, h_scores, NMS_IOU,
                               HOLE_KEEP_THRESHOLD, top_k)
    kept_hole_masks: list[np.ndarray] = []
    for i in kept_hole_ix:
        m = _sanitize_mask(h_masks[i])
        if m.sum() == 0:
            continue
        kept_hole_masks.append(m)
        out.append((CATEGORY_ID_BY_NAME["hole"], m, float(h_scores[i])))
    if not kept_hole_masks:
        return out

    processor.confidence_threshold = SAMPLE_THRESHOLD
    state_s = processor.set_image(image)
    prime_visual_text(processor, state_s)

    per_click: list[tuple[np.ndarray, float]] = []
    for hole_mask in kept_hole_masks:
        c = mask_centroid(hole_mask)
        if c is None:
            continue
        s_out = click_once(processor, state_s, c[0], c[1])
        s_masks = s_out["masks"].detach().cpu().numpy()
        s_scores = s_out["scores"].detach().float().cpu().numpy()
        if len(s_masks) == 0:
            continue
        top = int(np.argmax(s_scores))
        if float(s_scores[top]) < SAMPLE_THRESHOLD:
            continue
        sample_mask = s_masks[top].squeeze().astype(bool)
        if mask_iou(sample_mask, hole_mask) > SAMPLE_MAX_HOLE_IOU:
            continue
        hole_area = int(hole_mask.sum())
        if hole_area > 0 and sample_mask.sum() / hole_area > SAMPLE_MAX_AREA_RATIO:
            continue
        per_click.append((_sanitize_mask(sample_mask), float(s_scores[top])))

    if per_click:
        s_masks_only = [t[0] for t in per_click]
        s_scores_only = np.array([t[1] for t in per_click])
        keep = nms_by_mask(s_masks_only, s_scores_only, NMS_IOU,
                           SAMPLE_THRESHOLD, top_k)
        for i in keep:
            m, sc = per_click[i]
            if m.sum() == 0:
                continue
            out.append((CATEGORY_ID_BY_NAME["sample"], m, sc))
    return out


def segment_capillary(
    processor: Sam3Processor, image: Image.Image, orientation: str,
) -> list[tuple[int, np.ndarray, float]]:
    """Return list of (category_id, mask, score) for one capillary image.

    Text-only per Ming: no coordinate prompt. Emits both slit and
    capillary_tube candidates; the human triages in CVAT.
    """
    out: list[tuple[int, np.ndarray, float]] = []
    for prompt, threshold in CAPILLARY_PROMPTS.items():
        processor.confidence_threshold = threshold
        state = processor.set_image(image)
        det = processor.set_text_prompt(state=state, prompt=prompt)
        masks = [m.squeeze().astype(bool)
                 for m in det["masks"].detach().cpu().numpy()]
        scores = det["scores"].detach().float().cpu().numpy()
        keep = nms_by_mask(masks, scores, NMS_IOU,
                           CAPILLARY_KEEP_THRESHOLD, CAPILLARY_TOP_K)
        cat_id = CATEGORY_ID_BY_NAME[prompt]
        for i in keep:
            m = _sanitize_mask(masks[i])
            if m.sum() == 0:
                continue
            out.append((cat_id, m, float(scores[i])))
    return out


# --- driver --------------------------------------------------------------

def iter_images(root: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    seen: set[Path] = set()
    for ext in exts:
        for p in root.rglob(ext):
            seen.add(p)
    return sorted(seen)


def build_coco(
    plate_type: str,
    image_root: Path,
    checkpoint: Path,
    device: str,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if plate_type == "gcp":
        allowed = GCP_ORIENTATIONS
        segment = segment_gcp
    elif plate_type == "capillary":
        allowed = CAPILLARY_ORIENTATIONS
        segment = segment_capillary
    else:
        raise ValueError(f"unknown plate_type: {plate_type}")

    images = iter_images(image_root)
    tagged: list[tuple[Path, str]] = []
    for p in images:
        o = parse_orientation(p)
        if o is None:
            continue
        if o not in allowed:
            continue
        tagged.append((p, o))
    if limit:
        tagged = tagged[:limit]

    processor = load_processor(checkpoint, device, HOLE_THRESHOLD)
    coco = empty_coco()
    records: list[dict[str, Any]] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        for image_path, orientation in tqdm(tagged, ncols=100, desc="segment"):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                records.append({
                    "image": str(image_path.relative_to(image_root)),
                    "error": str(exc),
                })
                continue
            image_id = image_id_from_path(image_path, image_root)
            coco["images"].append(
                make_image(image_id, image_path, image_root, image.width, image.height)
            )
            try:
                dets = segment(processor, image, orientation)
            except Exception as exc:
                records.append({
                    "image": str(image_path.relative_to(image_root)),
                    "error": str(exc),
                })
                continue
            per_cat_counter: dict[int, int] = {}
            for cat_id, mask, score in dets:
                idx = per_cat_counter.get(cat_id, 0)
                per_cat_counter[cat_id] = idx + 1
                ann_id = annotation_id_from(image_id, cat_id, idx)
                coco["annotations"].append(
                    make_annotation(
                        ann_id=ann_id, image_id=image_id, category_id=cat_id,
                        mask=mask, score=score, source=f"sam3-{plate_type}",
                    )
                )
            records.append({
                "image": str(image_path.relative_to(image_root)),
                "orientation": orientation,
                "n_annotations": len(dets),
                "categories": {int(k): v for k, v in per_cat_counter.items()},
            })
    problems = validate(coco)
    if problems:
        raise RuntimeError(f"COCO validation failed: {problems[:5]}")
    return coco, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate-type", choices=["gcp", "capillary"], required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output COCO json path")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Optional per-image jsonl summary")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    coco, records = build_coco(
        plate_type=args.plate_type, image_root=args.image_root,
        checkpoint=args.checkpoint, device=args.device, limit=args.limit,
    )
    write_coco(coco, args.out)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    n_img = len(coco["images"])
    n_ann = len(coco["annotations"])
    print(f"wrote {args.out} — {n_img} images, {n_ann} annotations")


if __name__ == "__main__":
    main()
