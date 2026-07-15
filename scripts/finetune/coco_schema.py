"""Canonical COCO schema for the aps12id SAM3 finetune.

Every producer in this pipeline (LS export, SAM3 bootstrap, CVAT round-trip)
targets this schema so the downstream SAM3 finetune loader has one shape to
read. Masks are RLE-encoded (COCO compressed RLE) for pixel-faithful storage
of thin structures (slits, capillary tubes) that polygons would degrade.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

# Stable category IDs. Keep these frozen across runs — CVAT and downstream
# consumers key off IDs, not names, when merging COCOs.
CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "hole", "supercategory": "gcp"},
    {"id": 2, "name": "sample", "supercategory": "gcp"},
    {"id": 3, "name": "slit", "supercategory": "capillary"},
    {"id": 4, "name": "capillary tube", "supercategory": "capillary"},
]

CATEGORY_ID_BY_NAME: dict[str, int] = {c["name"]: c["id"] for c in CATEGORIES}


def image_id_from_path(image_path: Path, image_root: Path) -> int:
    """Deterministic 32-bit image id from repo-relative path.

    Stable across runs so the same image always gets the same id, which lets
    us merge partial COCOs (LS-migrated + SAM3-bootstrapped) without id
    collisions or churn.
    """
    rel = image_path.relative_to(image_root).as_posix()
    digest = hashlib.blake2b(rel.encode("utf-8"), digest_size=4).digest()
    # Top bit clear so ints stay JSON-safe / positive.
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def annotation_id_from(image_id: int, category_id: int, index: int) -> int:
    """Deterministic annotation id from (image, category, per-image index)."""
    key = f"{image_id}:{category_id}:{index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def polygon_pct_to_mask(
    points_pct: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a LS-style percent-coord polygon to a binary uint8 mask."""
    pts_px = [
        (float(x) * width / 100.0, float(y) * height / 100.0) for x, y in points_pct
    ]
    if len(pts_px) < 3:
        return np.zeros((height, width), dtype=np.uint8)
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(pts_px, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean/uint8 mask as COCO compressed RLE (JSON-safe).

    pycocotools returns counts as bytes; the COCO JSON spec wants a string.
    """
    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def rle_bbox_area(rle: dict[str, Any]) -> tuple[list[float], float]:
    """Return (bbox=[x,y,w,h], area) for an RLE segmentation."""
    # pycocotools wants bytes back; round-trip through encode is not needed.
    rle_bytes = {"size": rle["size"], "counts": rle["counts"].encode("ascii")
                 if isinstance(rle["counts"], str) else rle["counts"]}
    bbox = mask_utils.toBbox(rle_bytes).tolist()  # [x, y, w, h] floats
    area = float(mask_utils.area(rle_bytes))
    return [float(v) for v in bbox], area


def make_annotation(
    ann_id: int,
    image_id: int,
    category_id: int,
    mask: np.ndarray,
    score: float | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build one COCO annotation from a binary mask.

    `score` and `source` are stored in a `metadata` sidecar field on the
    annotation. Downstream training loaders should ignore unknown fields;
    the SAM3 loader we target does.
    """
    rle = mask_to_rle(mask)
    bbox, area = rle_bbox_area(rle)
    ann: dict[str, Any] = {
        "id": int(ann_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": rle,
        "bbox": bbox,
        "area": area,
        "iscrowd": 1,  # RLE segmentations use iscrowd=1 by COCO convention.
    }
    meta: dict[str, Any] = {}
    if score is not None:
        meta["score"] = float(score)
    if source is not None:
        meta["source"] = source
    if meta:
        ann["metadata"] = meta
    return ann


def make_image(
    image_id: int,
    image_path: Path,
    image_root: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "id": int(image_id),
        "file_name": image_path.relative_to(image_root).as_posix(),
        "height": int(height),
        "width": int(width),
    }


def empty_coco() -> dict[str, Any]:
    return {
        "info": {
            "description": "aps12id SAM3 finetune dataset",
            "version": "1",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": list(CATEGORIES),
    }


def write_coco(coco: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=2))


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def merge_cocos(cocos: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple COCOs. Image ids must be globally unique across inputs
    (they are, because image_id_from_path is deterministic on relative path).
    Annotation ids likewise.
    """
    out = empty_coco()
    seen_images: set[int] = set()
    seen_anns: set[int] = set()
    for c in cocos:
        for img in c.get("images", []):
            if img["id"] in seen_images:
                continue
            seen_images.add(img["id"])
            out["images"].append(img)
        for ann in c.get("annotations", []):
            if ann["id"] in seen_anns:
                continue
            seen_anns.add(ann["id"])
            out["annotations"].append(ann)
    return out


def validate(coco: dict[str, Any]) -> list[str]:
    """Cheap structural checks. Returns list of problems; empty = healthy."""
    problems: list[str] = []
    img_ids = {img["id"] for img in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    for ann in coco["annotations"]:
        if ann["image_id"] not in img_ids:
            problems.append(f"annotation {ann['id']} references missing image_id {ann['image_id']}")
        if ann["category_id"] not in cat_ids:
            problems.append(f"annotation {ann['id']} references missing category_id {ann['category_id']}")
        seg = ann.get("segmentation")
        if not (isinstance(seg, dict) and "counts" in seg and "size" in seg):
            problems.append(f"annotation {ann['id']} has non-RLE segmentation")
    return problems
