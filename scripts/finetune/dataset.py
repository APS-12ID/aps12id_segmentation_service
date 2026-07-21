"""Dataset loader for SAM3 fine-tune (round 1: GCP hole/sample).

Reads a canonical COCO file produced by :mod:`scripts.finetune.coco_schema` and
yields one sample per (image, category) pair that has at least one instance.
The trainer conditions on a text prompt (the category name) and asks the model
to predict masks for every instance of that category in the image, which is the
shape SAM3's grounded-detection forward expects.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class InstanceTarget:
    mask: np.ndarray
    bbox: list[float]


@dataclass(frozen=True)
class Sample:
    image_id: int
    image_path: Path
    image_width: int
    image_height: int
    category_id: int
    category_name: str
    instances: list[InstanceTarget]

    def load_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")


def _decode_rle(rle: dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    rle_bytes = {
        "size": rle["size"],
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }
    return np.asarray(mask_utils.decode(rle_bytes), dtype=np.uint8)


def _split_image_ids(
    image_ids: Sequence[int],
    val_fraction: float,
    split: str,
    seed: int = 0,
) -> set[int]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    ordered = sorted(image_ids)
    random.Random(seed).shuffle(ordered)
    n_val = max(1, int(round(len(ordered) * val_fraction))) if val_fraction > 0 else 0
    val_ids = set(ordered[len(ordered) - n_val:]) if n_val else set()
    if split == "val":
        return val_ids
    if split == "train":
        return set(ordered) - val_ids
    raise ValueError(f"split must be 'train' or 'val'; got {split!r}")


def split_coco_by_image_id(
    coco: dict[str, Any],
    val_fraction: float,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a canonical COCO into randomized (train_coco, val_coco) subsets.

    The random split is reproducible for a given ``seed``. Categories are
    duplicated verbatim into both outputs so downstream loaders see a stable
    category schema regardless of split.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    all_ids = [img["id"] for img in coco["images"]]
    val_ids = _split_image_ids(all_ids, val_fraction, "val", seed)
    train_ids = set(all_ids) - val_ids

    def _subset(image_ids: set[int]) -> dict[str, Any]:
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": list(coco["categories"]),
            "images": [img for img in coco["images"] if img["id"] in image_ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
        }

    return _subset(train_ids), _subset(val_ids)


class GcpCocoDataset:
    """One sample per (image, category) pair with at least one annotation.

    Iteration is stable: samples are ordered by (image_id, category_id) so a
    given (dataset, split) always visits samples in the same sequence, which
    keeps per-epoch metrics comparable across runs.
    """

    def __init__(
        self,
        coco_path: Path,
        image_root: Path,
        categories: Sequence[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> None:
        import json

        coco = json.loads(Path(coco_path).read_text())
        self.image_root = Path(image_root)

        cat_by_id: dict[int, dict[str, Any]] = {c["id"]: c for c in coco["categories"]}
        if categories is not None:
            allowed_ids = {c["id"] for c in coco["categories"] if c["name"] in set(categories)}
            missing = set(categories) - {c["name"] for c in coco["categories"] if c["id"] in allowed_ids}
            if missing:
                raise ValueError(f"categories not present in COCO: {sorted(missing)}")
        else:
            allowed_ids = set(cat_by_id)

        images_by_id: dict[int, dict[str, Any]] = {img["id"]: img for img in coco["images"]}
        split_image_ids = _split_image_ids(list(images_by_id), val_fraction, split, seed)

        # Group annotations by (image_id, category_id), keeping only annotations whose
        # image is in this split and whose category is allowed.
        groups: dict[tuple[int, int], list[InstanceTarget]] = {}
        for ann in coco["annotations"]:
            if ann["image_id"] not in split_image_ids:
                continue
            if ann["category_id"] not in allowed_ids:
                continue
            mask = _decode_rle(ann["segmentation"])
            if mask.sum() == 0:
                continue
            groups.setdefault((ann["image_id"], ann["category_id"]), []).append(
                InstanceTarget(mask=mask, bbox=[float(v) for v in ann["bbox"]])
            )

        samples: list[Sample] = []
        for (image_id, category_id), instances in sorted(groups.items()):
            image_meta = images_by_id[image_id]
            samples.append(
                Sample(
                    image_id=image_id,
                    image_path=self.image_root / image_meta["file_name"],
                    image_width=int(image_meta["width"]),
                    image_height=int(image_meta["height"]),
                    category_id=category_id,
                    category_name=cat_by_id[category_id]["name"],
                    instances=instances,
                )
            )
        self._samples = samples
        self._split = split
        self._val_fraction = val_fraction
        self._seed = seed
        self._categories = tuple(sorted(allowed_ids))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)

    @property
    def category_ids(self) -> tuple[int, ...]:
        return self._categories

    @property
    def image_ids(self) -> set[int]:
        return {s.image_id for s in self._samples}

    def summary(self) -> dict[str, Any]:
        per_cat: dict[str, dict[str, int]] = {}
        for s in self._samples:
            entry = per_cat.setdefault(s.category_name, {"images": 0, "instances": 0})
            entry["images"] += 1
            entry["instances"] += len(s.instances)
        return {
            "split": self._split,
            "val_fraction": self._val_fraction,
            "seed": self._seed,
            "num_samples": len(self._samples),
            "num_images": len(self.image_ids),
            "per_category": per_cat,
        }
