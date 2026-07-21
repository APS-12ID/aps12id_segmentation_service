"""Migrate a Label Studio project's annotations into the canonical COCO.

Preserves work Sam did in the LS UI (both his manual labels and any edits
he made on top of the SAM3-bulk-imported polygons) by reading the LS sqlite
directly. Percent-coordinate polygons are rasterized to masks and stored as
COCO RLE — that's the shape the SAM3 finetune loader expects.

Handles all four categories (hole, sample, slit, capillary tube). Ming
co-annotates in the same LS project.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any

from PIL import Image

from .coco_schema import (
    LS_LABEL_TO_CATEGORY_ID,
    annotation_id_from,
    empty_coco,
    image_id_from_path,
    make_annotation,
    make_image,
    polygon_pct_to_mask,
    validate,
    write_coco,
)

DATA_URL_RE = re.compile(r"[?&]d=([^&]+)")
UPLOAD_URL_PREFIX = "/data/upload/"


def parse_ls_image_ref(task_data_json: str) -> str:
    """Return the image path stored in a Label Studio task.

    Local-storage tasks retain their path after ``?d=``. Uploaded tasks only
    expose Label Studio's internal ``/data/upload/<project>/`` path, so use the
    uploaded filename for those.
    """
    data = json.loads(task_data_json)
    ref = data.get("image", "")
    m = DATA_URL_RE.search(ref)
    if m:
        return urllib.parse.unquote(m.group(1))

    parsed_path = urllib.parse.unquote(urllib.parse.urlparse(ref).path)
    if parsed_path.startswith(UPLOAD_URL_PREFIX):
        return Path(parsed_path).name
    raise ValueError(f"cannot parse image ref: {ref!r}")


def resolve_ls_image_path(task_data_json: str, image_root: Path) -> Path:
    rel_path = Path(parse_ls_image_ref(task_data_json))
    image_path = image_root / rel_path
    if image_path.exists():
        return image_path

    # Dataset directories are sometimes moved without their former Label
    # Studio storage prefix (for example Camera/foo.jpg becomes images/foo.jpg).
    return image_root / rel_path.name


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.width, img.height


def migrate(
    db_path: Path,
    project_id: int,
    image_root: Path,
    out_path: Path,
) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT t.id AS task_id, t.data AS task_data,
               c.id AS ann_id, c.result AS ann_result,
               c.was_cancelled, c.bulk_created, c.updated_at, c.created_at,
               c.last_action
        FROM task t
        LEFT JOIN task_completion c ON c.task_id = t.id
        WHERE t.project_id = ?
        ORDER BY t.id, c.id
        """,
        (project_id,),
    )
    rows = cur.fetchall()
    conn.close()

    coco = empty_coco()
    added_images: set[int] = set()

    counts = {
        "tasks_seen": 0,
        "tasks_with_annotation": 0,
        "annotations_written": 0,
        "regions_kept": 0,
        "regions_dropped_empty": 0,
        "regions_dropped_bad_label": 0,
        "regions_dropped_cancelled": 0,
    }
    last_task_id: int | None = None
    for row in rows:
        if row["task_id"] != last_task_id:
            counts["tasks_seen"] += 1
            last_task_id = row["task_id"]
        if row["ann_id"] is None:
            continue
        if row["was_cancelled"]:
            counts["regions_dropped_cancelled"] += 1
            continue
        try:
            regions = json.loads(row["ann_result"] or "[]")
        except json.JSONDecodeError:
            continue
        if not regions:
            continue

        image_path = resolve_ls_image_path(row["task_data"], image_root)
        if not image_path.exists():
            counts["regions_dropped_empty"] += len(regions)
            continue

        w, h = load_image_size(image_path)
        image_id = image_id_from_path(image_path, image_root)
        if image_id not in added_images:
            coco["images"].append(make_image(image_id, image_path, image_root, w, h))
            added_images.add(image_id)

        wrote_any = False
        for idx, region in enumerate(regions):
            value = region.get("value", {})
            poly_labels = value.get("polygonlabels") or []
            rect_labels = value.get("rectanglelabels") or []
            if poly_labels:
                label = poly_labels[0]
                points = value.get("points") or []
            elif rect_labels:
                label = rect_labels[0]
                # LS rectangle geometry is (x, y, width, height) in percent
                # of image dims. Convert to a 4-corner polygon so the rest
                # of the pipeline treats it identically to polygon regions.
                try:
                    x = float(value["x"])
                    y = float(value["y"])
                    rw = float(value["width"])
                    rh = float(value["height"])
                except (KeyError, TypeError, ValueError):
                    counts["regions_dropped_empty"] += 1
                    continue
                points = [(x, y), (x + rw, y), (x + rw, y + rh), (x, y + rh)]
            else:
                counts["regions_dropped_bad_label"] += 1
                continue
            cat_id = LS_LABEL_TO_CATEGORY_ID.get(label)
            if cat_id is None:
                counts["regions_dropped_bad_label"] += 1
                continue
            if len(points) < 3:
                counts["regions_dropped_empty"] += 1
                continue
            mask = polygon_pct_to_mask(points, w, h)
            if mask.sum() == 0:
                counts["regions_dropped_empty"] += 1
                continue

            source = "ls-manual" if not row["bulk_created"] else "ls-sam3-touched"
            # If bulk_created and updated_at == created_at, the human never
            # touched this — mark it distinctly so downstream can dedupe
            # against a fresh SAM3 pass if desired.
            if row["bulk_created"] and row["updated_at"] == row["created_at"]:
                source = "ls-sam3-untouched"

            ann_id = annotation_id_from(image_id, cat_id, idx)
            coco["annotations"].append(
                make_annotation(
                    ann_id=ann_id,
                    image_id=image_id,
                    category_id=cat_id,
                    mask=mask,
                    score=region.get("score"),
                    source=source,
                )
            )
            counts["regions_kept"] += 1
            wrote_any = True
        if wrote_any:
            counts["annotations_written"] += 1
            counts["tasks_with_annotation"] += 1

    problems = validate(coco)
    if problems:
        raise RuntimeError(f"COCO validation failed: {problems[:5]}")

    write_coco(coco, out_path)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True,
                    help="Label Studio sqlite path (e.g. .../label_studio.sqlite3)")
    ap.add_argument("--project-id", type=int, default=1)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output COCO json path")
    args = ap.parse_args()
    counts = migrate(args.db, args.project_id, args.image_root, args.out)
    print(json.dumps(counts, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
