"""Create a CVAT project + one task per plate type, upload images, import COCO.

Idempotent by project/task name: re-running with the same names will reuse
the existing project and refuse to create a duplicate task unless --replace
is passed (which deletes and recreates). Errs on the side of NOT clobbering
work — CVAT tasks are the source of truth once Sam starts editing.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from cvat_sdk import make_client
from cvat_sdk.api_client import exceptions
from cvat_sdk.core.proxies.tasks import ResourceType

from .coco_schema import CATEGORIES, load_coco


def _ensure_project(client, name: str) -> Any:
    existing = client.projects.list()
    for p in existing:
        if p.name == name:
            return p
    labels = [{"name": c["name"]} for c in CATEGORIES]
    return client.projects.create(spec={"name": name, "labels": labels})


def _find_task(project, name: str):
    for t in project.get_tasks():
        if t.name == name:
            return t
    return None


def _build_coco_zip(coco_path: Path, image_root: Path, tmpdir: Path) -> Path:
    """CVAT's COCO 1.0 import wants a zip with annotations/instances_default.json."""
    zip_path = tmpdir / "coco.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_path, arcname="annotations/instances_default.json")
    return zip_path


def upload_plate(
    client,
    project,
    task_name: str,
    coco_path: Path,
    image_root: Path,
    replace: bool,
) -> str:
    coco = load_coco(coco_path)
    image_paths: list[str] = []
    for img in coco["images"]:
        p = image_root / img["file_name"]
        if not p.exists():
            raise FileNotFoundError(f"COCO references missing image {p}")
        image_paths.append(str(p))
    if not image_paths:
        raise RuntimeError(f"COCO at {coco_path} has 0 images")

    existing = _find_task(project, task_name)
    if existing is not None:
        if replace:
            existing.remove()
        else:
            return f"task {task_name!r} already exists (id={existing.id}); pass --replace to recreate"

    task = project.create_task(
        spec={"name": task_name},
        resource_type=ResourceType.LOCAL,
        resources=image_paths,
    )
    with tempfile.TemporaryDirectory() as td:
        zip_path = _build_coco_zip(coco_path, image_root, Path(td))
        task.import_annotations(format_name="COCO 1.0", filename=str(zip_path))
    return f"task {task_name!r} created (id={task.id}); {len(image_paths)} images, "\
           f"{len(coco['annotations'])} annotations"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8180",
                    help="CVAT base URL (default assumes local tunnel)")
    ap.add_argument("--user", default=os.environ.get("CVAT_USER", "haskels"))
    ap.add_argument("--password", default=os.environ.get("CVAT_PASSWORD"),
                    help="or set CVAT_PASSWORD env var")
    ap.add_argument("--project-name", default="aps12id-seg")
    ap.add_argument("--gcp-coco", type=Path,
                    help="COCO for GCP plates (e.g. from ls_to_coco)")
    ap.add_argument("--gcp-image-root", type=Path)
    ap.add_argument("--capillary-coco", type=Path,
                    help="COCO for capillary plates (from sam3_bootstrap)")
    ap.add_argument("--capillary-image-root", type=Path)
    ap.add_argument("--replace", action="store_true",
                    help="Delete existing tasks with the same name and recreate")
    args = ap.parse_args()

    if not args.password:
        print("error: --password or CVAT_PASSWORD required", file=sys.stderr)
        sys.exit(2)

    from urllib.parse import urlparse
    u = urlparse(args.host)
    host_kw = {"host": u.hostname, "port": u.port or (443 if u.scheme == "https" else 80)}
    with make_client(credentials=(args.user, args.password), **host_kw) as client:
        try:
            project = _ensure_project(client, args.project_name)
        except exceptions.ApiException as exc:
            print(f"failed to create/find project: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.gcp_coco:
            assert args.gcp_image_root, "--gcp-image-root required with --gcp-coco"
            print(upload_plate(client, project, "aps12id-gcp",
                               args.gcp_coco, args.gcp_image_root, args.replace))
        if args.capillary_coco:
            assert args.capillary_image_root, "--capillary-image-root required with --capillary-coco"
            print(upload_plate(client, project, "aps12id-capillary",
                               args.capillary_coco, args.capillary_image_root, args.replace))
        if not args.gcp_coco and not args.capillary_coco:
            print("nothing to do — pass --gcp-coco or --capillary-coco")


if __name__ == "__main__":
    main()
