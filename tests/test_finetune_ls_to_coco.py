import json
from pathlib import Path

from scripts.finetune.ls_to_coco import parse_ls_image_ref, resolve_ls_image_path


def _task_data(image: str) -> str:
    return json.dumps({"image": image})


def test_parse_ls_image_ref_supports_local_storage_and_uploads() -> None:
    assert (
        parse_ls_image_ref(_task_data("/data/local-files/?d=Camera/image%201.jpg"))
        == "Camera/image 1.jpg"
    )
    assert (
        parse_ls_image_ref(_task_data("/data/upload/1/abc-image.jpg"))
        == "abc-image.jpg"
    )


def test_resolve_ls_image_path_falls_back_to_moved_flat_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.touch()

    resolved = resolve_ls_image_path(
        _task_data("/data/local-files/?d=Camera/image.jpg"),
        tmp_path,
    )

    assert resolved == image_path
