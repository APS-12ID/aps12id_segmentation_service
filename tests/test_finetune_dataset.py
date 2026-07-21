import numpy as np
from pycocotools import mask as mask_utils

from scripts.finetune.dataset import _decode_segmentation, split_coco_by_image_id


def _coco(num_images: int = 20) -> dict:
    return {
        "categories": [{"id": 1, "name": "sample"}],
        "images": [{"id": image_id} for image_id in range(num_images)],
        "annotations": [
            {"id": image_id, "image_id": image_id, "category_id": 1}
            for image_id in range(num_images)
        ],
    }


def _image_ids(coco: dict) -> set[int]:
    return {image["id"] for image in coco["images"]}


def test_coco_split_is_randomized_and_reproducible_by_seed() -> None:
    coco = _coco()

    train_a, val_a = split_coco_by_image_id(coco, val_fraction=0.25, seed=7)
    train_b, val_b = split_coco_by_image_id(coco, val_fraction=0.25, seed=7)
    _, val_c = split_coco_by_image_id(coco, val_fraction=0.25, seed=8)

    assert _image_ids(train_a) == _image_ids(train_b)
    assert _image_ids(val_a) == _image_ids(val_b)
    assert _image_ids(val_a) != _image_ids(val_c)
    assert len(_image_ids(val_a)) == 5
    assert _image_ids(train_a).isdisjoint(_image_ids(val_a))
    assert _image_ids(train_a) | _image_ids(val_a) == set(range(20))


def test_decode_polygon_segmentation() -> None:
    mask = _decode_segmentation(
        [[1, 1, 4, 1, 4, 3, 1, 3]],
        height=5,
        width=6,
    )

    assert mask.shape == (5, 6)
    assert mask.sum() > 0


def test_decode_compressed_rle_segmentation() -> None:
    expected = np.zeros((5, 6), dtype=np.uint8)
    expected[1:4, 2:5] = 1
    segmentation = mask_utils.encode(np.asfortranarray(expected))
    segmentation["counts"] = segmentation["counts"].decode("ascii")

    mask = _decode_segmentation(segmentation, height=5, width=6)

    np.testing.assert_array_equal(mask, expected)


def test_decode_uncompressed_rle_segmentation() -> None:
    expected = np.zeros((5, 6), dtype=np.uint8)
    expected[1:4, 2:5] = 1
    flattened = expected.flatten(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for value in flattened:
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = int(value)
    counts.append(run_length)

    mask = _decode_segmentation(
        {"size": [5, 6], "counts": counts},
        height=5,
        width=6,
    )

    np.testing.assert_array_equal(mask, expected)
