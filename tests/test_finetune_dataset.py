from scripts.finetune.dataset import split_coco_by_image_id


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
