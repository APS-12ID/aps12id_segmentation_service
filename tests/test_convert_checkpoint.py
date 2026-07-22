from pathlib import Path

import pytest
import torch

from scripts.finetune.convert_checkpoint import convert_checkpoint


def test_convert_checkpoint_adds_detector_prefix_and_drops_training_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training.pt"
    destination = tmp_path / "serving.pt"
    model_state = {
        "backbone.weight": torch.tensor([1.0, 2.0]),
        "head.bias": torch.tensor([3.0]),
    }
    torch.save({"model": model_state, "optimizer": {"state": "unused"}}, source)

    tensor_count = convert_checkpoint(source, destination)

    converted = torch.load(destination, map_location="cpu", weights_only=True)
    assert tensor_count == 2
    assert converted.keys() == {"model"}
    assert converted["model"].keys() == {
        "detector.backbone.weight",
        "detector.head.bias",
    }
    torch.testing.assert_close(
        converted["model"]["detector.backbone.weight"], model_state["backbone.weight"]
    )
    torch.testing.assert_close(
        converted["model"]["detector.head.bias"], model_state["head.bias"]
    )


def test_convert_checkpoint_refuses_to_overwrite_destination(tmp_path: Path) -> None:
    source = tmp_path / "training.pt"
    destination = tmp_path / "serving.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, source)
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Destination already exists"):
        convert_checkpoint(source, destination)

    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({"optimizer": {}}, "containing a 'model' state dictionary"),
        ({"model": {}}, "model state dictionary is empty"),
        ({"model": {"detector.weight": torch.tensor([1.0])}}, "already uses"),
    ],
)
def test_convert_checkpoint_rejects_unsupported_input(
    tmp_path: Path, checkpoint: dict, message: str
) -> None:
    source = tmp_path / "source.pt"
    torch.save(checkpoint, source)

    with pytest.raises(ValueError, match=message):
        convert_checkpoint(source, tmp_path / "destination.pt")
