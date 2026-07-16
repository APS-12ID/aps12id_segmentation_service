from pathlib import Path

import pytest

from scripts.finetune.train import parse_args


def test_config_supplies_required_and_optional_arguments(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        """
coco: config.json
image_root: images
base-checkpoint: sam3.pt
out_dir: output
lr: 0.002
eval_only: true
"""
    )

    args = parse_args(["--config", str(config)])

    assert args.coco == Path("config.json")
    assert args.image_root == Path("images")
    assert args.base_checkpoint == Path("sam3.pt")
    assert args.out_dir == Path("output")
    assert args.lr == 0.002
    assert args.eval_only is True


def test_explicit_arguments_override_config(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        """
coco: from-config.json
image_root: images
base_checkpoint: sam3.pt
out_dir: output
lr: 0.002
"""
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--coco",
            "from-cli.json",
            "--lr",
            "0.01",
        ]
    )

    assert args.coco == Path("from-cli.json")
    assert args.lr == 0.01


def test_unknown_config_field_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("unknown_field: value\n")

    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])

    assert "unknown config field(s): unknown_field" in capsys.readouterr().err
