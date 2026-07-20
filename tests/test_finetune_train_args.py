import logging
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.finetune.coco_schema import CATEGORIES
from scripts.finetune.train import (
    COCO_CATEGORY_NAMES,
    TrainConfig,
    _log_mlflow_losses,
    _restore_checkpoint,
    _save_checkpoint,
    _mlflow_run,
    parse_args,
)


def test_evaluation_categories_come_from_coco_schema() -> None:
    assert COCO_CATEGORY_NAMES == tuple(category["name"] for category in CATEGORIES)


def test_config_supplies_required_and_optional_arguments(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        """
coco_json: config.json
image_root: images
base-checkpoint: sam3.pt
out_dir: output
lr: 0.002
batch_size: 4
eval_only: true
"""
    )

    args = parse_args(["--config", str(config)])

    assert args.coco_json == Path("config.json")
    assert args.image_root == Path("images")
    assert args.base_checkpoint == Path("sam3.pt")
    assert args.out_dir == Path("output")
    assert args.lr == 0.002
    assert args.batch_size == 4
    assert args.eval_only is True
    assert isinstance(args, TrainConfig)


def test_explicit_arguments_override_config(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        """
coco_json: from-config.json
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
            "--coco-json",
            "from-cli.json",
            "--lr",
            "0.01",
            "--batch-size",
            "8",
        ]
    )

    assert args.coco_json == Path("from-cli.json")
    assert args.lr == 0.01
    assert args.batch_size == 8


def test_unknown_config_field_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("unknown_field: value\n")

    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])

    assert "unknown config field(s): unknown_field" in capsys.readouterr().err


def test_old_coco_argument_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--coco",
                "coco.json",
                "--image-root",
                "images",
                "--base-checkpoint",
                "sam3.pt",
                "--out-dir",
                "output",
            ]
        )

    assert "the following arguments are required: --coco-json" in capsys.readouterr().err


def test_mlflow_arguments() -> None:
    args = parse_args(
        [
            "--coco-json",
            "coco.json",
            "--image-root",
            "images",
            "--base-checkpoint",
            "sam3.pt",
            "--out-dir",
            "output",
            "--enable-mlflow",
            "--mlflow-base-uri",
            "http://mlflow:5000",
            "--mlflow-experiment-name",
            "finetune",
            "--mlflow-run-name",
            "test-run",
        ]
    )

    assert args.enable_mlflow is True
    assert args.mlflow_base_uri == "http://mlflow:5000"
    assert args.mlflow_experiment_name == "finetune"
    assert args.mlflow_run_name == "test-run"


def test_save_every_defaults_to_ten_and_accepts_positive_integer() -> None:
    required_args = [
        "--coco-json",
        "coco.json",
        "--image-root",
        "images",
        "--base-checkpoint",
        "sam3.pt",
        "--out-dir",
        "output",
    ]

    assert parse_args(required_args).save_every == 10
    assert parse_args([*required_args, "--save-every", "3"]).save_every == 3


def test_save_every_rejects_non_positive_integer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--coco-json",
                "coco.json",
                "--image-root",
                "images",
                "--base-checkpoint",
                "sam3.pt",
                "--out-dir",
                "output",
                "--save-every",
                "0",
            ]
        )

    assert "Input should be greater than 0" in capsys.readouterr().err


def test_mlflow_run_uses_timestamp_and_logs_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeRun:
        def __enter__(self):
            calls["entered"] = True

        def __exit__(self, exc_type, exc_value, traceback):
            calls["exit_type"] = exc_type

    def start_run(*, run_name):
        calls["run_name"] = run_name
        return FakeRun()

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.setdefault("tracking_uri", uri),
        set_experiment=lambda name: calls.setdefault("experiment", name),
        start_run=start_run,
        log_params=lambda params: calls.setdefault("params", params),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    args = SimpleNamespace(
        enable_mlflow=True,
        mlflow_base_uri="http://mlflow:5000",
        mlflow_experiment_name="finetune",
        mlflow_run_name=None,
        coco_json=Path("coco.json"),
    )

    with _mlflow_run(args):
        pass

    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["experiment"] == "finetune"
    assert re.fullmatch(r"\d{8}-\d{6}", calls["run_name"])
    assert calls["params"]["coco_json"] == "coco.json"
    assert calls["params"]["mlflow_run_name"] == calls["run_name"]
    assert calls["entered"] is True
    assert calls["exit_type"] is None


def test_mlflow_logs_training_and_validation_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    fake_mlflow = SimpleNamespace(
        log_metrics=lambda metrics, *, step: calls.update(metrics=metrics, step=step)
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    _log_mlflow_losses(True, train_loss=1.25, val_loss=0.75, epoch=3)

    assert calls == {
        "metrics": {"train_loss": 1.25, "val_loss": 0.75},
        "step": 3,
    }


def test_checkpoint_restores_optimizer_progress_and_rng_state(tmp_path: Path) -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    _save_checkpoint(checkpoint_path, model, optimizer, epoch=4, val_loss=1.5, best_val=1.25)
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    start_epoch, best_val = _restore_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        logging.getLogger("test"),
    )

    assert start_epoch == 5
    assert best_val == 1.25
    restored_optimizer_state = restored_optimizer.state_dict()["state"]
    original_optimizer_state = optimizer.state_dict()["state"]
    assert restored_optimizer_state.keys() == original_optimizer_state.keys()
    for parameter_id, parameter_state in original_optimizer_state.items():
        for name, value in parameter_state.items():
            assert torch.equal(restored_optimizer_state[parameter_id][name], value)
    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
