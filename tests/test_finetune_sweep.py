from pathlib import Path

import pytest

from scripts.finetune.sweep import build_run_configs, parse_args


def test_build_run_configs_uses_cartesian_product_and_named_out_dirs() -> None:
    training_config = {
        "coco_json": "coco.json",
        "image_root": "images",
        "base_checkpoint": "sam3.pt",
        "out_dir": "runs/template",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "epochs": 20,
    }
    sweep_config = {
        "lr": [1e-5, 1e-4, 1e-3],
        "weight_decay": [1e-3, 1e-2, 0],
    }

    runs = build_run_configs(training_config, sweep_config)

    assert len(runs) == 9
    assert runs[0]["lr"] == 1e-5
    assert runs[0]["weight_decay"] == 1e-3
    assert runs[0]["out_dir"] == "runs/sweep_lr_1e-05_weightDecay_0.001"
    assert runs[-1]["lr"] == 1e-3
    assert runs[-1]["weight_decay"] == 0
    assert runs[-1]["out_dir"] == "runs/sweep_lr_0.001_weightDecay_0"
    assert all(run["epochs"] == 20 for run in runs)


def test_build_run_configs_supports_kebab_case_training_fields() -> None:
    runs = build_run_configs(
        {"out-dir": "runs/template", "batch-size": 1},
        {"batch_size": [2, 4]},
    )

    assert [run["batch-size"] for run in runs] == [2, 4]
    assert [run["out-dir"] for run in runs] == [
        "runs/sweep_batchSize_2",
        "runs/sweep_batchSize_4",
    ]


@pytest.mark.parametrize("values", [1e-4, [], "not-a-list"])
def test_build_run_configs_rejects_invalid_sweep_values(values: object) -> None:
    with pytest.raises(ValueError, match="sweep field lr must contain"):
        build_run_configs({"out_dir": "runs/template", "lr": 1e-4}, {"lr": values})


def test_build_run_configs_rejects_fields_absent_from_template() -> None:
    with pytest.raises(ValueError, match="not present.*weight_decay"):
        build_run_configs({"out_dir": "runs/template", "lr": 1e-4}, {"weight_decay": [0.0]})


def test_parse_args_accepts_requested_interface() -> None:
    args = parse_args(
        [
            "--training-config",
            "train/config.yaml",
            "--sweep-config",
            "sweep/config.yaml",
        ]
    )

    assert args.training_config == Path("train/config.yaml")
    assert args.sweep_config == Path("sweep/config.yaml")
