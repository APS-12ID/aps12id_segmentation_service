"""Run the finetune trainer for every combination in a parameter sweep."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.finetune.train import parse_args as parse_training_args


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    with path.open() as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"{description} must contain a mapping: {path}")
    if not all(isinstance(key, str) for key in config):
        raise ValueError(f"{description} fields must be strings: {path}")
    return config


def _canonical_field_name(field: str) -> str:
    return field.replace("-", "_")


def _camel_case(field: str) -> str:
    parts = _canonical_field_name(field).split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def _fields_by_canonical_name(config: Mapping[str, Any], description: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in config:
        canonical = _canonical_field_name(field)
        if canonical in fields:
            raise ValueError(
                f"{description} contains duplicate forms of field {canonical}: "
                f"{fields[canonical]}, {field}"
            )
        fields[canonical] = field
    return fields


def build_run_configs(
    training_config: Mapping[str, Any], sweep_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Expand a training template into one config per Cartesian-product point."""
    training_fields = _fields_by_canonical_name(training_config, "training config")
    sweep_fields = _fields_by_canonical_name(sweep_config, "sweep config")

    unknown = sorted(set(sweep_fields) - set(training_fields))
    if unknown:
        raise ValueError(
            "sweep field(s) are not present in the training config: " + ", ".join(unknown)
        )
    if "out_dir" not in training_fields:
        raise ValueError("training config must contain out_dir")

    parameter_values: list[list[Any]] = []
    for canonical, sweep_field in sweep_fields.items():
        values = sweep_config[sweep_field]
        if not isinstance(values, list):
            raise ValueError(f"sweep field {sweep_field} must contain a list")
        if not values:
            raise ValueError(f"sweep field {sweep_field} must contain at least one value")
        parameter_values.append(values)

    out_dir_field = training_fields["out_dir"]
    out_dir_value = training_config[out_dir_field]
    if not isinstance(out_dir_value, str):
        raise ValueError("training config field out_dir must be a string")
    template_out_dir = Path(out_dir_value)

    runs = []
    for combination in itertools.product(*parameter_values):
        run_config = dict(training_config)
        name_parts = ["sweep"]
        for (canonical, _), value in zip(sweep_fields.items(), combination, strict=True):
            training_field = training_fields[canonical]
            run_config[training_field] = value
            name_parts.extend([_camel_case(canonical), _format_parameter_value(value)])
        run_config[out_dir_field] = str(template_out_dir.with_name("_".join(name_parts)))
        runs.append(run_config)
    return runs


def run_sweep(training_config_path: Path, sweep_config_path: Path) -> None:
    training_config = _load_mapping(training_config_path, "training config")
    sweep_config = _load_mapping(sweep_config_path, "sweep config")
    run_configs = build_run_configs(training_config, sweep_config)

    with tempfile.TemporaryDirectory(prefix="finetune-sweep-") as temp_dir:
        config_paths = []
        for index, run_config in enumerate(run_configs):
            config_path = Path(temp_dir) / f"run_{index:04d}.yaml"
            config_path.write_text(yaml.safe_dump(run_config, sort_keys=False))
            config_paths.append(config_path)

        # Validate the complete sweep before starting its first potentially long run.
        for config_path in config_paths:
            parse_training_args(["--config", str(config_path)])

        for config_path in config_paths:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.finetune.train",
                    "--config",
                    str(config_path),
                ],
                check=True,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        run_sweep(args.training_config, args.sweep_config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
