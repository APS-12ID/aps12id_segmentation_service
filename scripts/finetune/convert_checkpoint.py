from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def convert_checkpoint(source: Path, destination: Path) -> int:
    """Convert a fine-tuning checkpoint to the format expected by SAM3 serving."""
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different files.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError(
            "Expected a training checkpoint containing a 'model' state dictionary."
        )

    model_state = checkpoint["model"]
    if not model_state:
        raise ValueError("The checkpoint's model state dictionary is empty.")
    if any(key.startswith(("detector.", "tracker.")) for key in model_state):
        raise ValueError("The checkpoint already uses SAM3 serving key prefixes.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    converted_state = {f"detector.{key}": value for key, value in model_state.items()}

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save({"model": converted_state}, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return len(converted_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a fine-tuned SAM3 checkpoint for use by the segmentation server."
        )
    )
    parser.add_argument("source", type=Path, help="Fine-tuning checkpoint to convert.")
    parser.add_argument("destination", type=Path, help="New serving checkpoint to create.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_count = convert_checkpoint(args.source, args.destination)
    print(f"Created {args.destination} with {tensor_count} model tensors.")


if __name__ == "__main__":
    main()
