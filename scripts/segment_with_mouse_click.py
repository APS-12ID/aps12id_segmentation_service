from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sam3_common import (
    build_processor,
    inference_context,
    load_image,
    save_segmentation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment an image with a positive SAM3 mouse-click point prompt."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--x", type=float, help="Click x coordinate in pixels.")
    parser.add_argument("--y", type=float, help="Click y coordinate in pixels.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/sam3_click_overlay.png"),
        help="Path for the overlay PNG.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional local SAM3 checkpoint path. If omitted, SAM3 downloads from Hugging Face.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 confidence score.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device.",
    )
    return parser.parse_args()


def get_mouse_click(image) -> tuple[float, float]:
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Click the object to segment")
    ax.set_axis_off()
    fig.tight_layout()

    clicks = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not clicks:
        raise RuntimeError("No click was selected.")
    return clicks[0]


def add_mouse_click_prompt(processor, state: dict, x: float, y: float) -> dict:
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before adding a click prompt.")

    width = state["original_width"]
    height = state["original_height"]
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(
            f"Click ({x}, {y}) is outside image bounds: width={width}, height={height}."
        )

    if "language_features" not in state["backbone_out"]:
        text_outputs = processor.model.backbone.forward_text(["visual"], device=processor.device)
        state["backbone_out"].update(text_outputs)

    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    point = torch.tensor(
        [[[x / width, y / height]]],
        device=processor.device,
        dtype=torch.float32,
    )
    label = torch.tensor([[1]], device=processor.device, dtype=torch.long)
    state["geometric_prompt"].append_points(point, label)
    return processor._forward_grounding(state)


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    if (args.x is None) != (args.y is None):
        raise ValueError("Provide both --x and --y, or omit both for interactive click mode.")

    if args.x is None:
        x, y = get_mouse_click(image)
    else:
        x, y = args.x, args.y

    processor = build_processor(args.checkpoint, args.confidence_threshold, args.device)

    with inference_context(processor.device):
        state = processor.set_image(image)
        output = add_mouse_click_prompt(processor, state, x, y)
    save_segmentation_outputs(image, output, args.output)
    print(f"Selected click: x={x:.2f}, y={y:.2f}")
    print(f"Saved overlay to {args.output}")
    print(f"Saved masks to {args.output.with_suffix('')}")
    print(f"Saved metadata to {args.output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
