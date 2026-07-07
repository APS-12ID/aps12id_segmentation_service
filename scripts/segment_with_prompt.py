from __future__ import annotations

import argparse
from pathlib import Path

from sam3_common import (
    build_processor,
    inference_context,
    load_image,
    save_segmentation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment an image with a SAM3 text prompt.")
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("prompt", help='Text prompt, for example "person" or "red bead".')
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/sam3_prompt_overlay.png"),
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


def main() -> None:
    args = parse_args()
    image = load_image(args.image)
    processor = build_processor(args.checkpoint, args.confidence_threshold, args.device)

    with inference_context(processor.device):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
    save_segmentation_outputs(image, output, args.output)
    print(f"Saved overlay to {args.output}")
    print(f"Saved masks to {args.output.with_suffix('')}")
    print(f"Saved metadata to {args.output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
