from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def build_processor(
    checkpoint: str | None,
    confidence_threshold: float,
    device: str,
) -> Sam3Processor:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=device,
        load_from_HF=checkpoint is None,
    )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def inference_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def show_segmentation_results(
    original: Image.Image,
    preview: Image.Image,
    masks: list[Image.Image],
    scores: list[float],
) -> None:
    panels = [("Original", original), ("Preview", preview)]
    for index, mask in enumerate(masks):
        score = scores[index] if index < len(scores) else None
        title = f"Mask {index}"
        if score is not None:
            title = f"{title} score={score:.3f}"
        panels.append((title, mask))

    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (title, image) in zip(axes_list, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()

    for ax in axes_list[len(panels) :]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()


def save_segmentation_outputs(
    image: Image.Image,
    output: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masks = output["masks"].detach().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()

    image_array = np.asarray(image).astype(np.float32)
    overlay = image_array.copy()

    colors = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
        ],
        dtype=np.float32,
    )

    mask_dir = output_path.with_suffix("")
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index, mask in enumerate(masks):
        binary_mask = mask.squeeze().astype(bool)
        color = colors[index % len(colors)]
        overlay[binary_mask] = overlay[binary_mask] * 0.45 + color * 0.55

        mask_path = mask_dir / f"mask_{index:03d}.png"
        Image.fromarray((binary_mask.astype(np.uint8) * 255)).save(mask_path)

    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)

    metadata_path = output_path.with_suffix(".txt")
    with metadata_path.open("w", encoding="utf-8") as metadata:
        metadata.write(f"num_masks: {len(masks)}\n")
        for index, (score, box) in enumerate(zip(scores, boxes)):
            x0, y0, x1, y1 = box.tolist()
            metadata.write(
                f"mask_{index:03d}: score={score:.6f}, "
                f"box_xyxy=({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})\n"
            )
