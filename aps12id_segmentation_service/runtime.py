from __future__ import annotations

import base64
import binascii
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """Raised when request image data cannot be decoded as an image."""


@dataclass(frozen=True)
class SegmentResult:
    preview: dict[str, str]
    masks: list[dict[str, Any]]
    mask_metadata: list[dict[str, Any]]
    image_metadata: dict[str, int]


def decode_base64_image(encoded_image: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("encoded_image must be raw base64 image bytes.") from exc
    return decode_image_bytes(image_bytes)


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("Image data could not be decoded by Pillow.") from exc


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _media_payload(image: Image.Image) -> dict[str, str]:
    return {
        "data": encode_png(image),
        "encoding": "base64",
        "format": "png",
        "content_type": "image/png",
    }


def _render_result(image: Image.Image, output: dict[str, Any]) -> SegmentResult:
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

    encoded_masks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        binary_mask = mask.squeeze().astype(bool)
        color = colors[index % len(colors)]
        overlay[binary_mask] = overlay[binary_mask] * 0.45 + color * 0.55

        mask_image = Image.fromarray(binary_mask.astype(np.uint8) * 255)
        encoded_masks.append({"id": index, **_media_payload(mask_image)})

        metadata.append(
            {
                "id": index,
                "score": float(scores[index]),
                "box_xyxy": [float(value) for value in boxes[index].tolist()],
                "area_pixels": int(binary_mask.sum()),
            }
        )

    preview_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    return SegmentResult(
        preview=_media_payload(preview_image),
        masks=encoded_masks,
        mask_metadata=metadata,
        image_metadata={"width": image.width, "height": image.height},
    )


def _inference_context(device: str):
    if device == "cuda":
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _prefer_submodule_sam3_package() -> None:
    submodule_root = Path(__file__).resolve().parent.parent / "sam3"
    if not (submodule_root / "sam3" / "__init__.py").exists():
        return

    submodule_root_text = str(submodule_root)
    if submodule_root_text not in sys.path:
        sys.path.insert(0, submodule_root_text)

    loaded_sam3 = sys.modules.get("sam3")
    if loaded_sam3 is not None and getattr(loaded_sam3, "__file__", None) is None:
        del sys.modules["sam3"]


def _add_mouse_click_prompt(processor: Any, state: dict[str, Any], x: float, y: float) -> dict[str, Any]:
    import torch

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


class Sam3Segmenter:
    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ) -> None:
        _prefer_submodule_sam3_package()

        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            device=device,
            load_from_HF=checkpoint is None,
        )
        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self._lock = threading.Lock()

    def segment(
        self,
        image: Image.Image,
        *,
        prompt: str | None,
        x: float | None,
        y: float | None,
        confidence_threshold: float,
    ) -> SegmentResult:
        with self._lock:
            previous_threshold = self.processor.confidence_threshold
            self.processor.confidence_threshold = confidence_threshold
            try:
                with _inference_context(self.processor.device):
                    state = self.processor.set_image(image)
                    if prompt:
                        state = self.processor.set_text_prompt(state=state, prompt=prompt)
                    if x is not None and y is not None:
                        state = _add_mouse_click_prompt(self.processor, state, x, y)
            finally:
                self.processor.confidence_threshold = previous_threshold

        return _render_result(image, state)
