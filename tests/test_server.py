from __future__ import annotations

import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from aps12id_segmentation_service.runtime import SegmentResult, encode_png
from aps12id_segmentation_service.server import create_app


class FakeSegmenter:
    def __init__(self) -> None:
        self.calls = []

    def segment(self, image, *, prompt, x, y, confidence_threshold):
        self.calls.append(
            {
                "width": image.width,
                "height": image.height,
                "prompt": prompt,
                "x": x,
                "y": y,
                "confidence_threshold": confidence_threshold,
            }
        )
        preview = {
            "data": encode_png(image),
            "encoding": "base64",
            "format": "png",
            "content_type": "image/png",
        }
        mask_image = Image.new("L", image.size, 255)
        return SegmentResult(
            preview=preview,
            masks=[{"id": 0, **preview, "data": encode_png(mask_image)}],
            mask_metadata=[
                {
                    "id": 0,
                    "score": 0.9,
                    "box_xyxy": [0.0, 0.0, float(image.width), float(image.height)],
                    "area_pixels": image.width * image.height,
                }
            ],
            image_metadata={"width": image.width, "height": image.height},
        )


def _png_bytes() -> bytes:
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _client() -> tuple[TestClient, FakeSegmenter]:
    segmenter = FakeSegmenter()
    return TestClient(create_app(segmenter=segmenter)), segmenter


def test_segment_accepts_raw_base64_json() -> None:
    client, segmenter = _client()
    encoded_image = base64.b64encode(_png_bytes()).decode("ascii")

    response = client.post(
        "/segment",
        json={
            "encoded_image": encoded_image,
            "prompt": "hole",
            "confidence_threshold": 0.7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_metadata"] == {"width": 3, "height": 2}
    assert payload["mask_metadata"][0]["id"] == payload["masks"][0]["id"]
    Image.open(BytesIO(base64.b64decode(payload["preview"]["data"])))
    assert segmenter.calls[-1]["prompt"] == "hole"
    assert segmenter.calls[-1]["confidence_threshold"] == 0.7


def test_segment_accepts_multipart_file() -> None:
    client, segmenter = _client()

    response = client.post(
        "/segment",
        data={"x": "1", "y": "1"},
        files={"image_file": ("image.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert segmenter.calls[-1]["x"] == 1.0
    assert segmenter.calls[-1]["y"] == 1.0


def test_segment_requires_prompt_or_complete_point() -> None:
    client, _ = _client()
    encoded_image = base64.b64encode(_png_bytes()).decode("ascii")

    response = client.post("/segment", json={"encoded_image": encoded_image})

    assert response.status_code == 422
    assert "prompt or both x and y" in response.json()["detail"]


def test_segment_rejects_prefixed_base64() -> None:
    client, _ = _client()
    encoded_image = base64.b64encode(_png_bytes()).decode("ascii")

    response = client.post(
        "/segment",
        json={"encoded_image": f"data:image/png;base64,{encoded_image}", "prompt": "hole"},
    )

    assert response.status_code == 422


def test_segment_rejects_partial_point() -> None:
    client, _ = _client()
    encoded_image = base64.b64encode(_png_bytes()).decode("ascii")

    response = client.post(
        "/segment",
        json={"encoded_image": encoded_image, "x": 1},
    )

    assert response.status_code == 422


def test_convenience_endpoints_override_prompt_and_threshold() -> None:
    client, segmenter = _client()

    response = client.post(
        "/segment_12id_samv_beam_tube",
        files={"image_file": ("image.png", _png_bytes(), "image/png")},
        data={"prompt": "ignored", "confidence_threshold": "0.9"},
    )

    assert response.status_code == 200
    assert segmenter.calls[-1]["prompt"] == "metal_probe"
    assert segmenter.calls[-1]["confidence_threshold"] == 0.5


def test_segment_12id_samh_holes_overrides_prompt_and_threshold() -> None:
    client, segmenter = _client()

    response = client.post(
        "/segment_12id_samh_holes",
        files={"image_file": ("image.png", _png_bytes(), "image/png")},
        data={"prompt": "ignored", "confidence_threshold": "0.9"},
    )

    assert response.status_code == 200
    assert segmenter.calls[-1]["prompt"] == "hole"
    assert segmenter.calls[-1]["confidence_threshold"] == 0.5
