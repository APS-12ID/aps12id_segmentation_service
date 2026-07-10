from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from sam3_common import show_segmentation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the SAM3 FastAPI segmentation server with an image file."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--prompt", help='Text prompt, for example "hole".')
    parser.add_argument("--x", type=float, help="Point prompt x coordinate in pixels.")
    parser.add_argument("--y", type=float, help="Point prompt y coordinate in pixels.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 confidence score.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host. Ignored if --url is provided.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port. Ignored if --url is provided.",
    )
    parser.add_argument(
        "--url",
        help="Full server base URL, for example http://localhost:8000.",
    )
    parser.add_argument(
        "--endpoint",
        default="/segment",
        help="Endpoint path, for example /segment_12id_samv_holes.",
    )
    return parser.parse_args()


def build_url(args: argparse.Namespace) -> str:
    base_url = args.url or f"http://{args.host}:{args.port}"
    return f"{base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"


def encode_multipart_form(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----aps12id-{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def post_image(args: argparse.Namespace) -> dict[str, Any]:
    fields = {"confidence_threshold": str(args.confidence_threshold)}
    if args.prompt:
        fields["prompt"] = args.prompt
    if args.x is not None:
        fields["x"] = str(args.x)
    if args.y is not None:
        fields["y"] = str(args.y)

    content_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    body, request_content_type = encode_multipart_form(
        fields,
        {
            "image_file": (
                args.image.name,
                args.image.read_bytes(),
                content_type,
            )
        },
    )

    request = urllib.request.Request(
        build_url(args),
        data=body,
        headers={"Content-Type": request_content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to server: {exc}") from exc


def decode_png_payload(payload: dict[str, str]) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(payload["data"]))).convert("RGB")


def show_response(image_path: Path, response: dict[str, Any]) -> None:
    original = Image.open(image_path).convert("RGB")
    preview = decode_png_payload(response["preview"])
    masks = [decode_png_payload(mask) for mask in response["masks"]]
    scores = [item["score"] for item in response["mask_metadata"]]
    show_segmentation_results(original, preview, masks, scores)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    if (args.x is None) != (args.y is None):
        raise ValueError("Provide both --x and --y, or omit both.")
    if not args.prompt and args.x is None:
        raise ValueError("Provide --prompt or both --x and --y.")

    response = post_image(args)
    print(f"Received {len(response['masks'])} mask(s).")
    for item in response["mask_metadata"]:
        print(
            f"mask_{item['id']:03d}: score={item['score']:.6f}, "
            f"box_xyxy={item['box_xyxy']}, area_pixels={item['area_pixels']}"
        )
    show_response(args.image, response)


if __name__ == "__main__":
    main()
