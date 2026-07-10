# APS 12-ID SAM3 Segmentation Service

FastAPI service for running SAM3 image segmentation on APS 12-ID optical images.
The server loads SAM3 once at startup and keeps the model resident for subsequent
requests.

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd aps12id_segmentation_service
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Install dependencies with uv:

```bash
uv sync
```

Download the SAM3 model checkpoint from:

https://anl.box.com/s/yzr8w7ys9bxv8a8i7k01qp2l2ryrwsaq

Save it somewhere stable, for example `checkpoints/sam3.pt`, then set
`SAM3_CHECKPOINT` to that local file before launching the server:

```bash
export SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
```

## Run The Server

```bash
SAM3_DEVICE=auto \
uv run uvicorn aps12id_segmentation_service.server:app --host 0.0.0.0 --port 8000
```

`SAM3_DEVICE` may be `auto`, `cuda`, or `cpu`; the default is `auto`.

As a backup route, if `SAM3_CHECKPOINT` is omitted, SAM3 will attempt to
download the checkpoint from Hugging Face. That requires Hugging Face access to
the SAM3 checkpoint and local Hugging Face authentication, so the Box download
above is the recommended installation path.

Use one uvicorn worker unless you intentionally want each worker to load its
own SAM3 model copy.

## Endpoints

All segmentation endpoints are `POST` requests:

- `/segment`
- `/segment_12id_samv_holes`
- `/segment_12id_samh_holes`
- `/segment_12id_samv_beam_tube`

`/segment` requires one image source and at least one prompt source:

- `encoded_image`: raw base64 image bytes, with no `data:image/...;base64,`
  prefix.
- `image_file`: image file sent as `multipart/form-data`.
- `prompt`: text prompt.
- `x` and `y`: positive point prompt in image pixel coordinates.
- `confidence_threshold`: optional score threshold from `0` to `1`, default
  `0.5`.

If both `prompt` and `(x, y)` are provided, SAM3 uses both. If one coordinate is
provided, the other coordinate is required.

The convenience endpoints use the same image input format but force the prompt
and confidence threshold:

- `/segment_12id_samv_holes`: `prompt="hole"`, `confidence_threshold=0.3`
- `/segment_12id_samh_holes`: `prompt="hole"`, `confidence_threshold=0.3`
- `/segment_12id_samv_beam_tube`: `prompt="metal_probe"`,
  `confidence_threshold=0.5`

## Request Examples

Raw base64 JSON:

```bash
python - <<'PY'
import base64
import json
from pathlib import Path

payload = {
    "encoded_image": base64.b64encode(Path("image.png").read_bytes()).decode("ascii"),
    "prompt": "hole",
    "confidence_threshold": 0.5,
}
Path("request.json").write_text(json.dumps(payload), encoding="utf-8")
PY

curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  --data @request.json
```

Multipart file upload with a text prompt:

```bash
curl -X POST http://localhost:8000/segment \
  -F "image_file=@/path/to/image.png" \
  -F "prompt=hole" \
  -F "confidence_threshold=0.5"
```

Multipart file upload with a point prompt:

```bash
curl -X POST http://localhost:8000/segment \
  -F "image_file=@/path/to/image.png" \
  -F "x=320" \
  -F "y=240"
```

12-ID hole segmentation:

```bash
curl -X POST http://localhost:8000/segment_12id_samv_holes \
  -F "image_file=@/path/to/image.png"
```

12-ID beam tube segmentation:

```bash
curl -X POST http://localhost:8000/segment_12id_samv_beam_tube \
  -F "image_file=@/path/to/image.png"
```

You can also query the server with the example client script. It uploads an
image file, decodes the returned preview and masks, prints mask metadata, and
displays the results with matplotlib:

```bash
uv run python scripts/query_segmentation_server.py /path/to/image.png \
  --prompt hole \
  --confidence-threshold 0.5
```

Point prompt example:

```bash
uv run python scripts/query_segmentation_server.py /path/to/image.png \
  --x 320 \
  --y 240
```

Use `--host`, `--port`, or `--url` to target a non-default server.

## Local Segmentation Scripts

Run text-prompt segmentation locally with:

```bash
uv run python scripts/segment_with_prompt.py /path/to/image.png "hole"
```

Run point-prompt segmentation with explicit coordinates:

```bash
uv run python scripts/segment_with_mouse_click.py /path/to/image.png \
  --x 320 \
  --y 240
```

If `--x` and `--y` are omitted, the point-prompt script opens an interactive
window for selecting the point. Add `--show` to either script to display the
original image, overlay preview, and generated masks with their confidence
scores after segmentation. The output files are still saved normally.

## Response Format

The service returns PNG preview and mask images as base64 strings:

```json
{
  "preview": {
    "data": "<base64 png>",
    "encoding": "base64",
    "format": "png",
    "content_type": "image/png"
  },
  "masks": [
    {
      "id": 0,
      "data": "<base64 png mask>",
      "encoding": "base64",
      "format": "png",
      "content_type": "image/png"
    }
  ],
  "mask_metadata": [
    {
      "id": 0,
      "score": 0.7,
      "box_xyxy": [8.0, 400.0, 150.0, 480.0],
      "area_pixels": 12345
    }
  ],
  "image_metadata": {
    "width": 1024,
    "height": 768
  }
}
```

## Tests

The automated tests mock the SAM3 runtime, so they do not require a real
checkpoint:

```bash
uv run pytest
```
