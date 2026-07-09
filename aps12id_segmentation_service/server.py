from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile

from aps12id_segmentation_service.runtime import (
    ImageDecodeError,
    Sam3Segmenter,
    SegmentResult,
    decode_base64_image,
    decode_image_bytes,
)
from aps12id_segmentation_service.schemas import SegmentResponse


def create_app(segmenter: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if segmenter is not None:
            app.state.segmenter = segmenter
        else:
            checkpoint = os.getenv("SAM3_CHECKPOINT") or None
            device = os.getenv("SAM3_DEVICE", "auto")
            app.state.segmenter = await anyio.to_thread.run_sync(
                lambda: Sam3Segmenter(checkpoint=checkpoint, device=device)
            )
        yield

    app = FastAPI(
        title="APS 12-ID SAM3 Segmentation Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    if segmenter is not None:
        app.state.segmenter = segmenter

    async def run_segment(
        request: Request,
        *,
        forced_prompt: str | None = None,
        forced_threshold: float | None = None,
    ) -> SegmentResponse:
        payload = await _parse_segment_request(request)

        prompt = forced_prompt if forced_prompt is not None else payload["prompt"]
        threshold = (
            forced_threshold
            if forced_threshold is not None
            else payload["confidence_threshold"]
        )

        if (payload["x"] is None) != (payload["y"] is None):
            raise HTTPException(
                status_code=422,
                detail="Provide both x and y, or omit both.",
            )
        if not prompt and (payload["x"] is None or payload["y"] is None):
            raise HTTPException(
                status_code=422,
                detail="Provide prompt or both x and y.",
            )

        try:
            result: SegmentResult = await anyio.to_thread.run_sync(
                lambda: request.app.state.segmenter.segment(
                    payload["image"],
                    prompt=prompt,
                    x=payload["x"],
                    y=payload["y"],
                    confidence_threshold=threshold,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SegmentResponse.model_validate(result.__dict__)

    @app.post("/segment", response_model=SegmentResponse)
    async def segment(request: Request) -> SegmentResponse:
        return await run_segment(request)

    @app.post("/segment_12id_samv_holes", response_model=SegmentResponse)
    async def segment_12id_samv_holes(request: Request) -> SegmentResponse:
        return await run_segment(request, forced_prompt="hole", forced_threshold=0.5)

    @app.post("/segment_12id_samv_beam_tube", response_model=SegmentResponse)
    async def segment_12id_samv_beam_tube(request: Request) -> SegmentResponse:
        return await run_segment(
            request,
            forced_prompt="metal_probe",
            forced_threshold=0.5,
        )

    return app


async def _parse_segment_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        return await _parse_multipart_request(request)
    if content_type.startswith("application/json") or not content_type:
        return await _parse_json_request(request)
    raise HTTPException(
        status_code=415,
        detail="Use application/json or multipart/form-data.",
    )


async def _parse_json_request(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    encoded_image = body.get("encoded_image")
    if encoded_image is None:
        encoded_image = body.get("encoded_images")
    if not encoded_image:
        raise HTTPException(status_code=422, detail="encoded_image is required.")

    try:
        image = decode_base64_image(encoded_image)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "image": image,
        "prompt": _clean_prompt(body.get("prompt")),
        "x": _optional_float(body.get("x"), "x"),
        "y": _optional_float(body.get("y"), "y"),
        "confidence_threshold": _confidence_threshold(
            body.get("confidence_threshold", 0.5)
        ),
    }


async def _parse_multipart_request(request: Request) -> dict[str, Any]:
    form = await request.form()
    image_file = form.get("image_file")
    encoded_image = form.get("encoded_image") or form.get("encoded_images")

    if image_file and encoded_image:
        raise HTTPException(
            status_code=422,
            detail="Provide only one image source: encoded_image or image_file.",
        )
    if image_file is None and not encoded_image:
        raise HTTPException(
            status_code=422,
            detail="Provide encoded_image or image_file.",
        )

    try:
        if isinstance(image_file, UploadFile):
            image = decode_image_bytes(await image_file.read())
        else:
            image = decode_base64_image(str(encoded_image))
    except ImageDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "image": image,
        "prompt": _clean_prompt(form.get("prompt")),
        "x": _optional_float(form.get("x"), "x"),
        "y": _optional_float(form.get("y"), "y"),
        "confidence_threshold": _confidence_threshold(
            form.get("confidence_threshold", 0.5)
        ),
    }


def _clean_prompt(value: Any) -> str | None:
    if value is None:
        return None
    prompt = str(value).strip()
    return prompt or None


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a number.",
        ) from exc


def _confidence_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="confidence_threshold must be a number.",
        ) from exc
    if threshold < 0 or threshold > 1:
        raise HTTPException(
            status_code=422,
            detail="confidence_threshold must be between 0 and 1.",
        )
    return threshold


app = create_app()
