from __future__ import annotations

from pydantic import BaseModel, Field


class MediaPayload(BaseModel):
    data: str
    encoding: str = "base64"
    format: str = "png"
    content_type: str = "image/png"


class MaskPayload(MediaPayload):
    id: int


class MaskMetadata(BaseModel):
    id: int
    score: float
    box_xyxy: list[float] = Field(min_length=4, max_length=4)
    area_pixels: int


class ImageMetadata(BaseModel):
    width: int
    height: int


class SegmentResponse(BaseModel):
    preview: MediaPayload
    masks: list[MaskPayload]
    mask_metadata: list[MaskMetadata]
    image_metadata: ImageMetadata

