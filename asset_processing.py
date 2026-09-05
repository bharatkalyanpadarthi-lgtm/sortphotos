#!/usr/bin/env python3
"""Shared single-read image decoding and derived asset metadata."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedImage:
    path: Path
    bgr: np.ndarray
    sha256: str
    width: int
    height: int
    orientation: int


@dataclass(frozen=True)
class DecodedAsset(DecodedImage):
    pixel_sha256: str
    phash_bits: np.ndarray


def perceptual_hash(bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    resized = cv2.resize(
        gray,
        (hash_size * 4, hash_size * 4),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:hash_size, :hash_size].flatten()
    median = float(np.median(low[1:]))
    return low > median


def decoded_pixel_sha256(bgr: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(bgr.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(bgr).tobytes())
    return digest.hexdigest()


def phash_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def _decode_with_pillow(raw: bytes) -> tuple[np.ndarray | None, int]:
    try:
        from PIL import Image, ImageFile, ImageOps
        import pillow_heif

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if hasattr(pillow_heif, "register_heif_opener"):
            pillow_heif.register_heif_opener()
        with Image.open(io.BytesIO(raw)) as image:
            orientation = int(image.getexif().get(274, 1) or 1)
            image = ImageOps.exif_transpose(image)
            image.load()
            rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), orientation
    except Exception:
        return None, 1


def decode_image(path: Path) -> DecodedImage | None:
    """Read and decode one image without repeating filesystem reads."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    orientation = 1
    try:
        encoded = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        bgr = None
    if bgr is None or getattr(bgr, "size", 0) == 0:
        bgr, orientation = _decode_with_pillow(raw)
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    height, width = bgr.shape[:2]
    return DecodedImage(
        path=path,
        bgr=bgr,
        sha256=digest,
        width=int(width),
        height=int(height),
        orientation=orientation,
    )


def load_decoded_asset(path: Path) -> DecodedAsset | None:
    """Read a file once, decode it once, and derive reusable image metadata."""
    decoded = decode_image(path)
    if decoded is None:
        return None
    return DecodedAsset(
        path=decoded.path,
        bgr=decoded.bgr,
        sha256=decoded.sha256,
        width=decoded.width,
        height=decoded.height,
        orientation=decoded.orientation,
        pixel_sha256=decoded_pixel_sha256(decoded.bgr),
        phash_bits=perceptual_hash(decoded.bgr),
    )
