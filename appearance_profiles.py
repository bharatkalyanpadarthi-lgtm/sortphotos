#!/usr/bin/env python3
"""Conservative lighting and capture-era labels for identity prototypes."""

from __future__ import annotations

import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


LOW_LIGHT_MEAN = 68.0
MIN_ERA_SPAN_SECONDS = 365.0 * 24.0 * 60.0 * 60.0
EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


@lru_cache(maxsize=512)
def lighting_label(crop_jpeg: bytes) -> str:
    if not crop_jpeg:
        return "unknown"
    try:
        with Image.open(io.BytesIO(crop_jpeg)) as image:
            mean = float(ImageStat.Stat(image.convert("L")).mean[0])
    except (OSError, ValueError, TypeError):
        return "unknown"
    return "low_light" if mean < LOW_LIGHT_MEAN else "normal_light"


def _parse_exif_date(value: object) -> float:
    text = str(value or "").strip().replace("\x00", "")
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return float(time.mktime(time.strptime(text, pattern)))
        except (OverflowError, ValueError):
            continue
    return 0.0


@lru_cache(maxsize=16384)
def _capture_timestamp_cached(path_text: str, mtime_ns: int, byte_size: int) -> float:
    del mtime_ns, byte_size
    path = Path(path_text)
    if path.suffix.casefold() not in {".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".webp"}:
        return 0.0
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in EXIF_DATE_TAGS:
                timestamp = _parse_exif_date(exif.get(tag))
                if timestamp > 0:
                    return timestamp
    except (OSError, ValueError, TypeError, SyntaxError):
        return 0.0
    return 0.0


def capture_timestamp(path: Path | str) -> float:
    candidate = Path(path).expanduser()
    try:
        stat = candidate.stat()
    except OSError:
        return 0.0
    return _capture_timestamp_cached(
        str(candidate.resolve(strict=False)), int(stat.st_mtime_ns), int(stat.st_size)
    )


def era_cutoff(timestamps: list[float]) -> float:
    valid = sorted({float(value) for value in timestamps if float(value) > 0})
    if len(valid) < 3 or valid[-1] - valid[0] < MIN_ERA_SPAN_SECONDS:
        return 0.0
    return float(np.median(np.asarray(valid, dtype=np.float64)))


def labels(
    *,
    light: str = "unknown",
    timestamp: float = 0.0,
    person_era_cutoff: float = 0.0,
) -> tuple[str, ...]:
    values: list[str] = []
    if light in {"low_light", "normal_light"}:
        values.append(light)
    if timestamp > 0 and person_era_cutoff > 0:
        values.append("era_older" if timestamp < person_era_cutoff else "era_newer")
    return tuple(values)


def query_attributes(crop_jpeg: bytes, source: Path | str) -> tuple[str, float]:
    return lighting_label(crop_jpeg), capture_timestamp(source)
