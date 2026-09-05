#!/usr/bin/env python3
"""Pure image geometry and fallback-view helpers for face detection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

RECOVERY_VERSION = 2
QUALITY_VERSION = 1


@dataclass(frozen=True)
class DetectionView:
    image: np.ndarray
    to_original: np.ndarray
    kind: str


def original_geometry(bbox, keypoints, transform):
    x1, y1, x2, y2 = map(float, bbox)
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ transform.T
    box = (float(corners[:, 0].min()), float(corners[:, 1].min()),
           float(corners[:, 0].max()), float(corners[:, 1].max()))
    points = ()
    if keypoints is not None and len(keypoints):
        mapped = np.column_stack((np.asarray(keypoints), np.ones(len(keypoints)))) @ transform.T
        points = tuple(tuple(map(float, p[:2])) for p in mapped)
    return box, points


def same_detection(left, right, threshold=.5) -> bool:
    if len(left) != 4 or len(right) != 4:
        return False
    area_left = max(0, left[2]-left[0]) * max(0, left[3]-left[1])
    area_right = max(0, right[2]-right[0]) * max(0, right[3]-right[1])
    intersection = max(0, min(left[2], right[2])-max(left[0], right[0])) * max(0, min(left[3], right[3])-max(left[1], right[1]))
    return intersection / max(1, area_left + area_right - intersection) >= threshold


def sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def square_pad_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = x2 - x1, y2 - y1
    center_x, center_y = x1 + width / 2, y1 + height / 2
    half = max(width, height) / 2 * (1 + pad_ratio)
    return (
        max(0, int(round(center_x - half))),
        max(0, int(round(center_y - half))),
        min(image_width, int(round(center_x + half))),
        min(image_height, int(round(center_y + half))),
    )


def yaw_proxy_from_keypoints(keypoints, bbox: np.ndarray) -> float:
    if keypoints is None or len(keypoints) < 3:
        return 0.5
    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    face_width = max(1.0, bbox[2] - bbox[0])
    offset = abs(nose[0] - eye_mid_x) / face_width
    return float(np.clip(1.0 - offset * 4.0, 0.0, 1.0))


def pose_label_from_keypoints(keypoints, bbox: np.ndarray) -> str:
    """Return a stable coarse pose bucket without changing the quality score.

    Existing caches store only the unsigned yaw proxy, so they are upgraded to
    ``profile_unknown`` when appropriate. Newly detected faces retain the side
    information needed by pose-specific identity profiles.
    """
    if keypoints is None or len(keypoints) < 3:
        return "unknown"
    left_eye, right_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    face_width = max(1.0, float(bbox[2]) - float(bbox[0]))
    signed_offset = (float(nose[0]) - eye_mid_x) / face_width
    if abs(signed_offset) <= 0.045:
        return "frontal"
    return "left_profile" if signed_offset < 0 else "right_profile"


def pose_label_from_yaw_proxy(yaw_proxy: float) -> str:
    """Best-effort migration label for detections cached before pose support."""
    if float(yaw_proxy) >= 0.82:
        return "frontal"
    return "profile_unknown"


def quality_score(
    detection_score: float,
    bbox_size: float,
    image_sharpness: float,
    yaw_proxy: float,
) -> float:
    score_detection = np.clip((detection_score - 0.4) / 0.55, 0.0, 1.0)
    score_size = np.clip((bbox_size - 60.0) / 240.0, 0.0, 1.0)
    score_sharpness = np.clip(
        np.log1p(image_sharpness) / np.log1p(400.0), 0.0, 1.0
    )
    parts = np.array(
        [score_detection, score_size, score_sharpness, yaw_proxy]
    ) + 1e-3
    return float(np.exp(np.log(parts).mean()))


def fallback_detection_views(
    image: np.ndarray,
    *,
    max_dimension: int,
) -> Iterable[np.ndarray]:
    for frame in fallback_detection_frames(image, max_dimension=max_dimension):
        yield frame.image


def fallback_detection_frames(image: np.ndarray, *, max_dimension: int) -> Iterable[DetectionView]:
    """Yield recovery views only after normal full-frame detection fails."""
    height, width = image.shape[:2]
    context = image
    largest_dimension = max(height, width)
    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension
        context = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    context_height, context_width = context.shape[:2]
    scale = np.diag([width / context_width, height / context_height, 1.0])
    for padding_ratio in (0.35, 0.55):
        pad_y = max(16, int(round(context_height * padding_ratio)))
        pad_x = max(16, int(round(context_width * padding_ratio)))
        padded = cv2.copyMakeBorder(
            context,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_REPLICATE,
        )
        yield DetectionView(padded, scale @ np.array([[1, 0, -pad_x], [0, 1, -pad_y], [0, 0, 1]]), "padding")

    height, width = context.shape[:2]
    if min(height, width) >= 512:
        if height >= width * 1.15:
            tile_width = width
            tile_height = max(256, int(round(height * 0.62)))
            y_starts = sorted(
                {0, max(0, (height - tile_height) // 2), max(0, height - tile_height)}
            )
            regions = [(0, y) for y in y_starts]
        elif width >= height * 1.15:
            tile_width = max(256, int(round(width * 0.62)))
            tile_height = height
            x_starts = sorted(
                {0, max(0, (width - tile_width) // 2), max(0, width - tile_width)}
            )
            regions = [(x, 0) for x in x_starts]
        else:
            tile_width = max(256, int(round(width * 0.70)))
            tile_height = max(256, int(round(height * 0.70)))
            x_starts = sorted(
                {0, max(0, (width - tile_width) // 2), max(0, width - tile_width)}
            )
            y_starts = sorted(
                {0, max(0, (height - tile_height) // 2), max(0, height - tile_height)}
            )
            regions = [(x, y) for y in y_starts for x in x_starts]
        for x, y in regions:
            tile = context[y:min(height, y + tile_height), x:min(width, x + tile_width)]
            if tile.size:
                yield DetectionView(tile, scale @ np.array([[1, 0, x], [0, 1, y], [0, 0, 1]]), "tile")

    rotations = (
        (cv2.ROTATE_90_CLOCKWISE, [[0, 1, 0], [-1, 0, height], [0, 0, 1]]),
        (cv2.ROTATE_180, [[-1, 0, width], [0, -1, height], [0, 0, 1]]),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, [[0, -1, width], [1, 0, 0], [0, 0, 1]]),
    )
    for rotation, transform in rotations:
        yield DetectionView(cv2.rotate(context, rotation), scale @ np.asarray(transform), "rotation")
