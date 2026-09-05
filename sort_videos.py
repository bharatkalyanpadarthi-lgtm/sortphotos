#!/usr/bin/env python3
"""Identify people in intake videos and file on one recognized frame."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.utils import face_align

import operation_ledger
import pipeline_paths
import appearance_profiles
import face_detection
import identity_hard_negatives
import identity_profiles
import secondary_identity_matcher
import sort_photos


DEFAULT_SORTED = pipeline_paths.SORTED_ROOT
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
DEFAULT_SOURCE_REVIEW = DEFAULT_SORTED / "_source_review"
LEGACY_VIDEO_INBOX = Path.home() / "Pictures" / "videos"
VIDEO_REPORT_DIR = DEFAULT_SOURCE_REVIEW / "video_reports"
VIDEO_REVIEW_ROOT = DEFAULT_SOURCE_REVIEW / "unassigned_intake" / "videos"

MIN_SAMPLES = 8
INITIAL_MAX_SAMPLES = 12
RECOVERY_MIN_SAMPLES = 24
MAX_SAMPLES = 48
MATCH_MAX_DISTANCE = 0.34
MATCH_MIN_MARGIN = 0.06
SUPPORT_MAX_DISTANCE = 0.50
SUPPORT_MIN_MARGIN = 0.08
# A single recognized frame is enough to file a video. Distance and margin
# checks in _match_embedding_evidence still decide whether a frame counts.
MIN_MATCHED_FRAMES = 1
MIN_MATCH_RATIO = 0.0
MIN_SUPPORT_FRAMES = 1
MIN_SUPPORT_RATIO = 0.0
MULTIPLE_PERSON_RATIO = 0.55
MIN_IDENTITY_REFERENCE_FACES = 3
MIN_FACE_PIXELS = 60
MIN_DETECTION_SCORE = 0.65
MAX_FRAME_EDGE = 1600
MIN_INTERNAL_FREE_BYTES = 20 * 1024 * 1024 * 1024
MIN_RUNTIME_MEMORY_MB = 1_200
YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_MIN_SCORE = 0.85


@dataclass(frozen=True)
class VideoDecision:
    status: str
    person: str
    reason: str
    votes: dict[str, int]
    sampled_frames: int
    face_frames: int
    detected_faces: int
    confident_matches: int
    support_votes: dict[str, int] = field(default_factory=dict)
    fallback_face_frames: int = 0
    analysis_passes: int = 1


@dataclass
class VideoIdentityMatcher:
    identity_db: sort_photos.IdentityDB
    hard_negatives: dict[str, list[np.ndarray]]
    secondary: secondary_identity_matcher.SecondaryMatcher | None = None

    def __post_init__(self) -> None:
        self.identity_db = sort_photos.normalize_identity_db(self.identity_db)
        self.identities = {
            name: centroid
            for name, centroid in self.identity_db.identities.items()
            if self.identity_db.source_counts.get(name, 0) >= MIN_IDENTITY_REFERENCE_FACES
        }
        self.prototypes = {
            name: self.identity_db.prototypes.get(name, [centroid])
            for name, centroid in self.identities.items()
        }
        self.compiled = identity_profiles.CompiledProfiles(
            self.identities, self.prototypes, pose_prototypes=self.identity_db.pose_prototypes,
            appearance_prototypes=self.identity_db.appearance_prototypes,
            appearance_era_cutoffs=self.identity_db.appearance_era_cutoffs,
            hard_negatives=self.hard_negatives)

    def match(
        self,
        frame: np.ndarray,
        face,
        source: Path,
    ) -> tuple[str | None, str, float, float]:
        embedding = getattr(face, "normed_embedding", None)
        bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
        if embedding is None or bbox.size < 4 or not self.identities:
            return None, "none", 1.0, 0.0
        crop_jpeg = _face_crop_jpeg(frame, face)
        pose = face_detection.pose_label_from_keypoints(
            getattr(face, "kps", None), bbox
        )
        lighting, captured_at = appearance_profiles.query_attributes(
            crop_jpeg, source
        )
        candidates = identity_profiles.rank_candidates(
            np.asarray(embedding, dtype=np.float32),
            self.identities,
            self.prototypes,
            pose_label=pose,
            pose_prototypes=self.identity_db.pose_prototypes,
            lighting_label=lighting,
            capture_timestamp=captured_at,
            appearance_prototypes=self.identity_db.appearance_prototypes,
            appearance_era_cutoffs=self.identity_db.appearance_era_cutoffs,
            hard_negatives=self.hard_negatives,
            compiled=self.compiled,
        )
        if not candidates:
            return None, "none", 1.0, 0.0
        best = candidates[0]
        margin = identity_profiles.candidate_margin(candidates)
        strict_threshold = min(
            MATCH_MAX_DISTANCE,
            self.identity_db.strict_thresholds.get(best.name, MATCH_MAX_DISTANCE),
        )
        consensus_threshold = min(
            MATCH_MAX_DISTANCE,
            self.identity_db.match_thresholds.get(best.name, MATCH_MAX_DISTANCE),
        )
        if best.distance <= strict_threshold and margin >= MATCH_MIN_MARGIN:
            return best.name, "strict", float(best.distance), float(margin)
        if best.distance <= consensus_threshold and margin >= SUPPORT_MIN_MARGIN:
            return best.name, "support", float(best.distance), float(margin)
        face_pixels = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
        secondary_ok = (
            self.secondary is not None
            and crop_jpeg
            and face_pixels >= MIN_FACE_PIXELS
            and float(getattr(face, "det_score", 0.0)) >= MIN_DETECTION_SCORE
            and best.distance <= consensus_threshold + 0.08
            and margin >= 0.03
            and self.secondary.verify(crop_jpeg, best.name).accepted
        )
        if secondary_ok:
            return best.name, "strict", float(best.distance), float(margin)
        return None, "none", float(best.distance), float(margin)

    def flush(self) -> None:
        if self.secondary is not None:
            self.secondary.flush()


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0 or sample_count <= 0:
        return []
    count = min(frame_count, sample_count)
    if count == 1:
        return [max(0, frame_count // 2)]
    margin = min(max(1, frame_count // 50), max(1, frame_count // 4))
    start = margin
    end = max(start, frame_count - 1 - margin)
    return sorted({int(round(value)) for value in np.linspace(start, end, count)})


def decide_person(
    votes: Counter[str],
    *,
    support_votes: Counter[str] | None = None,
    sampled_frames: int,
    face_frames: int,
    detected_faces: int,
    confident_matches: int,
    min_matched_frames: int = MIN_MATCHED_FRAMES,
    min_match_ratio: float = MIN_MATCH_RATIO,
    multiple_person_ratio: float = MULTIPLE_PERSON_RATIO,
) -> VideoDecision:
    vote_dict = dict(sorted(votes.items(), key=lambda item: (-item[1], item[0].casefold())))
    support_votes = support_votes or Counter()
    support_vote_dict = dict(sorted(
        support_votes.items(), key=lambda item: (-item[1], item[0].casefold()),
    ))

    def decision(status: str, person: str, reason: str) -> VideoDecision:
        return VideoDecision(
            status, person, reason, vote_dict,
            sampled_frames, face_frames, detected_faces, confident_matches,
            support_votes=support_vote_dict,
        )

    if sampled_frames <= 0:
        return decision("unreadable_video", "", "No video frames could be read.")
    if face_frames <= 0:
        return decision("no_usable_face", "", "No usable face was found in sampled frames.")

    ranked = votes.most_common()
    support_ranked = support_votes.most_common()

    if ranked and support_ranked and ranked[0][0] != support_ranked[0][0]:
        return decision(
            "multiple_people", "",
            f"Strict and supporting evidence disagree: "
            f"{ranked[0][0]} and {support_ranked[0][0]}.",
        )

    if len(support_ranked) > 1:
        support_name, support_count = support_ranked[0]
        second_name, second_count = support_ranked[1]
        second_ratio = second_count / max(1, support_count)
        if second_count >= MIN_SUPPORT_FRAMES and second_ratio >= multiple_person_ratio:
            return decision(
                "multiple_people", "",
                f"Multiple known people appear in sampled frames: "
                f"{support_name} ({support_count}) and {second_name} ({second_count}).",
            )

    if ranked:
        top_name, top_votes = ranked[0]
        top_ratio = top_votes / max(1, face_frames)
        if top_votes >= min_matched_frames and top_ratio >= min_match_ratio:
            if len(ranked) > 1:
                second_name, second_votes = ranked[1]
                second_ratio = second_votes / max(1, top_votes)
                if second_votes >= min_matched_frames and second_ratio >= multiple_person_ratio:
                    return decision(
                        "multiple_people", "",
                        f"Multiple known people appear repeatedly: {top_name} ({top_votes}) and "
                        f"{second_name} ({second_votes}).",
                    )
            return decision(
                "matched", top_name,
                f"Matched {top_name} in {top_votes}/{face_frames} sampled face frames; "
                "one recognized frame is sufficient.",
            )

    if support_ranked:
        support_name, support_count = support_ranked[0]
        support_ratio = support_count / max(1, face_frames)
        strict_name = ranked[0][0] if ranked else support_name
        if (
            support_name == strict_name
            and support_count >= MIN_SUPPORT_FRAMES
            and support_ratio >= MIN_SUPPORT_RATIO
        ):
            if len(support_ranked) > 1:
                second_name, second_count = support_ranked[1]
                second_ratio = second_count / max(1, support_count)
                if second_count >= MIN_SUPPORT_FRAMES and second_ratio >= multiple_person_ratio:
                    return decision(
                        "multiple_people", "",
                        f"Multiple known people have repeated supporting evidence: "
                        f"{support_name} ({support_count}) and {second_name} ({second_count}).",
                    )
            return decision(
                "matched", support_name,
                f"Matched {support_name} using supporting evidence in "
                f"{support_count}/{face_frames} sampled face frames; "
                "one recognized frame is sufficient.",
            )

    if ranked:
        top_name, top_votes = ranked[0]
        reason = f"Strict match could not be accepted ({top_votes}/{face_frames} face frames)."
    elif support_ranked:
        support_name, support_count = support_ranked[0]
        reason = (
            f"Supporting match could not be accepted "
            f"({support_count}/{face_frames} face frames for {support_name})."
        )
    else:
        reason = "Faces were found, but none matched a known person safely."
    return decision("unknown_identity", "", reason)


def _identity_matrix(identity_db: sort_photos.IdentityDB) -> tuple[list[str], np.ndarray]:
    names = [
        name for name in sorted(identity_db.identities, key=str.casefold)
        if identity_db.source_counts.get(name, 0) >= MIN_IDENTITY_REFERENCE_FACES
    ]
    if not names:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack([identity_db.identities[name] for name in names]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return names, matrix / np.maximum(norms, 1e-9)


def _match_embedding(
    embedding: np.ndarray,
    names: list[str],
    identity_matrix: np.ndarray,
) -> str | None:
    name, tier, _distance, _margin = _match_embedding_evidence(
        embedding, names, identity_matrix,
    )
    return name if tier == "strict" else None


def _match_embedding_evidence(
    embedding: np.ndarray,
    names: list[str],
    identity_matrix: np.ndarray,
) -> tuple[str | None, str, float, float]:
    if not names or identity_matrix.size == 0:
        return None, "none", 1.0, 0.0
    emb = np.asarray(embedding, dtype=np.float32)
    emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
    similarities = identity_matrix @ emb
    order = np.argsort(-similarities)
    best_index = int(order[0])
    best_distance = 1.0 - float(similarities[best_index])
    if len(order) > 1:
        second_distance = 1.0 - float(similarities[int(order[1])])
        margin = second_distance - best_distance
    else:
        margin = 1.0
    name = names[best_index]
    if best_distance <= MATCH_MAX_DISTANCE and margin >= MATCH_MIN_MARGIN:
        return name, "strict", best_distance, margin
    if best_distance <= SUPPORT_MAX_DISTANCE and margin >= SUPPORT_MIN_MARGIN:
        return name, "support", best_distance, margin
    return None, "none", best_distance, margin


def _face_crop_jpeg(frame: np.ndarray, face) -> bytes:
    bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
    if bbox.size < 4:
        return b""
    height, width = frame.shape[:2]
    x1 = max(0, int(np.floor(bbox[0])))
    y1 = max(0, int(np.floor(bbox[1])))
    x2 = min(width, int(np.ceil(bbox[2])))
    y2 = min(height, int(np.ceil(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return b""
    crop = frame[y1:y2, x1:x2]
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes() if ok else b""


def _build_yunet_detector(app):
    if not YUNET_MODEL.is_file() or not hasattr(app, "models"):
        return None
    if app.models.get("recognition") is None:
        return None
    try:
        return cv2.FaceDetectorYN_create(
            str(YUNET_MODEL), "", (320, 320), YUNET_MIN_SCORE, 0.3, 5_000,
        )
    except (AttributeError, cv2.error):
        return None


def _fallback_faces(frame: np.ndarray, app, detector) -> list[SimpleNamespace]:
    if detector is None:
        return []
    height, width = frame.shape[:2]
    try:
        detector.setInputSize((width, height))
        _retval, detections = detector.detect(frame)
    except cv2.error:
        return []
    if detections is None:
        return []
    recognition = app.models.get("recognition")
    if recognition is None:
        return []

    recovered: list[SimpleNamespace] = []
    for row in detections:
        score = float(row[14])
        x, y, box_width, box_height = (float(value) for value in row[:4])
        if score < YUNET_MIN_SCORE or min(box_width, box_height) < MIN_FACE_PIXELS:
            continue
        landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        try:
            aligned = face_align.norm_crop(frame, landmarks, image_size=112)
            features = recognition.get_feat([aligned])
        except Exception:
            continue
        if features is None or len(features) == 0:
            continue
        embedding = np.asarray(features[0], dtype=np.float32)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-9)
        recovered.append(SimpleNamespace(
            det_score=score,
            bbox=np.asarray([x, y, x + box_width, y + box_height], dtype=np.float32),
            normed_embedding=embedding,
            kps=landmarks,
            video_fallback=True,
        ))
    return recovered


def _usable_faces(frame: np.ndarray, app, fallback_detector) -> tuple[list, bool]:
    detected = app.get(frame)
    usable = []
    for face in detected:
        score = float(getattr(face, "det_score", 0.0))
        bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32)
        embedding = getattr(face, "normed_embedding", None)
        if score < MIN_DETECTION_SCORE or bbox.size < 4 or embedding is None:
            continue
        if min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])) < MIN_FACE_PIXELS:
            continue
        usable.append(face)
    if usable:
        return usable, False
    fallback = _fallback_faces(frame, app, fallback_detector)
    return fallback, bool(fallback)


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= MAX_FRAME_EDGE:
        return frame
    scale = MAX_FRAME_EDGE / float(longest)
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def analyze_video(
    path: Path,
    app,
    names: list[str],
    identity_matrix: np.ndarray,
    *,
    max_samples: int = MAX_SAMPLES,
    fallback_detector=None,
    identity_matcher: VideoIdentityMatcher | None = None,
) -> VideoDecision:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return decide_person(
            Counter(), sampled_frames=0, face_frames=0,
            detected_faces=0, confident_matches=0,
        )

    frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    initial_limit = min(max_samples, INITIAL_MAX_SAMPLES)
    desired_samples = min(initial_limit, max(MIN_SAMPLES, int(duration / 3.0) + 2))
    indices = sample_frame_indices(frame_count, desired_samples)
    if not indices:
        indices = list(range(min(max_samples, INITIAL_MAX_SAMPLES)))

    votes: Counter[str] = Counter()
    support_votes: Counter[str] = Counter()
    sampled_frames = 0
    face_frames = 0
    detected_faces = 0
    confident_matches = 0
    fallback_face_frames = 0
    analysis_passes = 1
    if fallback_detector is None:
        fallback_detector = _build_yunet_detector(app)

    def process_indices(frame_indices: list[int]) -> None:
        nonlocal sampled_frames, face_frames, detected_faces
        nonlocal confident_matches, fallback_face_frames
        for frame_index in frame_indices:
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                if frame_count <= 0:
                    break
                continue
            sampled_frames += 1
            frame = _resize_frame(frame)
            faces, used_fallback = _usable_faces(frame, app, fallback_detector)
            if not faces:
                continue
            face_frames += 1
            if used_fallback:
                fallback_face_frames += 1
            detected_faces += len(faces)
            frame_people: set[str] = set()
            frame_support_people: set[str] = set()
            for face in faces:
                embedding = getattr(face, "normed_embedding", None)
                if embedding is None:
                    continue
                if identity_matcher is not None:
                    name, tier, _distance, _margin = identity_matcher.match(
                        frame, face, path
                    )
                else:
                    name, tier, _distance, _margin = _match_embedding_evidence(
                        embedding, names, identity_matrix,
                    )
                if name is None:
                    continue
                frame_support_people.add(name)
                if tier == "strict":
                    confident_matches += 1
                    frame_people.add(name)
            votes.update(frame_people)
            support_votes.update(frame_support_people)

    try:
        process_indices(indices)
        initial_decision = decide_person(
            votes,
            support_votes=support_votes,
            sampled_frames=sampled_frames,
            face_frames=face_frames,
            detected_faces=detected_faces,
            confident_matches=confident_matches,
        )
        if (
            initial_decision.status in {"unknown_identity", "no_usable_face"}
            and frame_count > 0
            and max_samples > len(indices)
        ):
            target_count = min(
                max_samples,
                max(RECOVERY_MIN_SAMPLES, len(indices) * 2),
            )
            recovery_indices = sample_frame_indices(frame_count, target_count)
            already_sampled = set(indices)
            extra_indices = [index for index in recovery_indices if index not in already_sampled]
            extra_indices = extra_indices[:max(0, target_count - len(indices))]
            if extra_indices:
                analysis_passes = 2
                process_indices(extra_indices)
    finally:
        capture.release()

    return replace(decide_person(
        votes,
        support_votes=support_votes,
        sampled_frames=sampled_frames,
        face_frames=face_frames,
        detected_faces=detected_faces,
        confident_matches=confident_matches,
    ), fallback_face_frames=fallback_face_frames, analysis_passes=analysis_passes)


def _review_bucket(status: str) -> str:
    return {
        "multiple_people": "multiple_people",
        "unknown_identity": "unknown_identity",
        "no_usable_face": "no_usable_face",
        "unreadable_video": "unreadable_video",
        "insufficient_space": "insufficient_space",
    }.get(status, "processing_failed")


def destination_for(
    source: Path,
    decision: VideoDecision,
    *,
    people_dir: Path,
    review_root: Path,
) -> Path:
    if decision.status == "matched" and decision.person:
        return people_dir / decision.person / "videos" / source.name
    return review_root / _review_bucket(decision.status) / source.name


def _has_internal_capacity(destination: Path, source_size: int) -> bool:
    ancestor = destination.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    return usage.free - source_size >= MIN_INTERNAL_FREE_BYTES


def should_pause_for_space(
    decision: VideoDecision,
    destination: Path,
    source_size: int,
    *,
    capacity_check=_has_internal_capacity,
) -> bool:
    return decision.status == "matched" and not capacity_check(destination, source_size)


def _available_memory_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except (ImportError, OSError):
        return None


def _release_native_memory() -> None:
    gc.collect()
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        relief = library.malloc_zone_pressure_relief
        relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        relief.restype = ctypes.c_size_t
        relief(None, 0)
    except (AttributeError, OSError):
        pass


def _report_path(report_dir: Path) -> Path:
    return report_dir / f"video_sort_{time.strftime('%Y%m%d_%H%M%S')}.json"


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def _append_progress(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=pipeline_paths.TO_PROCESS)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--dry-run", action="store_true", help="Analyze without moving videos.")
    parser.add_argument("--max-videos", type=int, default=0, help="Process at most this many videos.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument(
        "--file-list", type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--review-only", action="store_true",
        help="Recheck prior unknown/no-face videos and move after one recognized frame.",
    )
    parser.add_argument(
        "--no-legacy-inbox",
        action="store_true",
        help="Do not also classify videos left in the old ~/Pictures/videos holding folder.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    sorted_root = args.output.expanduser().resolve()
    people_dir = sorted_root / "photos_by_person"
    source_review = sorted_root / "_source_review"
    review_root = source_review / "unassigned_intake" / "videos"
    report_dir = source_review / "video_reports"

    if args.file_list:
        try:
            listed = json.loads(args.file_list.read_text(encoding="utf-8"))
            videos = [Path(path).expanduser().resolve() for path in listed]
        except (OSError, ValueError, TypeError) as exc:
            print(f"ERROR: invalid video worker file list: {exc}")
            return 2
        videos = [path for path in videos if path.is_file() and path.suffix.lower() in sort_photos.VIDEO_EXTS]
        input_dirs = [args.file_list]
    elif not args.review_only and not input_dir.exists():
        print(f"ERROR: video intake folder is unavailable: {input_dir}")
        return 2
    else:
        if args.review_only:
            input_dirs = [
                review_root / "unknown_identity",
                review_root / "no_usable_face",
            ]
        else:
            input_dirs = [input_dir]
            legacy_input = LEGACY_VIDEO_INBOX.expanduser().resolve()
            if not args.no_legacy_inbox and legacy_input.exists() and legacy_input != input_dir:
                input_dirs.append(legacy_input)
        videos_by_path: dict[str, Path] = {}
        for scan_root in input_dirs:
            if not scan_root.exists():
                continue
            for video in sort_photos.iter_videos(scan_root):
                videos_by_path[str(video.resolve())] = video
        videos = sorted(videos_by_path.values(), key=lambda path: str(path).casefold())
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    if not videos:
        if not args.quiet:
            print("No videos found in: " + ", ".join(str(path) for path in input_dirs))
        return 0

    identity_db = sort_photos.load_identity_db()
    if identity_db is None or not identity_db.identities:
        print("ERROR: known-person identity DB is missing. Run the Face Terminal identity rebuild first.")
        return 3
    names, identity_matrix = _identity_matrix(identity_db)
    if not names:
        print("ERROR: identity DB has no people with enough reference faces for safe video matching.")
        return 3

    print(f"Video intake:       {', '.join(str(path) for path in input_dirs)}")
    print(f"Videos to analyze:  {len(videos)}")
    print(f"Known identities:   {len(names)}")
    hard_negatives = identity_hard_negatives.vectors_by_person(
        sort_photos.IDENTITY_HARD_NEGATIVES_FILE
    )
    secondary_db = secondary_identity_matcher.load()
    secondary = (
        secondary_identity_matcher.SecondaryMatcher(secondary_db)
        if secondary_db is not None
        and secondary_db.primary_signature
        == secondary_identity_matcher.primary_signature(identity_db)
        else None
    )
    identity_matcher = VideoIdentityMatcher(identity_db, hard_negatives, secondary)
    print(f"Hard-negative guards:{sum(map(len, hard_negatives.values())):>5}")
    print(f"Secondary verifier: {'ready' if secondary else 'not built (safe skip)'}")
    if args.review_only:
        mode = "RECHECK REVIEW" if not args.dry_run else "DRY RUN REVIEW RECHECK"
    else:
        mode = "DRY RUN" if args.dry_run else "MOVE AFTER ONE RECOGNIZED FRAME"
    print(f"Mode:               {mode}")
    print("Initializing face detector...")
    app = sort_photos._build_app()
    fallback_detector = _build_yunet_detector(app)
    print(f"Fallback detector:  {'YuNet ready' if fallback_detector is not None else 'unavailable'}")

    records: list[dict] = []
    summary: Counter[str] = Counter()
    paused_for_space = False
    paused_for_memory = False
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = _report_path(report_dir)
    progress_path = report_path.with_suffix(".progress.jsonl")
    for index, source in enumerate(videos, start=1):
        available_mb = _available_memory_mb()
        if available_mb is not None and available_mb < MIN_RUNTIME_MEMORY_MB:
            print(
                f"    Pausing safely before {source.name}: only {available_mb} MB memory is available."
            )
            paused_for_memory = True
            break
        print(f"[{index}/{len(videos)}] Analyzing {source.name}", flush=True)
        decision = analyze_video(
            source, app, names, identity_matrix,
            max_samples=max(MIN_SAMPLES, min(args.max_samples, MAX_SAMPLES)),
            fallback_detector=fallback_detector,
            identity_matcher=identity_matcher,
        )
        destination = destination_for(
            source, decision, people_dir=people_dir, review_root=review_root,
        )

        if decision.status == "matched" and not (people_dir / decision.person).is_dir():
            decision = replace(
                decision,
                status="unknown_identity",
                person="",
                reason="Matched person folder no longer exists.",
            )
            destination = destination_for(
                source, decision, people_dir=people_dir, review_root=review_root,
            )
        if should_pause_for_space(decision, destination, source.stat().st_size):
            decision = replace(
                decision,
                status="insufficient_space",
                reason="Paused before moving this video because the Mac would fall below the 20 GB safety reserve.",
            )
            summary[decision.status] += 1
            records.append({
                "source": str(source),
                "destination": "",
                "moved_to": "",
                **asdict(decision),
            })
            _append_progress(progress_path, records[-1])
            print(f"    {decision.status}: {decision.reason}")
            print(f"    Left in place for resume: {source}")
            paused_for_space = True
            break

        leave_in_place = source.resolve() == destination.resolve(strict=False)
        if not leave_in_place:
            destination = sort_photos.unique_path(destination)
        moved_to = ""
        if not args.dry_run and not leave_in_place:
            moved = operation_ledger.move_path(
                source,
                destination,
                sorted_root=sorted_root,
                operation="sort_videos.classify",
                reason="classify intake video using one-or-more recognized sampled frames",
                extra={
                    "decision": decision.status,
                    "person": decision.person,
                    "votes": decision.votes,
                    "sampled_frames": decision.sampled_frames,
                    "face_frames": decision.face_frames,
                },
            )
            moved_to = str(moved)
        elif leave_in_place:
            moved_to = str(source)
        summary[decision.status] += 1
        print(f"    {decision.status}: {decision.reason}")
        if leave_in_place:
            print(f"    Remains in review: {source}")
        else:
            print(f"    {'Would move' if args.dry_run else 'Moved'} to: {destination}")
        records.append({
            "source": str(source),
            "destination": str(destination),
            "moved_to": moved_to,
            **asdict(decision),
        })
        _append_progress(progress_path, records[-1])
        if index % 5 == 0:
            _release_native_memory()

    payload = {
        "version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": bool(args.dry_run),
        "review_only": bool(args.review_only),
        "complete": not paused_for_space and not paused_for_memory,
        "inputs": [str(path) for path in input_dirs],
        "summary": dict(sorted(summary.items())),
        "records": records,
    }
    _write_report(report_path, payload)
    progress_path.unlink(missing_ok=True)

    print()
    print("Video Sort Summary")
    print("=" * 60)
    for status, count in sorted(summary.items()):
        print(f"{status:22} {count}")
    print(f"Report: {report_path}")
    identity_matcher.flush()
    if paused_for_space:
        print("Resume after freeing space with: python face.py daily --resume")
        return 4
    if paused_for_memory:
        print("Resume after closing memory-heavy apps with: python face.py daily --resume")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
