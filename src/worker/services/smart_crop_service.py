"""
Smart Crop Service
Intelligent face-tracking auto-framing for vertical (9:16) video clips.

This service replaces the naive center-crop with dynamic subject tracking
that follows the main face across the video — similar to TikTok / Instagram
Reels auto-framing.

Pipeline:
    clip → analyze_frames → detect_faces → compute_crop_positions
         → smooth_crop_positions → render_dynamic_crop → export

Follows SRP: this module handles ONLY smart framing logic.
All other editing tasks (cutting, subtitles) remain in video_editing_service.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from mediapipe import Image, ImageFormat
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.shared.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────

FRAME_SAMPLE_INTERVAL = 0.5  # seconds between sampled frames
SMOOTHING_WINDOW_SIZE = 5    # number of positions for moving-average
MIN_CLIP_DURATION = 1.0      # seconds — clips shorter than this skip analysis
FACE_DETECTION_CONFIDENCE = 0.3


# ── Step 1: Frame Analysis (How We Detect Subject) ──────────────────

def analyze_frames(
    clip: VideoFileClip,
    interval: float = FRAME_SAMPLE_INTERVAL,
) -> List[Tuple[float, np.ndarray]]:
    """
    Sample frames from the clip at fixed intervals.

    Args:
        clip: An open MoviePy VideoFileClip.
        interval: Seconds between each sampled frame.

    Returns:
        List of (timestamp, frame_as_numpy_array) tuples.
    """
    frames: List[Tuple[float, np.ndarray]] = []
    duration = clip.duration

    t = 0.0
    while t < duration:
        frame = clip.get_frame(t)
        frames.append((t, frame))
        t += interval

    logger.info(
        f"Analyzed {len(frames)} frames over {duration:.1f}s "
        f"(interval={interval}s)"
    )
    return frames


# ── Step 2: Face Detection ──────────────────────────────────────────

def detect_faces(
    frames: List[Tuple[float, np.ndarray]],
) -> List[Tuple[float, Optional[Tuple[float, float, float, float]]]]:
    """
    Run MediaPipe Face Detection on each sampled frame.

    For each frame returns the bounding box of the **largest** detected face,
    or None if no face is found.

    Args:
        frames: Output of analyze_frames — list of (timestamp, frame) tuples.

    Returns:
        List of (timestamp, bbox_or_None) where bbox is
        (x_center_ratio, y_center_ratio, width_ratio, height_ratio)
        in normalised [0, 1] coordinates relative to the frame size.
    """
    detections: List[Tuple[float, Optional[Tuple[float, float, float, float]]]] = []

    with FaceDetector.create_from_options(
        FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path='src/worker/services/models/face_detector.tflite'),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
        )
    ) as face_detector:

        for timestamp, frame in frames:
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame)
            detection_result = face_detector.detect(mp_image)
            print(detection_result)
            if not detection_result.detections:
                detections.append((timestamp, None))
                continue

            best = max(
                detection_result.detections,
                key=lambda d: (
                    d.bounding_box.width
                    * d.bounding_box.height
                ),
            )
            bb = best.bounding_box
            x_center = bb.origin_x + bb.width / 2.0
            y_center = bb.origin_y + bb.height / 2.0
            detections.append((timestamp, (x_center, y_center, bb.width, bb.height)))

    faces_found = sum(1 for _, d in detections if d is not None) #generator expression
    logger.info(
        f"Face detection complete: {faces_found}/{len(detections)} frames "
        f"have a detected face"
    )
    return detections


# ── Step 3: Computing Crop Positions ────────────────────────────────

def compute_crop_positions(
    detections: List[Tuple[float, Optional[Tuple[float, float, float, float]]]],
    video_w: int,
    video_h: int,
) -> List[Tuple[float, int]]:
    """
    Convert face-center positions into horizontal crop-window positions.

    Args:
        detections: Output of detect_faces.
        video_w: Original video width in pixels.
        video_h: Original video height in pixels.

    Returns:
        List of (timestamp, crop_x) pairs where crop_x is the left edge
        of the crop window in pixels.
    """
    target_width = int(video_h * 9 / 16)
    center_crop_x = (video_w - target_width) // 2

    positions: List[Tuple[float, int]] = []
    last_known_x: int = center_crop_x  # fallback if face disappears

    for timestamp, bbox in detections:
        if bbox is None:
            # No face detected → reuse last known position
            crop_x = last_known_x
        else:
            face_center_px = int(bbox[0])
            crop_x = face_center_px - target_width // 2
            # Clamp within video bounds
            crop_x = max(0, min(video_w - target_width, crop_x))
            last_known_x = crop_x

        positions.append((timestamp, crop_x))

    logger.info(f"Computed {len(positions)} crop positions (target_width={target_width}px)")
    return positions


# ── Step 4: Smoothing Movement ──────────────────────────────────────

def smooth_crop_positions(
    positions: List[Tuple[float, int]],
    window_size: int = SMOOTHING_WINDOW_SIZE,
) -> List[Tuple[float, int]]:
    """
    Apply a moving-average filter to crop positions to eliminate jitter
    and produce a smooth camera pan effect.

    Args:
        positions: Output of compute_crop_positions.
        window_size: Number of neighbours to average over.

    Returns:
        Smoothed list of (timestamp, crop_x) pairs.
    """
    if len(positions) <= 1:
        return positions

    crop_values = [x for _, x in positions]
    smoothed_values: List[int] = []

    half = window_size // 2
    for i in range(len(crop_values)):
        start = max(0, i - half)
        end = min(len(crop_values), i + half + 1)
        avg = int(sum(crop_values[start:end]) / (end - start))
        smoothed_values.append(avg)

    smoothed = [
        (positions[i][0], smoothed_values[i])
        for i in range(len(positions))
    ]

    logger.info(f"Smoothed {len(smoothed)} crop positions (window={window_size})")
    return smoothed


# ── Step 5: Applying Dynamic Crop ───────────────────────────────────

def render_dynamic_crop(
    clip: VideoFileClip,
    positions: List[Tuple[float, int]],
    target_w: int,
    target_h: int,
    output_path: str,
) -> str:
    """
    Apply a time-varying crop to the clip and export the vertical video.

    Uses numpy array slicing on each frame for maximum compatibility
    with MoviePy 2.x.

    Args:
        clip: The source VideoFileClip.
        positions: Smoothed (timestamp, crop_x) pairs.
        target_w: Desired output width (= height * 9/16).
        target_h: Desired output height (= original height).
        output_path: Where to write the result.

    Returns:
        The output_path on success.
    """

    # Build a lookup: for a given time t, find the crop_x via interpolation
    timestamps = np.array([t for t, _ in positions], dtype=np.float64)
    crop_xs = np.array([x for _, x in positions], dtype=np.float64)

    def _get_crop_x(t: float) -> int:
        """Interpolate crop_x for an arbitrary time t."""
        x = int(np.interp(t, timestamps, crop_xs))
        return max(0, x)

    def _crop_frame(get_frame, t):
        """Per-frame transform: slice the crop window from the frame."""
        frame = get_frame(t)
        x = _get_crop_x(t)
        return frame[:target_h, x: x + target_w]

    cropped = clip.transform(_crop_frame)
    cropped = cropped.with_duration(clip.duration)

    temp_audio = os.path.splitext(output_path)[0] + "_temp_audio.m4a"
    cropped.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=temp_audio,
        remove_temp=True,
        threads=2,
        preset="medium",
    )

    logger.info(f"Rendered dynamic-crop video → {output_path}")
    return output_path


# ── Step 6: Where This Fits In Your Existing Function ───────────────
# ── Step 7: Edge Cases You Must Handle ──────────────────────────────

def smart_crop_clip(input_path: str) -> str:
    """
    Main orchestrator — smart-crop a single clip to 9:16 vertical format.

    Pipeline:
        analyze_frames → detect_faces → compute_crop_positions
        → smooth_crop_positions → render_dynamic_crop

    Edge cases handled:
        • Clip shorter than MIN_CLIP_DURATION → falls back to center crop.
        • No face detected in ANY frame   → falls back to center crop.
        • Multiple faces in a frame       → largest face is chosen.
        • Face leaves frame mid-clip      → reuses last known position.

    Args:
        input_path: Path to the clip MP4 file (will be replaced in-place).

    Returns:
        The same input_path (file is overwritten with the vertical version).
    """
    clip_file = Path(input_path)
    output_dir = clip_file.parent
    temp_output = output_dir / f"{clip_file.stem}_smart_cropped.mp4"

    logger.info(f"Smart cropping clip: {input_path}")

    clip = None
    try:
        clip = VideoFileClip(input_path)
        w, h = clip.size
        duration = clip.duration

        target_w = int(h * 9 / 16)
        target_h = h

        # ── Edge case: extremely short clip → center crop ──
        if duration < MIN_CLIP_DURATION:
            logger.info(
                f"Clip is very short ({duration:.2f}s < {MIN_CLIP_DURATION}s), "
                "falling back to center crop"
            )
            return _center_crop_fallback(clip, w, h, target_w, target_h, input_path, temp_output)

        # ── Edge case: clip already narrower than 9:16 ──
        if target_w >= w:
            logger.info("Clip is already narrower than 9:16, skipping smart crop")
            clip.close()
            return input_path

        # 1. Analyze frames
        frames = analyze_frames(clip, interval=FRAME_SAMPLE_INTERVAL)

        # 2. Detect faces
        detections = detect_faces(frames)
        print(detections)
        # ── Edge case: no face detected in ANY frame → center crop ──
        has_any_face = any(d is not None for _, d in detections)
        if not has_any_face:
            logger.info("No face detected in any frame, falling back to center crop")
            return _center_crop_fallback(clip, w, h, target_w, target_h, input_path, temp_output)

        # 3. Compute crop positions
        positions = compute_crop_positions(detections, w, h)

        # 4. Smooth positions
        smoothed = smooth_crop_positions(positions)

        # 5. Render dynamic crop
        render_dynamic_crop(clip, smoothed, target_w, target_h, str(temp_output))

    finally:
        if clip is not None:
            clip.close()

    # In-place replacement (same pattern as crop_clips_to_9_16)
    if temp_output.exists():
        clip_file.unlink(missing_ok=True)
        temp_output.rename(clip_file)
        logger.info(f"Replaced original with smart-cropped clip: {clip_file}")
    else:
        raise RuntimeError(f"Smart-cropped file was not created at {temp_output}")

    return str(clip_file)


# ── Internal helper ─────────────────────────────────────────────────

def _center_crop_fallback(
    clip: VideoFileClip,
    w: int,
    h: int,
    target_w: int,
    target_h: int,
    input_path: str,
    temp_output: Path,
) -> str:
    """
    Simple center crop — used as a fallback when face detection
    is not applicable (no faces, very short clips, etc.).
    """
    from moviepy.video.fx.Crop import Crop

    x_center = w // 2
    x1 = max(0, x_center - target_w // 2)
    x2 = min(w, x1 + target_w)

    cropped = clip.with_effects([Crop(x1=x1, y1=0, x2=x2, y2=h)])
    temp_audio = os.path.splitext(str(temp_output))[0] + "_temp_audio.m4a"
    cropped.write_videofile(
        str(temp_output),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=temp_audio,
        remove_temp=True,
    )
    cropped.close()
    clip.close()

    clip_file = Path(input_path)
    if temp_output.exists():
        clip_file.unlink(missing_ok=True)
        temp_output.rename(clip_file)
        logger.info(f"Center-crop fallback applied: {clip_file}")

    return str(clip_file)
