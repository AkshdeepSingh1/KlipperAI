"""
Smart Crop Service
Intelligent face-tracking auto-framing for vertical (9:16) video clips.

This service replaces the naive center-crop with dynamic subject tracking
that follows the main face across the video — similar to TikTok / Instagram
Reels auto-framing.

Pipeline:
    clip → SmartCropEngine (Detectors -> Optimizer) → render_dynamic_crop → export

Follows SRP: this module handles ONLY smart framing logic.
All other editing tasks (cutting, subtitles) remain in video_editing_service.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.Crop import Crop

from src.shared.core.logger import get_logger
from src.worker.services.smart_crop.detectors.face_detector import FaceDetector
from src.worker.services.smart_crop.detectors.text_detector import TextDetector
from src.worker.services.smart_crop.smart_crop_engine import SmartCropEngine

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────

FRAME_SAMPLE_INTERVAL = 0.5  # seconds between sampled frames
MIN_CLIP_DURATION = 1.0      # seconds — clips shorter than this skip analysis


# ── Step 5: Applying Dynamic Crop ───────────────────────────────────

from PIL import Image

def render_dynamic_crop(
    clip: VideoFileClip,
    crop_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    target_w: int,
    target_h: int,
    output_path: str,
) -> str:
    """
    Apply a time-varying crop/pad to the clip and export the vertical video.

    Interpolates the crop box for each frame, crops it, scales it to fit
    the 9:16 target, and pads with black if necessary.

    Args:
        clip: The source VideoFileClip.
        crop_boxes: Smoothed (timestamp, (x1, y1, x2, y2)) tuples.
        target_w: Desired output width.
        target_h: Desired output height.
        output_path: Where to write the result.

    Returns:
        The output_path on success.
    """

    # Build lookups for interpolation
    timestamps = np.array([t for t, _ in crop_boxes], dtype=np.float64)
    # Extract coords columns: x1, y1, x2, y2
    coords = np.array([box for _, box in crop_boxes], dtype=np.float64) # Shape (N, 4)

    def _get_crop_box(t: float) -> Tuple[int, int, int, int]:
        """Interpolate (x1, y1, x2, y2) for an arbitrary time t."""
        # Interpolate each component independently
        c = np.zeros(4, dtype=int)
        for i in range(4):
            c[i] = int(np.interp(t, timestamps, coords[:, i]))
        return tuple(c)

    def _process_frame(get_frame, t):
        """Per-frame transform: crop -> resize -> pad."""
        frame = get_frame(t) # Numpy array (H, W, 3)
        fh, fw = frame.shape[:2]
        
        x1, y1, x2, y2 = _get_crop_box(t)
        
        # Clamp coordinates to frame boundaries
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(x1 + 1, min(fw, x2))
        y2 = max(y1 + 1, min(fh, y2))
        
        # Crop
        cropped_img = frame[y1:y2, x1:x2]
        
        # If crop is empty (shouldn't happen due to clamping), return black
        if cropped_img.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Resize and Pad using PIL
        pil_img = Image.fromarray(cropped_img)
        cw, ch = pil_img.size
        
        # Calculate resize dimensions to FIT inside target_w x target_h
        target_ratio = target_w / target_h
        current_ratio = cw / ch
        
        if current_ratio > target_ratio:
            # Wider than target -> Fit width
            new_w = target_w
            new_h = int(target_w / current_ratio)
        else:
            # Taller -> Fit height
            new_h = target_h
            new_w = int(target_h * current_ratio)
            
        # Resize (LANCZOS for quality)
        resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        
        # Create black background
        bg = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        
        # Paste centered
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        bg.paste(resized_pil, (paste_x, paste_y))
        
        return np.array(bg)

    cropped = clip.transform(_process_frame)
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




def smart_crop_clip(input_path: str) -> str:
    """
    Main orchestrator — smart-crop a single clip to 9:16 vertical format.

    Uses SmartCropEngine with FaceDetector and TextDetector.

    Edge cases handled:
        • Clip shorter than MIN_CLIP_DURATION → falls back to center crop.
        • No interesting content detected → falls back to center crop.
        • Clip already narrower than 9:16 → skipped.

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

        # Initialize Smart Crop Engine
        detectors = [
            FaceDetector(),
            TextDetector(backend="east")  # Use OpenCV EAST by default
        ]
        engine = SmartCropEngine(detectors, frame_sample_interval=FRAME_SAMPLE_INTERVAL)

        # Run Analysis
        crop_boxes = engine.process_video(clip)

        # Render with Dynamic Layout
        return render_dynamic_crop(clip, crop_boxes, 1080, 1920, str(temp_output))

    except Exception as e:
        logger.error(f"Smart crop failed: {e}", exc_info=True)
        # Fallback to simple center crop if engine fails
        if clip:
            return _center_crop_fallback(clip, w, h, target_w, target_h, input_path, temp_output)
        return input_path
    finally:
        if clip:
            clip.close()
        # Rename temp to original
        if os.path.exists(temp_output):
            if os.path.exists(input_path):
                 os.remove(input_path)
            os.rename(temp_output, input_path)
            logger.info(f"Replaced original with smart-cropped clip: {input_path}")

    return str(input_path)



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
