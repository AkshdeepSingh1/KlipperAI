"""
Video Editing Service
All video manipulation — cutting clips, cropping to aspect ratios, adding subtitles.
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, List

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.Crop import Crop

from src.shared.core.logger import get_logger

logger = get_logger(__name__)

# ── Subtitle constants ──────────────────────────────────────────────
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_SIZE_MIN = 14
FONT_SIZE_MAX = 42
FONT_SIZE_RATIO = 0.045
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "yellow"
STROKE_COLOR = "black"
STROKE_WIDTH = 3
BOTTOM_MARGIN = 120
MAX_WIDTH_RATIO = 0.8


# ── Clip cutting ─────────────────────────────────────────────────────

def _cut_single_clip(args) -> str:
    """Cut a single clip from a video. Internal helper."""
    video_path, start, end, output_path = args

    if start is None or end is None:
        raise ValueError(f"Missing timestamps: start={start}, end={end}")
    # AssemblyAI uses milliseconds; MoviePy expects seconds
    start = start / 1000.0
    end = end / 1000.0

    if end <= start:
        raise ValueError(f"Invalid timestamps: {start} -> {end}")
    temp_audio = os.path.splitext(output_path)[0] + "_temp.m4a"

    with VideoFileClip(video_path) as video:
        duration = video.duration
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end <= start:
            raise ValueError(
                f"Clip segment [{start}, {end}] is empty or invalid for duration {duration}"
            )
        clip = video.subclipped(start, end)

        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio,
            remove_temp=True,
            threads=2,
            preset="medium",
            bitrate="5000k",
        )

    return output_path


def cut_clips(user_id: str, video_id: str, timestamps: Any) -> List[str]:
    """
    Cut subclips from a video based on timestamps.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction
        timestamps: List of timestamp dicts or path to JSON file

    Returns:
        List of file paths to the generated clips
    """
    if isinstance(timestamps, str):
        with open(timestamps, "r", encoding="utf-8") as f:
            timestamps = json.load(f)
    if isinstance(timestamps, dict):
        flat_timestamps = [occ for occs in timestamps.values() for occ in occs]
    else:
        flat_timestamps = list(timestamps)

    video_path = os.path.join("downloads", user_id, video_id, "video.mp4")
    output_dir = os.path.join("downloads", user_id, video_id, "clips")
    os.makedirs(output_dir, exist_ok=True)

    jobs = []
    for i, ts in enumerate(flat_timestamps):
        start = ts.get("start_timestamp") or ts.get("start")
        end = ts.get("end_timestamp") or ts.get("end")
        output_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
        jobs.append((video_path, start, end, output_path))

    # Run sequentially; parallel FFmpeg workers often cause BrokenPipe / temp file conflicts
    results = [_cut_single_clip(job) for job in jobs]
    return results


def get_or_cut_clips(user_id: str, video_id: str) -> List[str]:
    """
    Checks for existing clips; if not found, cuts new clips from the video.

    Args:
        user_id: User identifier for path construction.
        video_id: Video identifier for path construction.

    Returns:
        List of file paths to the generated or existing clips.
    """
    clips_output_dir = os.path.join("downloads", user_id, video_id, "clips")
    existing_clips = [
        os.path.join(clips_output_dir, f)
        for f in os.listdir(clips_output_dir)
        if f.endswith(".mp4")
    ] if os.path.exists(clips_output_dir) else []

    if existing_clips:
        logger.info("Clips already exist, skipping cutting.")
        return existing_clips
    else:
        clips_timestamps_path = os.path.join(
            "downloads", user_id, video_id, "clips_timestamps.json"
        )
        clip_path_list = cut_clips(user_id, video_id, clips_timestamps_path)
        if not clip_path_list:
            raise RuntimeError("Failed to cut clips from video")
        return clip_path_list


# ── Crop to 9:16 ─────────────────────────────────────────────────────

def crop_clips_to_9_16(clips: List[str]) -> List[str]:
    """
    Crop video clips to 9:16 aspect ratio (vertical format).

    Writes the cropped file in-place (replaces the original) so that
    downstream upload paths remain unchanged.

    Args:
        clips: List of clip file paths

    Returns:
        List of cropped clip file paths (same paths as input, overwritten)
    """
    cropped_clips = []

    for i, clip_path in enumerate(clips):
        clip_file = Path(clip_path)
        output_dir = clip_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_cropped_path = output_dir / f"{clip_file.stem}_cropped.mp4"
        temp_audio_path = output_dir / f"temp_audio_{i}.m4a"

        logger.info(f"Cropping clip {i+1}/{len(clips)} to 9:16 aspect ratio: {clip_path}")

        clip = None
        cropped_clip = None
        try:
            clip = VideoFileClip(clip_path)

            w, h = clip.size
            logger.info(f"Original clip dimensions: {w}x{h}")

            target_width = int(h * 9 / 16)

            if target_width > w:
                # Source is narrower than 9:16 → crop height to fit
                target_height = int(w * 16 / 9)
                y_center = h // 2
                y1 = max(0, y_center - target_height // 2)
                y2 = min(h, y1 + target_height)
                cropped_clip = clip.with_effects([Crop(x1=0, y1=y1, x2=w, y2=y2)])
                logger.info(f"Cropped height to {y2-y1}px → {w}x{y2-y1}")
            else:
                # Standard landscape → crop width to 9:16
                x_center = w // 2
                x1 = max(0, x_center - target_width // 2)
                x2 = min(w, x1 + target_width)
                cropped_clip = clip.with_effects([Crop(x1=x1, y1=0, x2=x2, y2=h)])
                logger.info(f"Cropped width to {x2-x1}px → {x2-x1}x{h}")

            cropped_clip.write_videofile(
                str(temp_cropped_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(temp_audio_path),
                remove_temp=True,
            )

        finally:
            if cropped_clip is not None:
                cropped_clip.close()
            if clip is not None:
                clip.close()

        if temp_cropped_path.exists():
            clip_file.unlink(missing_ok=True)
            temp_cropped_path.rename(clip_file)
            logger.info(f"Replaced original with cropped clip: {clip_file}")
        else:
            raise RuntimeError(
                f"Cropped file was not created at {temp_cropped_path}"
            )

        cropped_clips.append(str(clip_file))

    logger.info(f"Cropped {len(cropped_clips)} clips to 9:16 aspect ratio")
    return cropped_clips


# ── Subtitles ────────────────────────────────────────────────────────

def _font_size_for_video(video_w: int, video_h: int) -> int:
    """Compute subtitle font size from video dimensions (shorter side)."""
    short_side = min(video_w, video_h)
    size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(short_side * FONT_SIZE_RATIO)))
    return size


def _create_caption_clip(
    text: str,
    video_w: int,
    video_h: int,
    start: float,
    end: float,
    single_line: bool = True,
):
    """Create a subtitle clip. If single_line is True, use full width so text stays on one line."""
    font_size = _font_size_for_video(video_w, video_h)
    width = video_w if single_line else int(video_w * MAX_WIDTH_RATIO)

    txt_clip = TextClip(
        text=text,
        font=FONT,
        font_size=font_size,
        color=TEXT_COLOR,
        stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH,
        method="caption",
        size=(width, None),
        text_align="center",
    )

    anchor_y = int(video_h * 0.75)
    pos_y = min(anchor_y, video_h - txt_clip.h)
    pos_y = max(0, pos_y)

    txt_clip = txt_clip.with_position(("center", pos_y))

    caption = (
        CompositeVideoClip([txt_clip], size=(video_w, video_h))
        .with_start(start)
        .with_end(end)
    )

    return caption


def _word_ends_sentence(word: Dict[str, Any]) -> bool:
    """True if this word ends a sentence (line boundary)."""
    text = (word.get("text") or "").strip()
    return any(text.endswith(p) for p in (".", "?", "!"))


def _chunk_words_for_subtitles(
    words: List[Dict[str, Any]],
    min_words: int = 3,
    max_words: int = 5,
) -> List[List[Dict[str, Any]]]:
    """
    Chunk words into segments of 3–5 words for subtitles.
    Never spans across sentence boundaries.
    """
    if not words:
        return []
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for w in words:
        current.append(w)
        ends_sentence = _word_ends_sentence(w)
        at_max = len(current) >= max_words
        if ends_sentence or at_max:
            if current:
                segments.append(current)
                current = []

    if current:
        segments.append(current)
    return segments


def add_subtitles(
    user_id: str, video_id: str, highlight_words: bool = False
) -> None:
    """
    Burn subtitles onto video clips.

    Reads clip timestamps and corresponding clip files, renders word-level
    captions, and writes subtitled versions to a separate folder.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction
        highlight_words: Whether to add word-level highlight overlay
    """
    clips_timestamps_path = os.path.join(
        "downloads", user_id, video_id, "clips_timestamps.json"
    )
    with open(clips_timestamps_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)
    clips_folder_path = os.path.join("downloads", user_id, video_id, "clips")
    output_folder = os.path.join("downloads", user_id, video_id, "clips_with_subtitles")
    os.makedirs(output_folder, exist_ok=True)

    clip_files = sorted(
        [f for f in os.listdir(clips_folder_path) if f.endswith(".mp4")]
    )

    if len(clip_files) != len(clips_data):
        raise ValueError("Mismatch: number of clips ≠ timestamp entries")

    for idx, (clip_file, clip_data) in enumerate(zip(clip_files, clips_data)):
        video_path = os.path.join(clips_folder_path, clip_file)
        logger.info(f"Processing subtitles for: {clip_file}")
        video = VideoFileClip(video_path)
        video_w, video_h = video.size
        caption_clips = []

        clip_start_ms = clip_data["start"]
        segments = _chunk_words_for_subtitles(clip_data["words"])

        for seg in segments:
            text = " ".join(w["text"] for w in seg)
            start_ms = seg[0]["start"]
            end_ms = seg[-1]["end"]
            start_sec = (start_ms - clip_start_ms) / 1000.0
            end_sec = (end_ms - clip_start_ms) / 1000.0
            start_sec = max(0.0, min(start_sec, video.duration))
            end_sec = max(0.0, min(end_sec, video.duration))
            if end_sec <= start_sec:
                continue
            caption_clips.append(
                _create_caption_clip(
                    text, video_w, video_h, start=start_sec, end=end_sec
                )
            )

        if highlight_words:
            font_size = _font_size_for_video(video_w, video_h)
            for w in clip_data["words"]:
                word_text = w["text"]
                word_clip = TextClip(
                    text=word_text,
                    font=FONT,
                    font_size=font_size,
                    color=HIGHLIGHT_COLOR,
                    stroke_color=STROKE_COLOR,
                    stroke_width=STROKE_WIDTH,
                )
                start_sec = (w["start"] - clip_start_ms) / 1000.0
                end_sec = (w["end"] - clip_start_ms) / 1000.0
                word_clip = (
                    word_clip.with_position(("center", int(video_h * 0.75)))
                    .with_start(start_sec)
                    .with_end(end_sec)
                )
                caption_clips.append(word_clip)
        final = CompositeVideoClip([video] + caption_clips)
        output_path = os.path.join(output_folder, f"subtitled_{clip_file}")
        final.write_videofile(
            output_path, codec="libx264", audio_codec="aac", preset="medium", threads=2
        )

        video.close()
        final.close()

    logger.info("All clips subtitled successfully.")
