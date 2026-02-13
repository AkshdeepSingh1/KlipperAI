import os
import json
import yt_dlp
import re
import requests
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Optional
from src.shared.core.logger import get_logger
from src.ai.assembly import transcribe_audio
from src.ai.gpt import get_clips_from_video, Clips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


logger = get_logger(__name__)

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


def download_youtube_video(url: str, user_id: str, video_id: str) -> str:
    """
    Download a video from a URL using yt-dlp
    """
    logger.info(f"Starting download for URL: {url}")

    download_dir = os.path.join("downloads", user_id, video_id)
    os.makedirs(download_dir, exist_ok=True)

    ydl_opts = {
        "format": "best[ext=mp4][protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]/best[protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]",
        "format_sort": ["lang:en"],
        "outtmpl": os.path.join(download_dir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
        "no_check_certificate": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found: {file_path}")

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")

            logger.info(
                f"Video downloaded successfully: {file_path} (size: {file_size} bytes)"
            )
            return file_path

    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def download_video_from_azure(user_id: str, video_id: str, link: str) -> None:
    parsed = urlparse(link)
    output_dir = os.path.join("downloads", user_id, video_id)
    os.makedirs(output_dir, exist_ok=True)

    dest_path = os.path.join("downloads", user_id, video_id, "video.mp4")
    query = parsed.query or ""
    qs = parse_qs(query)
    has_sas = ("sig" in qs) or ("sv" in qs) or ("se" in qs and "sp" in qs)

    if has_sas or parsed.scheme.startswith("http"):
        try:
            with requests.get(link, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded to {dest_path}")
            return
        except requests.HTTPError as e:
            raise requests.HTTPError(f"HTTP error while downloading {link}: {e}") from e
        except requests.RequestException as e:
            raise Exception(f"Network error while downloading {link}: {e}") from e


def get_audio_from_video(user_id: str, video_id: str) -> Optional[str]:
    """
    Extract audio from a video file using moviepy.
    """
    video_path = os.path.join("downloads", user_id, video_id, "video.mp4")
    if not os.path.exists(video_path):
        error_msg = f"Video file not found: {video_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    audio_path = os.path.join("downloads", user_id, video_id, "audio.mp3")
    logger.info(f"Extracting audio from {video_path} to {audio_path}")

    try:
        video = VideoFileClip(video_path)

        # Check if video has audio
        if video.audio is None:
            video.close()
            error_msg = f"Video file has no audio track: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extract audio to mp3
        video.audio.write_audiofile(
            audio_path,
            codec="mp3",
            bitrate="192k",
            logger=None,
        )

        video.close()

        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path

    except Exception as e:
        error_msg = f"Error extracting audio with moviepy: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def generate_transcript(user_id: str, video_id: str) -> Optional[str]:
    """
    Generate a transcript from an audio file
    """
    audio_path = os.path.join("downloads", user_id, video_id, "audio.mp3")
    transcript = transcribe_audio(audio_path)
    if transcript:
        transcript_path = os.path.join(
            "downloads", user_id, video_id, "transcript.json"
        )
        with open(transcript_path, "w") as f:
            json.dump(transcript, f)
        return transcript_path
    else:
        error_msg = f"Error generating transcript: {transcript}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_clips_from_transcript(user_id: str, video_id: str) -> Optional[Clips]:
    """
    Get clips from a transcript
    """
    transcript_path = os.path.join("downloads", user_id, video_id, "transcript.json")
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    clips = get_clips_from_video(transcript["text"])
    clips_data = []
    for clip in clips.clips:
        clips_data.append(
            {
                "clip_text": clip.clip_text,
            }
        )
    clips_data_path = os.path.join("downloads", user_id, video_id, "clips.json")
    with open(clips_data_path, "w") as f:
        json.dump(clips_data, f)
    return clips_data


def get_timestamps_from_clips(
    user_id: str, video_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    clips_path = os.path.join("downloads", user_id, video_id, "clips.json")
    with open(clips_path, "r", encoding="utf-8") as f:
        clips = json.load(f)

    transcript_path = os.path.join("downloads", user_id, video_id, "transcript.json")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    
    full_text = transcript["text"]
    words = transcript["words"]

    spans = []
    pos = 0

    for w in words:
        w_text = w["text"]

        m = re.search(re.escape(w_text), full_text[pos:], re.IGNORECASE)

        if m:
            start = pos + m.start()
            end = start + len(m.group(0))
        else:
            m2 = re.search(re.escape(w_text), full_text, re.IGNORECASE)
            if m2:
                start = m2.start()
                end = start + len(m2.group(0))
            else:
                start = pos
                end = start + len(w_text)

        spans.append((start, end))
        pos = end

    def char_to_word_index(char_pos: int) -> int:
        for i, (s, e) in enumerate(spans):
            if s <= char_pos < e:
                return i
        return len(spans) - 1

    results = []

    for clip_d in clips:
        clip = clip_d["clip_text"]

        matches = list(re.finditer(re.escape(clip.strip()), full_text, re.IGNORECASE))

        if not matches:
            continue

        m = matches[0]

        c_start = m.start()
        c_end = m.end()

        start_idx = char_to_word_index(c_start)
        end_idx = char_to_word_index(c_end - 1)

        clip_words = words[start_idx : end_idx + 1]

        result = {
            "text": clip,
            "start": clip_words[0]["start"],
            "end": clip_words[-1]["end"],
            "words": clip_words,
        }

        results.append(result)

    clips_timestamps_path = os.path.join(
        "downloads", user_id, video_id, "clips_timestamps.json"
    )
    with open(clips_timestamps_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results


def _cut_single_clip_moviepy(args) -> str:
    """Worker for cut_clips_from_video; must be at module level for multiprocessing."""
    video_path, start, end, output_path = args

    if start is None or end is None:
        raise ValueError(f"Missing timestamps: start={start}, end={end}")
    # AssemblyAI and clips_timestamps.json use milliseconds; moviepy expects seconds
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


def cut_clips_from_video(user_id: str, video_id: str, timestamps: Any) -> List[str]:
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
    results = [_cut_single_clip_moviepy(job) for job in jobs]
    return results


def font_size_for_video(video_w: int, video_h: int) -> int:
    """Compute subtitle font size from video dimensions (shorter side)."""
    short_side = min(video_w, video_h)
    size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(short_side * FONT_SIZE_RATIO)))
    return size


def create_caption_clip(
    text: str,
    video_w: int,
    video_h: int,
    start: float,
    end: float,
    single_line: bool = True,
):
    """Create a subtitle clip. If single_line is True, use full width so text stays on one line."""
    font_size = font_size_for_video(video_w, video_h)
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


def build_caption_text(words: List[Dict[str, Any]]) -> str:
    return " ".join([w["text"] for w in words])


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
    Never spans across sentence boundaries: if the line is ending, we do not
    pull in words from the next line to reach max_words.
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


def add_subtitles_to_clips(
    user_id: str, video_id: str, highlight_words: bool = False
) -> None:
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
        print(f"Processing subtitles for: {clip_file}")
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
            # Clamp to clip duration and skip if invalid
            start_sec = max(0.0, min(start_sec, video.duration))
            end_sec = max(0.0, min(end_sec, video.duration))
            if end_sec <= start_sec:
                continue
            caption_clips.append(
                create_caption_clip(
                    text, video_w, video_h, start=start_sec, end=end_sec
                )
            )

        if highlight_words:
            font_size = font_size_for_video(video_w, video_h)
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

    print("All clips subtitled successfully.")
