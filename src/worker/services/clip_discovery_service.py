"""
Clip Discovery Service
Uses LLM to identify viral-worthy clips from transcripts
and resolves their precise timestamps.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.ai.gpt import get_clips_from_video, Clips
from src.shared.core.logger import get_logger

logger = get_logger(__name__)


def discover_clips(user_id: str, video_id: str) -> bool:
    """
    Identify clip-worthy segments from a transcript using LLM.
    Saves the results to 'clips.json' in the video's download directory.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction

    Returns:
        True if clips were successfully discovered and saved, False otherwise.
    """
    clips_data_path = os.path.join("downloads", user_id, video_id, "clips.json")

    # Checkpoint: skip LLM call if clips already discovered
    if os.path.exists(clips_data_path) and os.path.getsize(clips_data_path) > 0:
        logger.info(f"Clips already exist at {clips_data_path}, skipping discovery")
        return True

    transcript_path = os.path.join("downloads", user_id, video_id, "transcript.json")
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    clips = get_clips_from_video(transcript["text"])
    if not clips or not clips.clips:
        return False

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
    return True


def resolve_timestamps(
    user_id: str, video_id: str
) -> bool:
    """
    Resolve precise word-level timestamps for discovered clips
    by matching clip text against the transcript.
    Saves the results to 'clips_timestamps.json'.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction

    Returns:
        True if timestamps were successfully resolved and saved, False otherwise.
    """
    clips_timestamps_path = os.path.join(
        "downloads", user_id, video_id, "clips_timestamps.json"
    )

    # Checkpoint: skip timestamp resolution if already done
    if os.path.exists(clips_timestamps_path) and os.path.getsize(clips_timestamps_path) > 0:
        logger.info(f"Timestamps already exist at {clips_timestamps_path}, skipping resolution")
        return True

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

    if not results:
        return False

    clips_timestamps_path = os.path.join(
        "downloads", user_id, video_id, "clips_timestamps.json"
    )
    with open(clips_timestamps_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return True
