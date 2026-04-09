"""
Subtitle Engine
Orchestrates subtitle generation: segments words, delegates rendering
to SubtitleRenderer, and returns a list of composited clips.
"""

from typing import Any, Dict, List, Optional

from src.shared.core.logger import get_logger
from src.worker.services.subtitles.subtitle_styles import SubtitleStyle
from src.worker.services.subtitles.subtitle_renderer import SubtitleRenderer

logger = get_logger(__name__)


class SubtitleEngine:
    """
    High-level subtitle pipeline.

    Receives word-level timestamps, chunks them into readable segments,
    and produces positioned MoviePy clips ready for compositing.
    """

    def __init__(self, style: SubtitleStyle) -> None:
        self.style = style
        self.renderer = SubtitleRenderer()

    # ── public API ───────────────────────────────────────────────────

    def generate_subtitles(
        self,
        video,
        words: List[Dict[str, Any]],
        clip_start_ms: int,
    ) -> list:
        """
        Generate subtitle clips for a single video clip.

        Args:
            video:           MoviePy VideoClip (used only for dimensions / duration).
            words:           List of word dicts with ``text``, ``start``, ``end`` (ms).
            clip_start_ms:   Absolute start timestamp (ms) of the clip in the
                             full video — used to convert absolute word
                             timestamps to clip-relative seconds.
            highlight_words: If True, overlay each word individually in the
                             style's highlight colour.

        Returns:
            List of MoviePy clips to be layered on top of the base video.
        """
        video_w, video_h = video.size
        duration = video.duration
        caption_clips: list = []

        mode = (self.style.mode or "chunk").lower()

        if mode == "chunk":
            # multi-word segments
            segments = self._chunk_words(words)
            for seg in segments:
                text = " ".join(w["text"] for w in seg)
                start_sec = self._ms_to_clip_sec(seg[0]["start"], clip_start_ms)
                end_sec = self._ms_to_clip_sec(seg[-1]["end"], clip_start_ms)

                # Clamp to clip duration
                start_sec = max(0.0, min(start_sec, duration))
                end_sec = max(0.0, min(end_sec, duration))
                if end_sec <= start_sec:
                    continue

                caption_clips.append(
                    self.renderer.render_segment(
                        text, video_w, video_h, self.style, start_sec, end_sec
                    )
                )

        elif mode == "word":
            # per-word overlays (standalone)
            for w in words:
                start_sec = self._ms_to_clip_sec(w["start"], clip_start_ms)
                end_sec = self._ms_to_clip_sec(w["end"], clip_start_ms)
                start_sec = max(0.0, min(start_sec, duration))
                end_sec = max(0.0, min(end_sec, duration))
                if end_sec <= start_sec:
                    continue

                caption_clips.append(
                    self.renderer.render_highlight_word(
                        w["text"],
                        video_w,
                        video_h,
                        self.style,
                        start_sec,
                        end_sec,
                    )
                )

        elif mode == "karaoke":
            # karaoke: render each word individually (no overlays)
            # Active word gets highlight color, inactive words get base color
            segments = self._chunk_words(words)
            for seg in segments:
                full_text = " ".join(w["text"] for w in seg)
                word_list = [w["text"] for w in seg]
                
                # For each word in segment, render it multiple times:
                # - When it's active (during its time): highlighted
                # - When it's inactive (before/after): base color
                for idx, w in enumerate(seg):
                    word_start = self._ms_to_clip_sec(w["start"], clip_start_ms)
                    word_end = self._ms_to_clip_sec(w["end"], clip_start_ms)
                    word_start = max(0.0, min(word_start, duration))
                    word_end = max(0.0, min(word_end, duration))
                    if word_end <= word_start:
                        continue
                    
                    segment_start = self._ms_to_clip_sec(seg[0]["start"], clip_start_ms)
                    segment_end = self._ms_to_clip_sec(seg[-1]["end"], clip_start_ms)
                    segment_start = max(0.0, min(segment_start, duration))
                    segment_end = max(0.0, min(segment_end, duration))
                    
                    # Render word in base color for times before it's active
                    if word_start > segment_start:
                        caption_clips.append(
                            self.renderer.render_karaoke_word(
                                full_text=full_text,
                                word_index=idx,
                                words=word_list,
                                video_w=video_w,
                                video_h=video_h,
                                style=self.style,
                                start=segment_start,
                                end=word_start,
                                is_highlighted=False,
                            )
                        )
                    
                    # Render word in highlight color during its active time
                    caption_clips.append(
                        self.renderer.render_karaoke_word(
                            full_text=full_text,
                            word_index=idx,
                            words=word_list,
                            video_w=video_w,
                            video_h=video_h,
                            style=self.style,
                            start=word_start,
                            end=word_end,
                            is_highlighted=True,
                        )
                    )
                    
                    # Render word in base color for times after it's active
                    if word_end < segment_end:
                        caption_clips.append(
                            self.renderer.render_karaoke_word(
                                full_text=full_text,
                                word_index=idx,
                                words=word_list,
                                video_w=video_w,
                                video_h=video_h,
                                style=self.style,
                                start=word_end,
                                end=segment_end,
                                is_highlighted=False,
                            )
                        )

        logger.debug(
            "Generated %d subtitle clips (mode=%s)", len(caption_clips), mode
        )
        return caption_clips

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _ms_to_clip_sec(abs_ms: int, clip_start_ms: int) -> float:
        """Convert an absolute millisecond timestamp to clip-relative seconds."""
        return (abs_ms - clip_start_ms) / 1000.0

    @staticmethod
    def _word_ends_sentence(word: Dict[str, Any]) -> bool:
        """Return True if this word ends a sentence (line boundary)."""
        text = (word.get("text") or "").strip()
        return any(text.endswith(p) for p in (".", "?", "!"))

    @classmethod
    def _chunk_words(
        cls,
        words: List[Dict[str, Any]],
        min_words: int = 3,
        max_words: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Chunk words into segments of *min_words*–*max_words*.

        Never spans across sentence boundaries (determined by trailing
        punctuation).
        """
        if not words:
            return []

        segments: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for w in words:
            current.append(w)
            if cls._word_ends_sentence(w) or len(current) >= max_words:
                segments.append(current)
                current = []

        if current:
            segments.append(current)

        return segments
