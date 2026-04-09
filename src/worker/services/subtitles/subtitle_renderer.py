"""
Subtitle Renderer
Responsible for creating MoviePy text clips from text + style config.
Contains no business/segmentation logic — only rendering concerns.
"""

from typing import List, Tuple

from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from src.worker.services.subtitles.subtitle_styles import SubtitleStyle


class SubtitleRenderer:
    """
    Renders individual subtitle clips using MoviePy.

    All positioning and font-size computation is derived from the
    supplied ``SubtitleStyle`` so this class never hard-codes any
    visual property.
    """

    # ── public API ───────────────────────────────────────────────────

    def render_segment(
        self,
        text: str,
        video_w: int,
        video_h: int,
        style: SubtitleStyle,
        start: float,
        end: float,
    ):
        """
        Render a multi-word caption segment as a positioned clip.

        Returns a ``CompositeVideoClip`` with the text centred
        horizontally and placed according to *style.position*.
        """
        font_size = self._compute_font_size(style, video_w, video_h)
        width = int(video_w * style.max_width_ratio)

        # Use identical rendering method as karaoke word overlays to avoid metric drift
        # Karaoke: single-line TextClip for consistent glyph metrics with highlights
        if (style.mode or "").lower() == "karaoke":
            txt_clip = TextClip(
                text=text,
                font=style.font,
                font_size=font_size,
                color=style.color,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
            )
        else:
            txt_clip = TextClip(
                text=text,
                font=style.font,
                font_size=font_size,
                color=style.color,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
                method="caption",
                size=(width, None),
                text_align="center",
            )

        pos_y = self._compute_position(style.position, video_h, txt_clip.h)
        if (style.mode or "").lower() == "karaoke":
            # Calculate left edge: must match highlight word calculation exactly
            left_x = int((video_w - txt_clip.w) / 2)
            txt_clip = txt_clip.with_position((left_x, pos_y))
        else:
            txt_clip = txt_clip.with_position(("center", pos_y))

        caption = (
            CompositeVideoClip([txt_clip], size=(video_w, video_h))
            .with_start(start)
            .with_end(end)
        )
        return caption

    def render_highlight_word(
        self,
        word_text: str,
        video_w: int,
        video_h: int,
        style: SubtitleStyle,
        start: float,
        end: float,
    ):
        """
        Render a single highlighted word overlay.

        Uses the style's ``highlight_color`` instead of the default
        text colour.  Positioned at the same vertical anchor as normal
        segments.
        """
        font_size = self._compute_font_size(style, video_w, video_h)

        word_clip = TextClip(
            text=word_text,
            font=style.font,
            font_size=font_size,
            color=style.highlight_color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width,
        )

        pos_y = self._compute_position(style.position, video_h, word_clip.h)
        word_clip = (
            word_clip.with_position(("center", pos_y))
            .with_start(start)
            .with_end(end)
        )
        return word_clip

    def render_karaoke_word(
        self,
        full_text: str,
        word_index: int,
        words: List[str],
        video_w: int,
        video_h: int,
        style: SubtitleStyle,
        start: float,
        end: float,
        is_highlighted: bool = True,
    ):
        """
        Render a single highlighted word positioned exactly where it appears
        within the full sentence context.

        This is used for karaoke-style subtitles where the full sentence is
        displayed and individual words are highlighted in sequence.

        Args:
            full_text:   The complete sentence text.
            word_index:  Index of the word to highlight (0-based).
            words:       List of individual word strings in the sentence.
            video_w:     Video width in pixels.
            video_h:     Video height in pixels.
            style:       SubtitleStyle configuration.
            start:       Start time in seconds.
            end:         End time in seconds.

        Returns:
            TextClip positioned at the exact horizontal offset of the word.
        """
        font_size = self._compute_font_size(style, video_w, video_h)

        # Render full sentence as single line (no caption wrapping) for measurements
        # This gives us the actual text width for positioning calculations
        full_clip = TextClip(
            text=full_text,
            font=style.font,
            font_size=font_size,
            color=style.color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width,
        )

        # Calculate Y position (same as base sentence)
        pos_y = self._compute_position(style.position, video_h, full_clip.h)

        # Calculate horizontal offset for the target word
        # Measure everything before target word INCLUDING the trailing space as single unit
        prefix_width = 0
        if word_index > 0:
            # Include all words before target PLUS trailing space as one measurement
            prefix_text_with_space = " ".join(words[:word_index]) + " "
            
            prefix_clip = TextClip(
                text=prefix_text_with_space,
                font=style.font,
                font_size=font_size,
                color=style.color,
                stroke_color=style.stroke_color,
                stroke_width=style.stroke_width,
            )
            
            prefix_width = int(prefix_clip.w)

        # Render the target word in appropriate color based on highlight state
        word_text = words[word_index]
        word_color = style.highlight_color if is_highlighted else style.color
        word_clip = TextClip(
            text=word_text,
            font=style.font,
            font_size=font_size,
            color=word_color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width,
        )

        # Calculate absolute X position using identical logic to base sentence
        sentence_left_edge = int((video_w - full_clip.w) / 2)
        word_x = sentence_left_edge + prefix_width

        word_clip = (
            word_clip.with_position((word_x, pos_y))
            .with_start(start)
            .with_end(end)
        )
        return word_clip

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_font_size(
        style: SubtitleStyle, video_w: int, video_h: int
    ) -> int:
        """
        Compute the actual font size.

        If ``style.font_size_ratio`` is set the size scales with the
        shorter side of the video; otherwise ``style.font_size`` is
        returned as-is.
        """
        if style.font_size_ratio is not None:
            short_side = min(video_w, video_h)
            return max(
                style.font_size_min,
                min(style.font_size_max, int(short_side * style.font_size_ratio)),
            )
        return style.font_size

    @staticmethod
    def _compute_position(
        position: str, video_h: int, clip_h: int
    ) -> int:
        """
        Map a named position to a pixel Y-coordinate.

        Supported values: ``"top"``, ``"center"``, ``"bottom"``.
        """
        if position == "top":
            return max(0, int(video_h * 0.08))
        if position == "center":
            return max(0, (video_h - clip_h) // 2)
        # default: "bottom"
        anchor_y = int(video_h * 0.75)
        pos_y = min(anchor_y, video_h - clip_h)
        return max(0, pos_y)
