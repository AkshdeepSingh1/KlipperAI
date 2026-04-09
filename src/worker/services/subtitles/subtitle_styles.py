"""
Subtitle Styles
Defines the SubtitleStyle configuration dataclass and a registry
of predefined style presets selectable by name.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SubtitleStyle:
    """
    Immutable configuration for subtitle rendering.

    Attributes:
        font:            Path or name of the font file.
        font_size:       Base font size (used when font_size_ratio is None).
        color:           Primary text colour.
        stroke_color:    Outline colour (None → no stroke).
        stroke_width:    Outline thickness in pixels.
        highlight_color: Colour used for word-level highlights.
        position:        Vertical anchor — "top", "center", or "bottom".
        max_width_ratio: Maximum text width as a fraction of video width.
        font_size_ratio: If set, font size = short_side × ratio (dynamic).
        font_size_min:   Floor for dynamic sizing.
        font_size_max:   Ceiling for dynamic sizing.
    """

    font: str = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font_size: int = 60
    color: str = "white"
    stroke_color: Optional[str] = "black"
    stroke_width: int = 3
    highlight_color: str = "yellow"
    position: str = "bottom"           # "top" | "center" | "bottom"
    max_width_ratio: float = 0.8
    font_size_ratio: Optional[float] = 0.045
    font_size_min: int = 14
    font_size_max: int = 42
    mode: str = "chunk"                 # "chunk" | "word" | "karaoke"


class SubtitleStyles:
    """
    Registry of predefined subtitle style presets.

    Usage:
        style = SubtitleStyles.TIKTOK_BOLD
        style = SubtitleStyles.get("MINIMAL")
    """

    TIKTOK_BOLD = SubtitleStyle(
        font="/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        font_size=60,
        color="white",
        stroke_color="black",
        stroke_width=3,
        highlight_color="yellow",
        position="bottom",
        max_width_ratio=0.8,
        font_size_ratio=0.045,
        font_size_min=14,
        font_size_max=42,
        mode="karaoke",
    )

    MINIMAL = SubtitleStyle(
        font="/System/Library/Fonts/Helvetica.ttc",
        font_size=45,
        color="white",
        stroke_color=None,
        stroke_width=0,
        highlight_color="white",
        position="bottom",
        max_width_ratio=0.85,
        font_size_ratio=None,
        font_size_min=14,
        font_size_max=45,
        mode="word",
    )

    CINEMATIC = SubtitleStyle(
        font="/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        font_size=50,
        color="white",
        stroke_color="#333333",
        stroke_width=2,
        highlight_color="#FFD700",
        position="bottom",
        max_width_ratio=0.75,
        font_size_ratio=0.04,
        font_size_min=16,
        font_size_max=50,
        mode="chunk",
    )

    # ── lookup helper ────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> SubtitleStyle:
        """
        Retrieve a preset style by name (case-insensitive).

        Raises:
            ValueError: If the name does not match any preset.
        """
        key = name.strip().upper()
        style = getattr(cls, key, None)
        if not isinstance(style, SubtitleStyle):
            available = [
                attr for attr in dir(cls)
                if isinstance(getattr(cls, attr), SubtitleStyle)
            ]
            raise ValueError(
                f"Unknown subtitle style '{name}'. "
                f"Available: {', '.join(available)}"
            )
        return style


class SubtitleStyleRegistry:
    """
    Explicit registry mapping style names to instances.

    Prefer this over attribute introspection for clarity and control.
    Keys are UPPERCASE names.
    """

    STYLES = {
        "TIKTOK_BOLD": SubtitleStyles.TIKTOK_BOLD,
        "MINIMAL": SubtitleStyles.MINIMAL,
        "CINEMATIC": SubtitleStyles.CINEMATIC,
    }

    @classmethod
    def get(cls, name: str) -> SubtitleStyle:
        key = (name or "").strip().upper()
        if key not in cls.STYLES:
            available = ", ".join(sorted(cls.STYLES.keys()))
            raise ValueError(f"Unknown subtitle style '{name}'. Available: {available}")
        return cls.STYLES[key]
