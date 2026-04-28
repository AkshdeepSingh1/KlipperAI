import enum


class RenderFormat(str, enum.Enum):
    """Render format enum for content job requests"""
    VERTICAL_9_16 = "vertical_9_16"
    SQUARE_1_1 = "square_1_1"
    LANDSCAPE_16_9 = "landscape_16_9"
