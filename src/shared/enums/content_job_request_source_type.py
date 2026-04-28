import enum


class ContentJobRequestSourceType(str, enum.Enum):
    """Source type enum for content job requests"""
    AI = "ai"
    MY_SCRIPT = "my_script"
