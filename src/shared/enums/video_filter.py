from enum import IntEnum

class VideoFilterStatus(IntEnum):
    """Enum for filtering user videos by processing status"""
    INCOMPLETE = 0
    COMPLETED = 1
    ALL = 2
