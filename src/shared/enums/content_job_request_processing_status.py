import enum


class ContentJobRequestProcessingStatus(str, enum.Enum):
    """Processing status enum for content job requests"""
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
