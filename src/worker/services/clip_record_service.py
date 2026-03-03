"""
Clip Record Service
Uploads processed clips to Azure Blob Storage and persists Clip records to the database.
"""
from typing import Iterable, List

from sqlalchemy.orm import Session

from src.shared.core.logger import get_logger
from src.shared.models import Clip
from src.shared.models.enums import GenerateThumbnailProcess
from src.shared.services.clip_storage_service import clip_storage_service
from src.shared.services.thumbnail_queue_service import thumbnail_queue_service

logger = get_logger(__name__)


def upload_and_record_clips(
    clips: Iterable[str],
    user_id: str,
    video_id: str,
    job_id: int,
    db: Session,
    timestamps: List[dict] = None,
) -> List[Clip]:
    """
    Upload each clip to Azure Blob Storage, create a Clip DB record,
    and queue thumbnail generation.

    Args:
        clips: Iterable of local clip file paths
        user_id: User identifier
        video_id: Video identifier
        job_id: Processing job ID
        db: Active SQLAlchemy session
        timestamps: Optional list of timestamp dicts for each clip
    """
    uploaded_clips: List[Clip] = []

    for idx, clip_path in enumerate(clips):
        blob_url = clip_storage_service.upload_clip(
            local_path=clip_path,
            user_id=user_id,
            video_id=video_id,
            clip_index=idx + 1,
        )

        # Extract timestamps if available
        start_time = None
        end_time = None
        duration = None
        if timestamps and idx < len(timestamps):
            ts = timestamps[idx]
            # Convert ms to seconds
            start_time = ts.get("start", 0) / 1000.0
            end_time = ts.get("end", 0) / 1000.0
            duration = end_time - start_time

        clip_record = Clip(
            job_id=job_id,
            video_id=int(video_id) if video_id.isdigit() else None,
            clip_url=blob_url,
            start_time_sec=start_time,
            end_time_sec=end_time,
            duration_sec=duration,
        )
        db.add(clip_record)
        uploaded_clips.append(clip_record)

        thumbnail_queue_service.send_thumbnail_generation_message(
            entity_id=clip_record.id,
            process_type=GenerateThumbnailProcess.CLIP_THUMBNAIL,
        )

    db.commit()
    return uploaded_clips
