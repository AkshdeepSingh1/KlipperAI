"""
Processing Lock Service
Manages idempotent processing locks for video jobs to prevent duplicate work.
"""
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.shared.core.logger import get_logger
from src.shared.enums import ProcessingStatus
from src.shared.models import ProcessingJob

logger = get_logger(__name__)

STALE_THRESHOLD_MINUTES = 30


def acquire_lock(db: Session, video_id: int, user_id: int) -> ProcessingJob:
    """
    Atomically acquire a processing lock by inserting a RUNNING job.

    If insertion fails due to an existing RUNNING job:
    - If the existing job is stale (>30 min), mark it FAILED and retry.
    - If the existing job is still active, return None.

    Args:
        db: Active SQLAlchemy session
        video_id: ID of the video to lock
        user_id: ID of the user who initiated processing

    Returns:
        ProcessingJob instance if lock acquired, None otherwise
    """
    for attempt in range(2):
        try:
            job = ProcessingJob(
                video_id=video_id,
                user_id=user_id,
                status=ProcessingStatus.RUNNING,
                current_step="queued",
                progress_percentage=10.00,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(
                f"Acquired processing lock for video_id={video_id}, job_id={job.id}"
            )
            return job

        except IntegrityError:
            db.rollback()
            logger.warning(
                f"Failed to acquire lock for video_id={video_id} on attempt {attempt + 1}. "
                "Checking for stale jobs..."
            )

            existing_job = (
                db.query(ProcessingJob)
                .filter(
                    ProcessingJob.video_id == video_id,
                    ProcessingJob.status == ProcessingStatus.RUNNING,
                )
                .order_by(ProcessingJob.created_at.desc())
                .first()
            )

            if existing_job:
                age = datetime.utcnow() - existing_job.created_at
                if age > timedelta(minutes=STALE_THRESHOLD_MINUTES):
                    logger.warning(
                        f"Found stale job {existing_job.id} for video_id={video_id} "
                        f"(age: {age}). Marking as FAILED."
                    )
                    existing_job.status = ProcessingStatus.FAILED
                    existing_job.error_message = (
                        f"Job marked as stale after {STALE_THRESHOLD_MINUTES} minutes"
                    )
                    existing_job.completed_at = datetime.utcnow()
                    db.commit()
                    continue
                else:
                    logger.info(
                        f"Job {existing_job.id} for video_id={video_id} is still active "
                        f"(age: {age}). Cannot acquire lock."
                    )
                    return None
            else:
                logger.error(
                    f"IntegrityError but no RUNNING job found for video_id={video_id}. "
                    "This should not happen."
                )
                return None

    logger.error(
        f"Failed to acquire processing lock for video_id={video_id} after retries."
    )
    return None
