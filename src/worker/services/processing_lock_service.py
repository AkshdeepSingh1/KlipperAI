"""
Processing Lock Service
Manages idempotent processing locks for video jobs to prevent duplicate work.
"""
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.shared.core.logger import get_logger
from src.shared.enums import ProcessingStatus, ContentJobRequestProcessingStatus
from src.shared.models import ProcessingJob, ContentJobRequest

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


def acquire_content_job_lock(db: Session, request_id: int) -> ContentJobRequest:
    """
    Atomically acquire a lock for a content job request.
    Uses row-level locking (FOR UPDATE) to ensure idempotency.

    Args:
        db: Active SQLAlchemy session
        request_id: ID of the content_job_request to lock

    Returns:
        ContentJobRequest instance if lock acquired, None otherwise
    """
    try:
        # Fetch the request row with a row-level lock
        job_request = (
            db.query(ContentJobRequest)
            .filter(ContentJobRequest.id == request_id)
            .with_for_update()
            .first()
        )

        if not job_request:
            logger.error(f"ContentJobRequest with id={request_id} not found.")
            return None

        # Check for idempotency
        if job_request.processing_status == ContentJobRequestProcessingStatus.COMPLETED:
            logger.info(f"Request {request_id} is already COMPLETED. Skipping.")
            return None

        if job_request.processing_status == ContentJobRequestProcessingStatus.PROCESSING:
            # Check for stale jobs (optional, but good for robustness)
            age = datetime.utcnow() - job_request.updated_at_utc.replace(tzinfo=None)
            if age > timedelta(minutes=STALE_THRESHOLD_MINUTES):
                logger.warning(
                    f"Found stale PROCESSING job for request_id={request_id} (age: {age}). Retrying."
                )
            else:
                logger.warning(
                    f"Request {request_id} is already in PROCESSING state and not stale. Skipping."
                )
                return None

        # Acquire lock by updating status
        logger.info(f"Acquiring lock: Updating status to PROCESSING for request_id={request_id}")
        job_request.processing_status = ContentJobRequestProcessingStatus.PROCESSING
        job_request.updated_at_utc = datetime.utcnow()
        db.commit()
        db.refresh(job_request)

        return job_request

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to acquire content job lock for request_id={request_id}: {e}")
        return None
