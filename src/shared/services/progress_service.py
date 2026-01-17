from typing import Optional

from src.shared.core.database import SessionLocal
from src.shared.core.logger import get_logger
from src.shared.models import ProcessingJob


logger = get_logger(__name__)


def update_job_progress(
    job_id: int,
    step: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    Update processing job progress fields.

    Args:
        job_id: ID of the processing job
        step: Current step name (e.g., 'queued', 'transcription')
        progress: Progress percentage (0.00 - 100.00)
        error_message: Optional error message

    Returns:
        bool indicating success
    """
    db = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if not job:
            logger.error(f"ProcessingJob not found: id={job_id}")
            return False

        if step is not None:
            job.current_step = step
        if progress is not None:
            job.progress_percentage = progress
        if error_message is not None:
            job.error_message = error_message

        db.commit()
        logger.info(
            f"Job progress updated: id={job_id}, step={step}, progress={progress}, error={bool(error_message)}"
        )
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update job progress id={job_id}: {e}", exc_info=True)
        return False
    finally:
        db.close()
