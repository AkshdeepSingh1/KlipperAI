from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import List, Optional
from src.shared.models import ContentJobRequest, VideoTemplate
from src.shared.enums import ContentJobRequestProcessingStatus
from src.shared.core.logger import get_logger

logger = get_logger(__name__)


class ContentJobService:
    """Service class for handling content job request operations"""

    @staticmethod
    def create_content_job_request(
        db: Session,
        user_id: int,
        title: str,
        source_type: str,
        prompt: str = None,
        user_script: str = None,
        voice_over_id: int = None,
        render_format: str = None,
        template_id: int = None,
        scheduled_at_utc: Optional[datetime] = None,
        run_now: bool = False,
    ) -> ContentJobRequest:
        """
        Create a new content job request.

        - run_now=True: schedule it for "now" and enqueue it to the P2 queue
          immediately, so the worker starts on it right away (no wait for cron).
        - run_now=False: use the caller-provided ``scheduled_at_utc`` (the day the
          user picked, as a UTC timestamp). The daily cron dispatches it on that
          UTC date. Falls back to now if no date was supplied.
        """
        if run_now:
            effective_scheduled_at = datetime.now(timezone.utc)
        else:
            effective_scheduled_at = scheduled_at_utc or datetime.now(timezone.utc)

        content_job = ContentJobRequest(
            user_id=user_id,
            title=title,
            source_type=source_type,
            prompt=prompt,
            user_script=user_script,
            voice_over_id=voice_over_id,
            render_format=render_format,
            template_id=template_id,
            scheduled_at_utc=effective_scheduled_at,
            processing_status=ContentJobRequestProcessingStatus.SCHEDULED,
        )

        db.add(content_job)
        db.commit()
        db.refresh(content_job)

        if run_now:
            # Push straight onto the P2 queue using the same path as the cron.
            from src.worker.services.scheduled_jobs_dispatcher_service import (
                ScheduledJobsDispatcherService,
            )

            enqueued = ScheduledJobsDispatcherService().enqueue_job(content_job)
            if not enqueued:
                logger.error(
                    f"Run-now enqueue failed for content job {content_job.id}"
                )
                raise RuntimeError(
                    "Content job was created but could not be enqueued for immediate processing"
                )
            logger.info(f"Content job {content_job.id} enqueued for immediate (run-now) processing")

        return content_job

    @staticmethod
    def get_all_video_templates(db: Session) -> List[VideoTemplate]:
        """Get all video templates"""
        templates = db.query(VideoTemplate).all()
        return templates

    @staticmethod
    def get_all_voice_templates(db: Session) -> List:
        """Get all active voice templates"""
        from src.shared.models import VoiceTemplate
        return db.query(VoiceTemplate).filter(VoiceTemplate.is_active == True).order_by(VoiceTemplate.sort_order).all()

    @staticmethod
    def get_latest_renders(db: Session, last_id: int = None, quantity: int = 5) -> List[ContentJobRequest]:
        """Get latest completed renders with pagination"""
        query = db.query(ContentJobRequest).filter(
            ContentJobRequest.processing_status == ContentJobRequestProcessingStatus.COMPLETED,
            ContentJobRequest.output_url != None
        )
        
        if last_id:
            query = query.filter(ContentJobRequest.id < last_id)
            
        return query.order_by(ContentJobRequest.id.desc()).limit(quantity).all()

    @staticmethod
    def get_scheduled_content(db: Session, user_id: int, target_date: datetime.date) -> List[ContentJobRequest]:
        """Get all content job requests scheduled for a specific date for a user"""
        from sqlalchemy import func
        return db.query(ContentJobRequest).filter(
            ContentJobRequest.user_id == user_id,
            func.date(ContentJobRequest.scheduled_at_utc) == target_date
        ).all()

    @staticmethod
    def get_dashboard_stats(db: Session, user_id: int) -> dict:
        """Get dashboard statistics for a user"""
        from sqlalchemy import func
        from datetime import datetime
        
        now = datetime.utcnow()
        today = now.date()
        first_day_of_month = today.replace(day=1)
        
        # Scheduled this month
        scheduled_this_month = db.query(ContentJobRequest).filter(
            ContentJobRequest.user_id == user_id,
            ContentJobRequest.scheduled_at_utc >= first_day_of_month
        ).count()
        
        # In progress
        in_progress = db.query(ContentJobRequest).filter(
            ContentJobRequest.user_id == user_id,
            ContentJobRequest.processing_status == ContentJobRequestProcessingStatus.PROCESSING
        ).count()
        
        # Scheduled today
        scheduled_today = db.query(ContentJobRequest).filter(
            ContentJobRequest.user_id == user_id,
            func.date(ContentJobRequest.scheduled_at_utc) == today
        ).count()
        
        return {
            "scheduledThisMonth": scheduled_this_month,
            "inProgress": in_progress,
            "scheduledToday": scheduled_today
        }

    @staticmethod
    def get_monthly_schedule(db: Session, user_id: int, year: int, month: int) -> List[ContentJobRequest]:
        """Get all content job requests scheduled for a specific month for a user"""
        from sqlalchemy import extract
        return db.query(ContentJobRequest).filter(
            ContentJobRequest.user_id == user_id,
            extract('year', ContentJobRequest.scheduled_at_utc) == year,
            extract('month', ContentJobRequest.scheduled_at_utc) == month
        ).all()
