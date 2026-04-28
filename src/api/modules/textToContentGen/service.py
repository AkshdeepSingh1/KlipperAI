from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import List
from src.shared.models import ContentJobRequest, VideoTemplate
from src.shared.enums import ContentJobRequestProcessingStatus


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
    ) -> ContentJobRequest:
        """Create a new content job request"""
        content_job = ContentJobRequest(
            user_id=user_id,
            title=title,
            source_type=source_type,
            prompt=prompt,
            user_script=user_script,
            voice_over_id=voice_over_id,
            render_format=render_format,
            template_id=template_id,
            scheduled_at_utc=datetime.now(timezone.utc),
            processing_status=ContentJobRequestProcessingStatus.SCHEDULED,
        )

        db.add(content_job)
        db.commit()
        db.refresh(content_job)
        return content_job

    @staticmethod
    def get_all_video_templates(db: Session) -> List[VideoTemplate]:
        """Get all video templates"""
        templates = db.query(VideoTemplate).all()
        return templates
