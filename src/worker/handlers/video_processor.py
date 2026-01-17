"""
Video Processor Handler
Handles video processing jobs from the queue
"""
from sqlalchemy.orm import Session
from datetime import datetime
from src.shared.core.logger import get_logger
from src.shared.core.database import SessionLocal
from src.shared.models import Video, ProcessingJob
from src.shared.enums import ProcessingStatus
from src.shared.services.progress_service import update_job_progress
import importlib

logger = get_logger(__name__)


class VideoProcessor:
    """Handles video processing tasks"""
    
    def process_video(
        self,
        video_id: int,
        blob_name: str,
        blob_url: str,
        user_id: int = None
    ):
        """
        Process a video job
        
        Args:
            video_id: Database ID of the video
            blob_name: Azure blob name
            blob_url: Full blob URL
            user_id: Optional user ID
        """
        db: Session = SessionLocal()
        
        try:
            logger.info(f"Starting video processing for video_id={video_id}")

            job = ProcessingJob(
                video_id=video_id,
                user_id=user_id,
                status=ProcessingStatus.PENDING,
                current_step="queued",
                progress_percentage=10.00,
                created_at=datetime.utcnow(),
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            job_id = job.id
            update_job_progress(job_id, step="queued", progress=10.00)

            ai_service = importlib.import_module("src.ai.service")

            update_job_progress(job_id, step="transcription", progress=40.00)
            get_audio = getattr(ai_service, "get_audio_from_video", None)
            generate_transcript = getattr(ai_service, "generate_transcript", None)
            audio_path = get_audio(blob_url) if callable(get_audio) else None
            transcript = generate_transcript(audio_path) if callable(generate_transcript) else None

            update_job_progress(job_id, step="llm", progress=55.00)
            call_llm = getattr(ai_service, "call_llm", None)
            cuts = call_llm(transcript) if callable(call_llm) else []

            update_job_progress(job_id, step="cutting", progress=85.00)
            cut_video = getattr(ai_service, "cut_video", None)
            clips = cut_video(blob_url, cuts) if callable(cut_video) else []

            subtitles_enabled = getattr(ai_service, "subtitles_enabled", lambda: False)
            if subtitles_enabled():
                add_subtitles = getattr(ai_service, "add_subtitles", None)
                if callable(add_subtitles):
                    add_subtitles(clips)
                update_job_progress(job_id, step="subtitles", progress=95.00)

            update_job_progress(job_id, step="done", progress=100.00)
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"Video processing completed for video_id={video_id}, job_id={job_id}")

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            try:
                if 'job' in locals() and job.id:
                    update_job_progress(job.id, step="error", error_message=str(e))
                    job.status = ProcessingStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            raise
            
        finally:
            db.close()
