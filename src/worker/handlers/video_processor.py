"""
Video Processor Handler
Orchestrates the end-to-end video processing pipeline.

This is a pure orchestrator — all business logic lives in dedicated services
under src/worker/services/. Swap any service implementation and this file
stays unchanged.
"""
import os
from datetime import datetime

from sqlalchemy.orm import Session

from src.shared.core.database import SessionLocal
from src.shared.core.logger import get_logger
from src.shared.enums import ProcessingStatus
from src.shared.services.progress_service import update_job_progress

from src.worker.services.processing_lock_service import acquire_lock
from src.worker.services.video_download_service import download_from_azure
from src.worker.services.audio_service import extract_audio
from src.worker.services.transcript_service import generate_transcript
from src.worker.services.clip_discovery_service import discover_clips, resolve_timestamps
from src.worker.services.video_editing_service import cut_clips
from src.worker.services.smart_crop_service import smart_crop_clip
from src.worker.services.clip_record_service import upload_and_record_clips
from src.worker.services.file_cleanup_service import cleanup_downloads
from src.worker.services.video_editing_service import get_or_cut_clips

logger = get_logger(__name__)


class VideoProcessor:
    """Orchestrates video processing tasks."""

    def process_video(
        self,
        video_id: int,
        blob_name: str,
        blob_url: str,
        user_id: int = None,
    ):
        """
        Process a video job end-to-end.

        Args:
            video_id: Database ID of the video
            blob_name: Azure blob name
            blob_url: Full blob URL
            user_id: Optional user ID
        """
        db: Session = SessionLocal()

        try:
            logger.info(f"Starting video processing for video_id={video_id}")

            # 1. Acquire idempotent processing lock
            job = acquire_lock(db, video_id, user_id)
            if job is None:
                logger.warning(
                    f"Could not acquire processing lock for video_id={video_id}. "
                    "Another active job is running. Exiting gracefully."
                )
                return

            job_id = job.id
            update_job_progress(job_id, step="queued", progress=3.00)

            user_str = str(user_id) if user_id is not None else "unknown"
            video_str = str(video_id)

            # 2. Download raw video
            download_from_azure(user_str, video_str, blob_url)
            update_job_progress(job_id, step="audio_extraction", progress=10.00)

            # 3. Audio extraction
            audio_path = extract_audio(user_str, video_str)
            update_job_progress(job_id, step="transcript_generation", progress=13.00)

            #4 transcript generation
            transcript_path = generate_transcript(user_str, video_str)
            if not audio_path or not transcript_path:
                raise RuntimeError("Failed to generate audio/transcript artifacts")
            update_job_progress(job_id, step="LLM", progress=25.00)

            # 5. LLM clip discovery
            success = discover_clips(user_str, video_str)
            if not success:
                raise RuntimeError("No clips returned from transcript analysis")
            update_job_progress(job_id, step="resolving_timestamps", progress=45.00)

            #6. Resolve timestamps
            success = resolve_timestamps(user_str, video_str)
            if not success:
                raise RuntimeError("No timestamps generated for clips")
            update_job_progress(job_id, step="cutting_clips", progress=50.00)

            # 7. Cut clips 
            clip_path_list = get_or_cut_clips(user_str, video_str)
            update_job_progress(job_id, step="smart_clip_cropping", progress=60.00)

            # 8. Smart crop clips 
            clip_path_list = [smart_crop_clip(clip_path) for clip_path in clip_path_list]
            update_job_progress(job_id, step="adding_subtitles", progress=70.00)

            # 9. (Optional) Subtitles — uncomment when ready
            # from src.worker.services.video_editing_service import add_subtitles
            # add_subtitles(user_str, video_str)

            update_job_progress(job_id, step="subtitles", progress=90.00)

            # 10. Upload clips + persist DB records
            clip_records = upload_and_record_clips(
                clips=clip_path_list,
                user_id=user_str,
                video_id=video_str,
                job_id=job_id,
                db=db,
                timestamps=clips_timestamps_path,
            )
            logger.info(f"Uploaded {len(clip_records)} clips for video_id={video_id}")

            # 11. Cleanup local workspace
            # cleanup_downloads(user_str, video_str)
            update_job_progress(job_id, step="done", progress=100.00)

            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"Video processing completed for video_id={video_id}, job_id={job_id}")

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            try:
                if "job" in locals() and job.id:
                    update_job_progress(job.id, step="error", error_message=str(e))
                    job.status = ProcessingStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            raise

        finally:
            db.close()
