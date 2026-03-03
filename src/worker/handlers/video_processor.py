"""
Video Processor Handler
Handles video processing jobs from the queue
"""
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Iterable, List

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.Crop import Crop
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.shared.core.database import SessionLocal
from src.shared.core.logger import get_logger
from src.shared.enums import ProcessingStatus
from src.shared.models import Clip, ProcessingJob, Video
from src.shared.models.enums import GenerateThumbnailProcess
from src.shared.services.clip_storage_service import clip_storage_service
from src.shared.services.progress_service import update_job_progress
from src.shared.services.thumbnail_queue_service import thumbnail_queue_service
from src.ai.service import (
    download_video_from_azure,
    get_audio_from_video,
    generate_transcript,
    get_clips_from_transcript,
    get_timestamps_from_clips,
    cut_clips_from_video,
    add_subtitles_to_clips,
)

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

            # Idempotent processing lock: try to acquire exclusive RUNNING status
            job = self._acquire_processing_lock(db, video_id, user_id)
            if job is None:
                logger.warning(
                    f"Could not acquire processing lock for video_id={video_id}. "
                    "Another active job is running. Exiting gracefully."
                )
                return

            job_id = job.id
            update_job_progress(job_id, step="queued", progress=10.00)

            user_str = str(user_id) if user_id is not None else "unknown"
            video_str = str(video_id)

            # 1. Ensure raw video is present locally
            download_video_from_azure(user_str, video_str, blob_url)

            # 2. Audio + transcript generation
            update_job_progress(job_id, step="transcription", progress=40.00)
            audio_path = get_audio_from_video(user_str, video_str)
            transcript_path = generate_transcript(user_str, video_str)

            if not audio_path or not transcript_path:
                raise RuntimeError("Failed to generate audio/transcript artifacts")

            # 3. LLM + clip discovery
            update_job_progress(job_id, step="llm", progress=55.00)
            clips_payload = get_clips_from_transcript(user_str, video_str)
            if not clips_payload:
                raise RuntimeError("No clips returned from transcript analysis")

            timestamps = get_timestamps_from_clips(user_str, video_str)
            if not timestamps:
                raise RuntimeError("No timestamps generated for clips")

            # 4. Clip cutting with 9:16 aspect ratio
            update_job_progress(job_id, step="clip_cutting", progress=70.00)
            clips = cut_clips_from_video(user_str, video_str, timestamps)
            if not clips:
                raise RuntimeError("Failed to cut clips from video")
            
            # Crop clips to 9:16 aspect ratio
            clips = self._crop_clips_to_9_16(clips, user_str, video_str)

            # # 5. Optional subtitles
            # if subtitles_enabled():
            #     add_subtitles_to_clips(user_str, video_str)
                
            update_job_progress(job_id, step="subtitles", progress=90.00)

            # 6. Upload generated clips + persist metadata
            clip_records = self._upload_and_record_clips(
                clips=clips,
                user_id=user_str,
                video_id=video_str,
                job_id=job_id,
                db=db,
            )
            logger.info(f"Uploaded {len(clip_records)} clips for video_id={video_id}")

            # 7. Cleanup local workspace
            self._cleanup_downloads(user_str, video_str)

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

    def _acquire_processing_lock(
        self, db: Session, video_id: int, user_id: int
    ) -> ProcessingJob:
        """
        Atomically acquire a processing lock by inserting a RUNNING job.
        If insertion fails, check if the existing RUNNING job is stale (>30 min).
        If stale, mark it FAILED and retry insertion.
        If not stale, return None to signal the caller to exit gracefully.
        """
        STALE_THRESHOLD_MINUTES = 30

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

    def _crop_clips_to_9_16(self, clips: List[str], user_id: str, video_id: str) -> List[str]:
        """
        Crop video clips to 9:16 aspect ratio (vertical format).

        Writes the cropped file in-place (replaces the original) so that
        downstream upload paths remain unchanged.

        Args:
            clips: List of clip file paths
            user_id: User ID for path construction
            video_id: Video ID for path construction

        Returns:
            List of cropped clip file paths (same paths as input, overwritten)
        """
        cropped_clips = []

        for i, clip_path in enumerate(clips):
            clip_file = Path(clip_path)
            output_dir = clip_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Temporary file next to the original; renamed over it on success
            temp_cropped_path = output_dir / f"{clip_file.stem}_cropped.mp4"
            temp_audio_path = output_dir / f"temp_audio_{i}.m4a"

            logger.info(f"Cropping clip {i+1}/{len(clips)} to 9:16 aspect ratio: {clip_path}")

            clip = None
            cropped_clip = None
            try:
                clip = VideoFileClip(clip_path)

                w, h = clip.size
                logger.info(f"Original clip dimensions: {w}x{h}")

                # Target 9:16 — keep as much content as possible
                target_width = int(h * 9 / 16)

                if target_width > w:
                    # Source is narrower than 9:16 → crop height to fit
                    target_height = int(w * 16 / 9)
                    y_center = h // 2
                    y1 = max(0, y_center - target_height // 2)
                    y2 = min(h, y1 + target_height)
                    cropped_clip = clip.with_effects([Crop(x1=0, y1=y1, x2=w, y2=y2)])
                    logger.info(f"Cropped height to {y2-y1}px → {w}x{y2-y1}")
                else:
                    # Standard landscape → crop width to 9:16
                    x_center = w // 2
                    x1 = max(0, x_center - target_width // 2)
                    x2 = min(w, x1 + target_width)
                    cropped_clip = clip.with_effects([Crop(x1=x1, y1=0, x2=x2, y2=h)])
                    logger.info(f"Cropped width to {x2-x1}px → {x2-x1}x{h}")

                cropped_clip.write_videofile(
                    str(temp_cropped_path),
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=str(temp_audio_path),
                    remove_temp=True,
                )

            finally:
                # Always release file handles
                if cropped_clip is not None:
                    cropped_clip.close()
                if clip is not None:
                    clip.close()

            # Replace original with cropped version
            if temp_cropped_path.exists():
                clip_file.unlink(missing_ok=True)
                temp_cropped_path.rename(clip_file)
                logger.info(f"Replaced original with cropped clip: {clip_file}")
            else:
                raise RuntimeError(
                    f"Cropped file was not created at {temp_cropped_path}"
                )

            cropped_clips.append(str(clip_file))

        logger.info(f"Cropped {len(cropped_clips)} clips to 9:16 aspect ratio")
        return cropped_clips

    @staticmethod
    def _upload_and_record_clips(
        clips: Iterable[str],
        user_id: str,
        video_id: str,
        job_id: int,
        db: Session,
    ) -> List[Clip]:
        uploaded_clips: List[Clip] = []

        for idx, clip_path in enumerate(clips, start=1):
            blob_url = clip_storage_service.upload_clip(
                local_path=clip_path,
                user_id=user_id,
                video_id=video_id,
                clip_index=idx,
            )

            clip_record = Clip(
                job_id=job_id,
                video_id=int(video_id) if video_id.isdigit() else None,
                clip_url=blob_url,
            )
            db.add(clip_record)
            uploaded_clips.append(clip_record)

            thumbnail_queue_service.send_thumbnail_generation_message(
                entity_id=clip_record.id,
                process_type=GenerateThumbnailProcess.CLIP_THUMBNAIL,
            )

        db.commit()
        return uploaded_clips

    @staticmethod
    def _cleanup_downloads(user_id: str, video_id: str) -> None:
        download_dir = Path("downloads") / user_id / video_id
        if download_dir.exists():
            shutil.rmtree(download_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {download_dir}")
