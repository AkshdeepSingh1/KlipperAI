"""
Text-to-Content Processor Handler
Orchestrates the generation of content from text/scripts.
"""
import os
import json
import shutil
import requests
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.Loop import Loop

from src.shared.core.database import SessionLocal
from src.shared.core.logger import get_logger
from src.shared.enums import ContentJobRequestProcessingStatus
from src.shared.models import ContentJobRequest, VideoTemplate
from src.worker.services.processing_lock_service import acquire_content_job_lock
from src.worker.services.elevenlabs_service import eleven_labs_service
from src.worker.services.subtitles import SubtitleEngine, SubtitleStyleRegistry
from src.worker.services.processor2_storage_service import processor2_storage_service

logger = get_logger(__name__)

class TextToContentProcessor:
    """Orchestrates text-to-content processing tasks."""

    def process_request(self, request_id: int):
        """
        Process a content job request.
        
        Args:
            request_id: Database ID of the content_job_request
        """
        db: Session = SessionLocal()
        
        try:
            logger.info(f"Starting text-to-content processing for request_id={request_id}")
            
            # 1. Acquire idempotent processing lock
            job_request = acquire_content_job_lock(db, request_id)
            if job_request is None:
                # Lock not acquired (already processing, completed, or error)
                return

            # 2. Start the processor logic
            logger.info(f"Processor started for request_id={request_id}")
            
            # Get template info
            if job_request.template_id is None:
                raise ValueError(f"No template_id provided for request {request_id}")
                
            template = db.query(VideoTemplate).filter(VideoTemplate.id == job_request.template_id).first()
            if not template:
                raise ValueError(f"Template with id {job_request.template_id} not found")
                
            # Create download directory
            # root/Processor2Downloads/templates/templateId
            base_download_dir = "Processor2Downloads"
            template_dir = os.path.join(base_download_dir, "templates", str(template.id))
            os.makedirs(template_dir, exist_ok=True)
            
            video_path = os.path.join(template_dir, "template_video.mp4")
            
            # Download template video if it doesn't exist
            if not os.path.exists(video_path):
                logger.info(f"Downloading template video from {template.video_url} to {video_path}")
                response = requests.get(template.video_url, stream=True, timeout=60)
                response.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Template video downloaded to {video_path}")
            else:
                logger.info(f"Template video already exists at {video_path}")

            # 3. TEXT TO SPEECH (ElevenLabs)
            job_dir = os.path.join(base_download_dir, "jobs", str(request_id))
            os.makedirs(job_dir, exist_ok=True)
            
            audio_path = os.path.join(job_dir, "speech.mp3")
            timestamps_path = audio_path.replace(".mp3", "_timestamps.json")

            # Skip TTS if files already exist
            if os.path.exists(audio_path) and os.path.exists(timestamps_path):
                logger.info(f"TTS files already exist for request_id={request_id}, skipping ElevenLabs call.")
            else:
                logger.info(f"Starting Text-to-Speech for request_id={request_id}")
                
                script_to_use = job_request.user_script or job_request.generated_script
                if not script_to_use:
                    # Fallback to prompt if script is missing
                    if job_request.prompt:
                        logger.warning("No script found, using prompt for TTS as fallback")
                        script_to_use = job_request.prompt
                    else:
                        raise ValueError(f"No script or prompt found for TTS in request {request_id}")
                
                eleven_labs_service.text_to_speech(
                    text=script_to_use,
                    voice_over_id=job_request.voice_over_id,
                    output_path=audio_path,
                    db=db
                )

            # 4. Final Video Assembly
            logger.info(f"Starting Final Video Assembly for request_id={request_id}")
            
            # Paths
            final_output_dir = os.path.join(job_dir, "output")
            os.makedirs(final_output_dir, exist_ok=True)
            
            # Using a unique GUID for the final video filename
            final_video_filename = f"{uuid.uuid4().hex}.mp4"
            final_video_path = os.path.join(final_output_dir, final_video_filename)
            
            self._assemble_video_with_subtitles(
                video_path=video_path,
                audio_path=audio_path,
                timestamps_path=timestamps_path,
                output_path=final_video_path,
                subtitle_style="REDDIT_STORIES"
            )

            # 5. UPLOAD TO AZURE
            logger.info(f"Uploading final video to Azure for request_id={request_id}")
            blob_url = processor2_storage_service.upload_video(
                local_path=final_video_path,
                request_id=request_id,
                filename=final_video_filename
            )
            
            # 6. Finalize
            job_request.output_url = final_video_filename
            job_request.processing_status = ContentJobRequestProcessingStatus.COMPLETED
            job_request.updated_at_utc = datetime.utcnow()
            db.commit()
            
            # 7. Cleanup
            logger.info(f"Cleaning up job directory: {job_dir}")
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
                
            logger.info(f"Successfully processed request_id={request_id}. Video GUID: {final_video_filename}")
            
        except Exception as e:
            logger.error(f"Error processing content request {request_id}: {e}", exc_info=True)
            try:
                # Re-query to mark as FAILED if we had a lock
                # (Ideally acquire_content_job_lock handles the initial state)
                db.rollback()
                job_request = db.query(ContentJobRequest).filter(ContentJobRequest.id == request_id).first()
                if job_request:
                    job_request.processing_status = ContentJobRequestProcessingStatus.FAILED
                    job_request.updated_at_utc = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            raise
            
        finally:
            db.close()

    def _assemble_video_with_subtitles(self, video_path: str, audio_path: str, timestamps_path: str, output_path: str, subtitle_style: str = "TIKTOK_BOLD"):
        """
        Overlay audio on video and add subtitles.
        """
        logger.info(f"Assembling video: {video_path} + {audio_path} with style {subtitle_style}")
        
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # Looping/Cutting video to match audio duration
        if video.duration < audio.duration:
            logger.info(f"Video duration ({video.duration}s) < audio duration ({audio.duration}s). Looping video.")
            video = video.with_effects([Loop(duration=audio.duration)])
        else:
            logger.info(f"Video duration ({video.duration}s) >= audio duration ({audio.duration}s). Cutting video.")
            video = video.subclipped(0, audio.duration)
            
        video = video.with_audio(audio)
        
        # Parse timestamps
        if not os.path.exists(timestamps_path):
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")
            
        with open(timestamps_path, "r") as f:
            alignment = json.load(f)
        
        words = self._parse_elevenlabs_timestamps(alignment)
        
        # Generate subtitles using pre-existing subtitle engine
        style = SubtitleStyleRegistry.get(subtitle_style)
        engine = SubtitleEngine(style)
        
        logger.info("Generating subtitle clips...")
        caption_clips = engine.generate_subtitles(
            video=video,
            words=words,
            clip_start_ms=0
        )
        
        logger.info(f"Compositing {len(caption_clips)} subtitle clips onto video.")
        final = CompositeVideoClip([video] + caption_clips)
        
        # Write final video file
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=output_path.replace(".mp4", "_temp_audio.m4a"),
            remove_temp=True,
            threads=2
        )
        
        video.close()
        audio.close()
        final.close()

    def _parse_elevenlabs_timestamps(self, alignment: dict) -> list:
        """
        Convert ElevenLabs character alignment to word-level timestamps.
        """
        words = []
        current_word = ""
        start_time = None
        last_end = 0
        
        characters = alignment.get("characters", [])
        start_times = alignment.get("character_start_times_seconds", [])
        end_times = alignment.get("character_end_times_seconds", [])
        
        for char, start, end in zip(characters, start_times, end_times):
            if char.isspace():
                if current_word:
                    words.append({
                        "text": current_word,
                        "start": int(start_time * 1000),
                        "end": int(last_end * 1000)
                    })
                    current_word = ""
                    start_time = None
                continue
                
            if start_time is None:
                start_time = start
            current_word += char
            last_end = end
            
        if current_word:
            words.append({
                "text": current_word,
                "start": int(start_time * 1000),
                "end": int(last_end * 1000)
            })
            
        return words
