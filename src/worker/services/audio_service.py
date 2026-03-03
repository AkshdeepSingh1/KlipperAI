"""
Audio Service
Extracts audio tracks from video files.
"""
import os
from typing import Optional

from moviepy.video.io.VideoFileClip import VideoFileClip

from src.shared.core.logger import get_logger

logger = get_logger(__name__)


def extract_audio(user_id: str, video_id: str) -> Optional[str]:
    """
    Extract audio from a video file using MoviePy.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction

    Returns:
        Path to the extracted audio file

    Raises:
        FileNotFoundError: If the video file does not exist
        ValueError: If the video has no audio track
        RuntimeError: If audio extraction fails
    """
    video_path = os.path.join("downloads", user_id, video_id, "video.mp4")
    if not os.path.exists(video_path):
        error_msg = f"Video file not found: {video_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    audio_path = os.path.join("downloads", user_id, video_id, "audio.mp3")
    logger.info(f"Extracting audio from {video_path} to {audio_path}")

    try:
        video = VideoFileClip(video_path)

        if video.audio is None:
            video.close()
            error_msg = f"Video file has no audio track: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        video.audio.write_audiofile(
            audio_path,
            codec="mp3",
            bitrate="192k",
            logger=None,
        )

        video.close()

        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path

    except Exception as e:
        error_msg = f"Error extracting audio with moviepy: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
