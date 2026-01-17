import os
import json
import yt_dlp
from typing import List, Tuple, Optional
from src.shared.core.logger import get_logger
from src.ai.assembly import transcribe_audio
from src.ai.gpt import get_clips_from_video, Clips
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore

logger = get_logger(__name__)


def download_youtube_video(url: str, user_id: str, video_id: str) -> str:
    """
    Download a video from a URL using yt-dlp
    """
    logger.info(f"Starting download for URL: {url}")

    # Ensure downloads directory exists
    download_dir = os.path.join("downloads", user_id, video_id)
    os.makedirs(download_dir, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        "format": "best[ext=mp4][protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]/best[protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]",
        "format_sort": ["lang:en"],
        "outtmpl": os.path.join(download_dir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 3,  # Retry failed downloads
        "fragment_retries": 3,  # Retry failed fragments
        "ignoreerrors": False,
        "no_check_certificate": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info and download
            info = ydl.extract_info(url, download=True)
            # Get the path of the downloaded file
            file_path = ydl.prepare_filename(info)

            # Verify the file was actually downloaded and is not empty
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found: {file_path}")

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")

            logger.info(
                f"Video downloaded successfully: {file_path} (size: {file_size} bytes)"
            )
            return file_path

    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_audio_from_video(user_id: str, video_id: str) -> Optional[str]:
    """
    Extract audio from a video file using moviepy.
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

        # Check if video has audio
        if video.audio is None:
            video.close()
            error_msg = f"Video file has no audio track: {video_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extract audio to mp3
        video.audio.write_audiofile(
            audio_path,
            codec="mp3",
            bitrate="192k",
            logger=None,  # Suppress moviepy's verbose logging
        )

        video.close()

        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path

    except Exception as e:
        error_msg = f"Error extracting audio with moviepy: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def generate_transcript(user_id: str, video_id: str) -> Optional[str]:
    """
    Generate a transcript from an audio file
    """
    audio_path = os.path.join("downloads", user_id, video_id, "audio.mp3")
    transcript = transcribe_audio(audio_path)
    if transcript:
        transcript_path = os.path.join(
            "downloads", user_id, video_id, "transcript.json"
        )
        with open(transcript_path, "w") as f:
            json.dump(transcript, f)
        return transcript_path
    else:
        error_msg = f"Error generating transcript: {transcript}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_clips_from_transcript(user_id: str, video_id: str) -> Optional[Clips]:
    """
    Get clips from a transcript
    """
    transcript_path = os.path.join("downloads", user_id, video_id, "transcript.json")
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    clips = get_clips_from_video(transcript["text"])
    clips_data = []
    for clip in clips.clips:
        clips_data.append(
            {
                "clip_text": clip.clip_text,
            }
        )
    clips_data_path = os.path.join("downloads", user_id, video_id, "clips.json")
    with open(clips_data_path, "w") as f:
        json.dump(clips_data, f)
    return clips_data_path


def cut_clips_from_video(
    video_path: str, timestamps: List[Tuple[str, str]]
) -> List[str]:
    """
    Cut clips from a video
    """
    # TODO
    return ["clip1.mp4", "clip2.mp4"]


def add_subtitles_to_videos(videos: List[str], subtitles: List[str]) -> str:
    """
    Add subtitles to a video
    """
    # TODO
    return "Subtitles added successfully!"
