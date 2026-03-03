"""
Transcript Service
Generates transcripts from audio files via AssemblyAI.
"""
import json
import os
from typing import Optional

from src.ai.assembly import transcribe_audio
from src.shared.core.logger import get_logger

logger = get_logger(__name__)


def generate_transcript(user_id: str, video_id: str) -> Optional[str]:
    """
    Generate a transcript from an audio file using AssemblyAI.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction

    Returns:
        Path to the saved transcript JSON file

    Raises:
        Exception: If transcription fails
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
