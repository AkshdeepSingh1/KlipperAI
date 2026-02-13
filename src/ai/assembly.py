import assemblyai as aai
from src.shared.core.config import settings
from src.shared.core.logger import get_logger
from typing import Optional

logger = get_logger(__name__)

# Configure API key
aai.settings.api_key = settings.ASSEMBLYAI_API_KEY


def transcribe_audio(file_path: str) -> Optional[dict]:
    """
    Transcribe an audio file using AssemblyAI with language detection.
    Returns the full transcript object which includes word-level timestamps.
    """
    logger.info(f"Starting transcription for file: {file_path}")

    try:
        config = aai.TranscriptionConfig(
            language_detection=True,
            speaker_labels=True,
            speaker_options=aai.SpeakerOptions(use_two_stage_clustering=True),
        )

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"Transcription failed: {transcript.error}")
            raise Exception(f"Transcription failed: {transcript.error}")
    except Exception as e:
        logger.error(f"Error transcribing video: {str(e)}")
        return None

    logger.info("Transcription completed.")
    try:
        data = transcript.json_response
        return {
            "text": data.get("text"),
            "words": data.get("words"),
        }
    except Exception as e:
        logger.error(f"Error building transcript response: {str(e)}")
        return None
