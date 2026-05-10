import os
import base64
import json
import requests
from sqlalchemy.orm import Session
from src.shared.core.config import settings
from src.shared.core.logger import get_logger
from src.shared.models import VoiceTemplate

logger = get_logger(__name__)

class ElevenLabsService:
    """Handles Text-to-Speech using ElevenLabs API with timestamps."""

    def __init__(self):
        self.api_key = settings.ELEVEN_LABS_API_KEY
        # Fallback voice ID if DB lookup fails
        self.FALLBACK_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

    def text_to_speech(self, text: str, voice_over_id: int, output_path: str, db: Session = None) -> dict:
        """
        Convert text to speech with timestamps using ElevenLabs API and save to disk.
        
        Args:
            text: The text to convert to speech
            voice_over_id: The primary key ID of the VoiceTemplate
            output_path: Local path to save the generated audio file
            db: SQLAlchemy database session for voice lookup
            
        Returns:
            Dict containing paths to the audio and alignment files
        """
        # 1. Lookup Voice ID from Database
        voice_id = self.FALLBACK_VOICE_ID
        
        if db and voice_over_id:
            logger.info(f"Looking up VoiceTemplate for id={voice_over_id}")
            voice_template = db.query(VoiceTemplate).filter(VoiceTemplate.id == voice_over_id).first()
            if voice_template and voice_template.provider_voice_id:
                voice_id = voice_template.provider_voice_id
                logger.info(f"Found provider_voice_id: {voice_id} for template: {voice_template.name}")
            else:
                logger.warning(f"VoiceTemplate {voice_over_id} not found or missing provider_voice_id. Using fallback.")
        else:
            logger.warning("No DB session or voice_over_id provided for TTS. Using fallback.")

        logger.info(f"Converting text to speech with timestamps. Voice ID: {voice_id}")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Log key prefix for verification (masking the rest for security)
        key_prefix = self.api_key[:6] if self.api_key else "NONE"
        logger.info(f"Using API Key prefix: {key_prefix}...")
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            # The API returns audio as a base64 string
            audio_base64 = result.get("audio_base64")
            if not audio_base64:
                raise ValueError("No audio_base64 found in response")
            
            audio_data = base64.b64decode(audio_base64)
            alignment = result.get("alignment")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio file
            with open(output_path, "wb") as f:
                f.write(audio_data)
                
            # Save alignment/timestamps to a JSON file
            alignment_path = output_path.replace(".mp3", "_timestamps.json")
            with open(alignment_path, "w") as f:
                json.dump(alignment, f, indent=2)
                
            logger.info(f"TTS audio saved to {output_path}")
            logger.info(f"TTS alignment saved to {alignment_path}")
            
            return {
                "audio_path": output_path,
                "alignment_path": alignment_path
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"ElevenLabs API error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise Exception(f"Failed to generate TTS with timestamps: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in TTS generation: {str(e)}")
            raise

eleven_labs_service = ElevenLabsService()
