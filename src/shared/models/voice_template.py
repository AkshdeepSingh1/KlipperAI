from sqlalchemy import Column, BigInteger, Text, TIMESTAMP, Boolean, Integer, func
from src.shared.core.database import Base

class VoiceTemplate(Base):
    """Voice template model for text-to-speech providers"""
    __tablename__ = "voice_templates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    
    name = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    accent = Column(Text, nullable=True)
    tone = Column(Text, nullable=True)
    
    provider = Column(Text, nullable=False)                # e.g. 'elevenlabs', 'azure', 'google'
    provider_voice_id = Column(Text, nullable=False)       # voice id from provider
    
    preview_audio_url = Column(Text, nullable=True)        # CDN/Blob URL
    in_house_db_url = Column(Text, nullable=True)          # internal storage reference
    
    is_active = Column(Boolean, nullable=False, server_default="TRUE")
    sort_order = Column(Integer, nullable=False, server_default="0")
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<VoiceTemplate(id={self.id}, name={self.name}, provider={self.provider})>"
