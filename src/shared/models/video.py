from sqlalchemy import Column, BigInteger, Integer, Text, TIMESTAMP, ForeignKey, func, Enum
from src.shared.core.database import Base
from src.shared.enums.processing_status import ProcessingStatus


class Video(Base):
    """Video model"""
    __tablename__ = "videos"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    blob_url = Column(Text, nullable=False)
    thumbnail_url = Column(Text, nullable=True)
    processing_status = Column(Enum(ProcessingStatus, name='processing_status', native_enum=True, values_callable=lambda x: [e.value for e in x]), nullable=True, default=ProcessingStatus.PENDING)
    duration_seconds = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    def __repr__(self):
        return f"<Video(id={self.id}, user_id={self.user_id})>"
