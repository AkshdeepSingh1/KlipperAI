from sqlalchemy import Column, BigInteger, Text, TIMESTAMP, ForeignKey, Enum, func, String, Numeric
from src.shared.core.database import Base
from src.shared.enums import ProcessingStatus


class ProcessingJob(Base):
    """Processing job model"""
    __tablename__ = "processing_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    video_id = Column(BigInteger, ForeignKey("videos.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    status = Column(
        Enum(
            ProcessingStatus,
            name="processing_status",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=ProcessingStatus.PENDING,
    )
    current_step = Column(String(50), nullable=True)
    progress_percentage = Column(Numeric(5, 2), nullable=False, server_default="0.00")
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    completed_at = Column(TIMESTAMP, nullable=True)

    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, status={self.status})>"
