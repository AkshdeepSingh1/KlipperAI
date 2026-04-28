from sqlalchemy import Column, BigInteger, Text, TIMESTAMP, ForeignKey, Enum, func
from src.shared.core.database import Base
from src.shared.enums import (
    ContentJobRequestProcessingStatus,
    ContentJobRequestSourceType,
    RenderFormat,
)


class ContentJobRequest(Base):
    """Content job request model"""
    __tablename__ = "content_job_requests"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    title = Column(Text, nullable=False)
    source_type = Column(
        Enum(
            ContentJobRequestSourceType,
            name="content_job_request_source_type",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=ContentJobRequestSourceType.MY_SCRIPT,
        nullable=False,
    )
    prompt = Column(Text, nullable=True)
    user_script = Column(Text, nullable=True)
    generated_script = Column(Text, nullable=True)
    voice_over_id = Column(BigInteger, nullable=True)
    render_format = Column(
        Enum(
            RenderFormat,
            name="render_format_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=RenderFormat.VERTICAL_9_16,
        nullable=False,
    )
    template_id = Column(BigInteger, nullable=True)
    scheduled_at_utc = Column(TIMESTAMP(timezone=True), nullable=False)
    processing_status = Column(
        Enum(
            ContentJobRequestProcessingStatus,
            name="content_job_request_processing_status",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=ContentJobRequestProcessingStatus.SCHEDULED,
        nullable=False,
    )
    created_at_utc = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at_utc = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<ContentJobRequest(id={self.id}, title={self.title}, status={self.processing_status})>"
