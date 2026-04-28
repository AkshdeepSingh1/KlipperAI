from sqlalchemy import Column, BigInteger, Text, TIMESTAMP, Enum, Boolean, Numeric, func
from src.shared.core.database import Base
from src.shared.enums import RenderFormat


class VideoTemplate(Base):
    """Video template model"""
    __tablename__ = "video_templates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(Text, nullable=False)
    video_url = Column(Text, nullable=False)
    render_format = Column(
        Enum(
            RenderFormat,
            name="render_format_enum",
            native_enum=True,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=RenderFormat.VERTICAL_9_16,
        nullable=False,
    )
    default_music_url = Column(Text, nullable=True)
    default_music_volume = Column(Numeric(3, 2), nullable=False, server_default="0.20")
    allow_music_override = Column(Boolean, nullable=False, server_default="TRUE")
    override_music_url = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<VideoTemplate(id={self.id}, name={self.name}, category={self.category})>"
