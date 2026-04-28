from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from decimal import Decimal
from src.shared.enums import ContentJobRequestSourceType, RenderFormat


class CreateContentJobRequest(BaseModel):
    """Request schema for creating a content job request"""
    title: str = Field(..., description="Title of the content job")
    source_type: ContentJobRequestSourceType = Field(
        default=ContentJobRequestSourceType.MY_SCRIPT,
        description="Source type (ai or my_script)"
    )
    prompt: Optional[str] = Field(None, description="AI prompt for content generation")
    user_script: Optional[str] = Field(None, description="User provided script")
    voice_over_id: Optional[int] = Field(None, description="Voice over ID")
    render_format: RenderFormat = Field(
        default=RenderFormat.VERTICAL_9_16,
        description="Render format (vertical_9_16, square_1_1, landscape_16_9)"
    )
    template_id: Optional[int] = Field(None, description="Template ID")


class ContentJobRequestResponse(BaseModel):
    """Response schema for content job request"""
    id: int = Field(..., description="Job request ID")
    user_id: int = Field(..., description="User ID")
    title: str = Field(..., description="Title of the content job")
    source_type: str = Field(..., description="Source type")
    prompt: Optional[str] = Field(None, description="AI prompt")
    user_script: Optional[str] = Field(None, description="User provided script")
    generated_script: Optional[str] = Field(None, description="Generated script")
    voice_over_id: Optional[int] = Field(None, description="Voice over ID")
    render_format: str = Field(..., description="Render format")
    template_id: Optional[int] = Field(None, description="Template ID")
    scheduled_at_utc: datetime = Field(..., description="Scheduled timestamp")
    processing_status: str = Field(..., description="Processing status")
    created_at_utc: datetime = Field(..., description="Creation timestamp")
    updated_at_utc: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class VideoTemplateResponse(BaseModel):
    """Response schema for video template"""
    id: int = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Template category")
    video_url: str = Field(..., description="Video URL")
    render_format: str = Field(..., description="Render format")
    default_music_url: Optional[str] = Field(None, description="Default music URL")
    default_music_volume: Decimal = Field(..., description="Default music volume")
    allow_music_override: bool = Field(..., description="Allow music override")
    override_music_url: Optional[str] = Field(None, description="Override music URL")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True
