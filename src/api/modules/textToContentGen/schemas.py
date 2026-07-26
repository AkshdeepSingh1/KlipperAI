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
    scheduled_at_utc: Optional[datetime] = Field(
        None,
        description="UTC timestamp for the day this should run. Ignored when runNow=true."
    )


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


class VoiceTemplateResponse(BaseModel):
    """Response schema for voice template"""
    id: int = Field(..., description="Voice template ID")
    name: str = Field(..., description="Voice name")
    category: str = Field(..., description="Voice category")
    accent: Optional[str] = Field(None, description="Voice accent")
    tone: Optional[str] = Field(None, description="Voice tone")
    url: Optional[str] = Field(None, validation_alias="in_house_db_url", description="Voice preview/base URL")
    sort_order: int = Field(0, description="Sort order")

    class Config:
        from_attributes = True
        populate_by_name = True


class RenderResponse(BaseModel):
    """Response schema for a single render"""
    url: str = Field(..., description="Full Azure Blob URL of the rendered video")
    jobId: int = Field(..., description="Job request ID", alias="jobId")
    contentTitle: str = Field(..., description="Title of the content", alias="contentTitle")
    scheduledDate: datetime = Field(..., description="Scheduled date of the render", alias="scheduledDate")

    class Config:
        populate_by_name = True


class ScheduledContentResponse(ContentJobRequestResponse):
    """Response schema for scheduled content with optional full URL"""
    full_output_url: Optional[str] = Field(None, description="Full Azure Blob URL if processed")

    class Config:
        from_attributes = True


class DashboardStatsResponse(BaseModel):
    """Response schema for dashboard statistics"""
    scheduledThisMonth: int = Field(..., description="Count of jobs scheduled this month")
    inProgress: int = Field(..., description="Count of jobs currently in progress")
    scheduledToday: int = Field(..., description="Count of jobs scheduled for today")


class MonthlyScheduleResponse(BaseModel):
    """Response schema for monthly schedule item"""
    id: str = Field(..., description="Job request ID")
    date: datetime = Field(..., description="Scheduled timestamp")
    type: str = Field(..., description="Source type (ai or self)")
    title: str = Field(..., description="Title of the content")
    status: str = Field(..., description="Processing status")
