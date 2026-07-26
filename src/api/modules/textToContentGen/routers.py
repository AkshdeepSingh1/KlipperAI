from datetime import datetime, date as date_type
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from src.shared.core.database import get_db
from src.shared.core.logger import get_logger
from src.shared.enums import ContentJobRequestProcessingStatus
from .schemas import (
    CreateContentJobRequest, 
    ContentJobRequestResponse, 
    VideoTemplateResponse, 
    VoiceTemplateResponse,
    RenderResponse,
    ScheduledContentResponse,
    DashboardStatsResponse,
    MonthlyScheduleResponse
)
from .service import ContentJobService

logger = get_logger(__name__)
router = APIRouter(prefix="/textToContentGen", tags=["Text to Content Generation"])


@router.post(
    "/CreateRequest",
    response_model=ContentJobRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create content job request",
    description="Create a new content job request for text-to-content generation"
)
async def create_content_job_request(
    request: CreateContentJobRequest,
    run_now: bool = Query(
        False,
        alias="runNow",
        description="If true, enqueue the job for immediate processing instead of waiting for the daily cron.",
    ),
    db: Session = Depends(get_db),
    http_request: Request = None,
):
    """
    Create a new content job request.

    - **title**: Title of the content job (required)
    - **source_type**: Source type - 'ai' or 'my_script' (default: my_script)
    - **prompt**: AI prompt for content generation (optional)
    - **user_script**: User provided script (optional)
    - **voice_over_id**: Voice over ID (optional)
    - **render_format**: Render format - 'vertical_9_16', 'square_1_1', or 'landscape_16_9' (default: vertical_9_16)
    - **template_id**: Template ID (optional)
    - **scheduled_at_utc**: UTC timestamp for the chosen day (ignored when runNow=true)
    - **runNow** (query): run immediately instead of on the scheduled day
    """
    try:
        # Get user_id from request state (set by auth middleware)
        user_id = getattr(http_request.state, "user_id", None)
        if not user_id:
            logger.error("User ID not found in request state")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        logger.info(f"Creating content job request for user {user_id}: {request.title}")

        # Create content job request
        content_job = ContentJobService.create_content_job_request(
            db=db,
            user_id=user_id,
            title=request.title,
            source_type=request.source_type.value,
            prompt=request.prompt,
            user_script=request.user_script,
            voice_over_id=request.voice_over_id,
            render_format=request.render_format.value,
            template_id=request.template_id,
            scheduled_at_utc=request.scheduled_at_utc,
            run_now=run_now,
        )

        logger.info(f"Content job request created successfully: {content_job.id}")

        return ContentJobRequestResponse.model_validate(content_job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating content job request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating content job request"
        )


@router.get(
    "/videoTemplate/get",
    response_model=List[VideoTemplateResponse],
    summary="Get all video templates",
    description="Retrieve all video templates from the database"
)
async def get_video_templates(
    db: Session = Depends(get_db),
):
    """
    Get all video templates.
    
    Returns a list of all available video templates with their details.
    """
    try:
        logger.info("Fetching all video templates")

        templates = ContentJobService.get_all_video_templates(db=db)

        logger.info(f"Retrieved {len(templates)} video templates")

        return [VideoTemplateResponse.model_validate(template) for template in templates]

    except Exception as e:
        logger.error(f"Error fetching video templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching video templates"
        )


@router.get(
    "/getAudio",
    response_model=List[VoiceTemplateResponse],
    summary="Get all active voice templates",
    description="Retrieve all active voice templates for text-to-speech"
)
async def get_audio_templates(
    db: Session = Depends(get_db),
):
    """
    Get all active voice templates.
    
    Returns a list of all available voice templates with their details.
    """
    try:
        logger.info("Fetching all active voice templates")

        voices = ContentJobService.get_all_voice_templates(db=db)

        logger.info(f"Retrieved {len(voices)} voice templates")

        return [VoiceTemplateResponse.model_validate(voice) for voice in voices]

    except Exception as e:
        logger.error(f"Error fetching voice templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching voice templates"
        )


@router.get(
    "/getLatestRenders",
    response_model=List[RenderResponse],
    summary="Get latest completed renders",
    description="Retrieve the latest completed renders with pagination support"
)
async def get_latest_renders(
    last_id: Optional[int] = Query(None, alias="Id"),
    quantity: int = Query(5),
    db: Session = Depends(get_db),
):
    """
    Get latest completed renders.
    
    - **last_id** (Id): The ID to start fetching renders before (for pagination).
    - **quantity**: Number of renders to fetch (default: 5).
    """
    try:
        logger.info(f"Fetching latest renders: last_id={last_id}, quantity={quantity}")

        renders = ContentJobService.get_latest_renders(db=db, last_id=last_id, quantity=quantity)

        response = []
        for render in renders:
            # Construct the full Azure Blob URL
            # Format: https://learningqueues.blob.core.windows.net/processor2-outputs/{job_id}/{guid}
            # render.output_url contains the guid (e.g. uuid.mp4)
            url = f"https://learningqueues.blob.core.windows.net/processor2-outputs/{render.id}/{render.output_url}"
            response.append(RenderResponse(
                url=url, 
                jobId=render.id,
                contentTitle=render.title,
                scheduledDate=render.scheduled_at_utc
            ))

        logger.info(f"Retrieved {len(response)} renders")
        return response

    except Exception as e:
        logger.error(f"Error fetching latest renders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching latest renders"
        )


@router.get(
    "/getScheduledContent",
    response_model=List[ScheduledContentResponse],
    summary="Get scheduled content for a specific date",
    description="Retrieve all content job requests scheduled for a specific date for the authenticated user"
)
async def get_scheduled_content(
    target_date: date_type = Query(..., alias="date", description="Target date (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
    http_request: Request = None,
):
    """
    Get scheduled content for a specific date.
    
    - **date**: The date to fetch scheduled content for.
    """
    try:
        # Get user_id from request state (set by auth middleware)
        user_id = getattr(http_request.state, "user_id", None)
        if not user_id:
            logger.error("User ID not found in request state")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        logger.info(f"Fetching scheduled content for user {user_id} on {target_date}")

        jobs = ContentJobService.get_scheduled_content(db=db, user_id=user_id, target_date=target_date)

        response = []
        for job in jobs:
            job_data = ScheduledContentResponse.model_validate(job)
            
            # Construct full URL if completed and GUID (output_url) is present
            if job.processing_status == ContentJobRequestProcessingStatus.COMPLETED and job.output_url:
                job_data.full_output_url = f"https://learningqueues.blob.core.windows.net/processor2-outputs/{job.id}/{job.output_url}"
            
            response.append(job_data)

        logger.info(f"Retrieved {len(response)} scheduled items")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scheduled content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching scheduled content"
        )


@router.get(
    "/getDashboardStats",
    response_model=DashboardStatsResponse,
    summary="Get dashboard statistics",
    description="Retrieve counts of scheduled and in-progress jobs for the dashboard"
)
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    http_request: Request = None,
):
    """
    Get dashboard statistics for the authenticated user.
    """
    try:
        # Get user_id from request state (set by auth middleware)
        user_id = getattr(http_request.state, "user_id", None)
        if not user_id:
            logger.error("User ID not found in request state")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        logger.info(f"Fetching dashboard stats for user {user_id}")

        stats = ContentJobService.get_dashboard_stats(db=db, user_id=user_id)

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching dashboard stats"
        )


@router.get(
    "/getMonthlySchedule",
    response_model=List[MonthlyScheduleResponse],
    summary="Get monthly schedule for a user",
    description="Retrieve all content job requests scheduled for a specific month for the authenticated user"
)
async def get_monthly_schedule(
    year: int = Query(..., description="Year (e.g., 2026)"),
    month: int = Query(..., description="Month (1-12)"),
    db: Session = Depends(get_db),
    http_request: Request = None,
):
    """
    Get monthly schedule for a user.
    """
    try:
        # Get user_id from request state (set by auth middleware)
        user_id = getattr(http_request.state, "user_id", None)
        if not user_id:
            logger.error("User ID not found in request state")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )

        logger.info(f"Fetching monthly schedule for user {user_id} for {year}-{month}")

        jobs = ContentJobService.get_monthly_schedule(db=db, user_id=user_id, year=year, month=month)

        response = []
        for job in jobs:
            # Map source_type to 'ai' or 'self'
            job_type = "ai" if job.source_type == "ai" else "self"
            
            # Use raw enum value for status (e.g. 'completed', 'processing', 'scheduled', 'failed')
            job_status = job.processing_status.value if hasattr(job.processing_status, 'value') else job.processing_status

            response.append(MonthlyScheduleResponse(
                id=str(job.id),
                date=job.scheduled_at_utc,
                type=job_type,
                title=job.title,
                status=job_status
            ))

        logger.info(f"Retrieved {len(response)} monthly schedule items")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching monthly schedule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while fetching monthly schedule"
        )