from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List
from src.shared.core.database import get_db
from src.shared.core.logger import get_logger
from .schemas import CreateContentJobRequest, ContentJobRequestResponse, VideoTemplateResponse
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
