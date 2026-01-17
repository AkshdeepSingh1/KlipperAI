from fastapi import APIRouter, HTTPException, Query, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
from src.api.modules.video_input_output.services import video_upload_service
from src.shared.core.logger import get_logger
from src.shared.core.database import get_db
from src.shared.services.queue_service import queue_service
from src.shared.models import Video

logger = get_logger(__name__)

router = APIRouter(
    prefix="/videoInputOutput",
    tags=["Video Input Output"]
)


class SASTokenResponse(BaseModel):
    """Response model for SAS token generation"""
    blob_name: str = Field(..., description="Unique name of the blob in Azure Storage")
    sas_url: str = Field(..., description="SAS URL for uploading the video")
    container_name: str = Field(..., description="Azure Storage container name")
    expiry_time: str = Field(..., description="Expiry time of the SAS token (ISO format)")
    blob_url: str = Field(..., description="Final blob URL (without SAS token)")
    message: str = Field(..., description="Success message with upload instructions")


@router.get("/generate-upload-url", response_model=SASTokenResponse)
async def generate_upload_url(
    request: Request,
    file_extension: str = Query(
        default="mp4",
        description="File extension for the video (e.g., mp4, mov, avi)",
        regex="^[a-zA-Z0-9]+$"
    ),
    expiry_hours: int = Query(
        default=1,
        ge=1,
        le=24,
        description="Number of hours the SAS token is valid (1-24 hours)"
    )
):
    """
    Generate a SAS URL for uploading a video directly to Azure Blob Storage
    
    This endpoint creates a unique blob name and generates a Shared Access Signature (SAS) URL
    that allows the frontend to upload videos directly to Azure Blob Storage without going
    through the backend server.
    
    **Usage:**
    1. Call this endpoint to get a SAS URL
    2. Use the returned `sas_url` to upload the video file using HTTP PUT request
    3. Set Content-Type header to the appropriate video MIME type
    4. Upload the video binary data in the request body
    
    **Example upload (using fetch):**
    ```javascript
    const response = await fetch(sas_url, {
        method: 'PUT',
        headers: {
            'x-ms-blob-type': 'BlockBlob',
            'Content-Type': 'video/mp4'
        },
        body: videoFile
    });
    ```
    
    Args:
        file_extension: File extension for the video (default: mp4)
        expiry_hours: Number of hours the SAS token is valid (1-24 hours, default: 1)
        
    Returns:
        SASTokenResponse: Contains the SAS URL and related information
        
    Raises:
        HTTPException: If SAS URL generation fails
    """
    try:
        logger.info(f"Generating SAS URL for file extension: {file_extension}, expiry: {expiry_hours}h")
        
        result = video_upload_service.generate_upload_sas_url(
            file_extension=file_extension,
            expiry_hours=expiry_hours,
            user_id=request.state.user_id,
        )
        
        return SASTokenResponse(
            blob_name=result["blob_name"],
            sas_url=result["sas_url"],
            container_name=result["container_name"],
            expiry_time=result["expiry_time"],
            blob_url=result["blob_url"],
            message=(
                "Use the 'sas_url' to upload your video file directly to Azure Blob Storage. "
                "Make a PUT request with 'x-ms-blob-type: BlockBlob' header and the video file as body."
            )
        )
        
    except Exception as e:
        logger.error(f"Failed to generate SAS URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.get("/verify-upload/{blob_name}")
async def verify_upload(
    blob_name: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Verify if a video has been successfully uploaded to Azure Blob Storage,
    create a video record in the database, and trigger video processing via Azure Queue.
    
    Args:
        blob_name: Name of the blob to verify
        request: Request object (contains authenticated user_id)
        db: Database session
        
    Returns:
        dict: Contains verification status, video ID, and processing status
        
    Raises:
        HTTPException: If verification fails
    """
    try:
        # Get user_id from authenticated request
        user_id = request.state.user_id
        
        logger.info(f"Verifying blob existence: {blob_name} for user_id={user_id}")
        
        # Verify blob exists in Azure Storage
        exists = video_upload_service.verify_blob_exists(blob_name, user_id=user_id)
        
        if not exists:
            return {
                "exists": False,
                "blob_name": blob_name,
                "message": "Video not found or upload incomplete"
            }
        
        # Get blob URL
        blob_url = video_upload_service.get_blob_url(blob_name, user_id=user_id)
        
        # Create video record in database
        video = Video(
            user_id=user_id,
            blob_url=blob_url,
            duration_seconds=None  # Will be populated during processing
        )
        
        db.add(video)
        db.commit()
        db.refresh(video)
        
        logger.info(f"Video record created with ID: {video.id}")
        
        # Send message to Azure Queue for video processing
        queue_success = queue_service.send_video_processing_message(
            video_id=video.id,
            blob_name=blob_name,
            blob_url=blob_url,
            user_id=user_id
        )
        
        if not queue_success:
            logger.warning(f"Failed to send queue message for video ID: {video.id}")
        
        return {
            "exists": True,
            "blob_name": blob_name,
            "video_id": video.id,
            "blob_url": blob_url,
            "queue_message_sent": queue_success,
            "message": "Video uploaded successfully and processing initiated" if queue_success else "Video uploaded but processing queue failed"
        }
            
    except Exception as e:
        logger.error(f"Failed to verify blob and process: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to verify upload: {str(e)}"
        )


@router.get("/processing-status/{video_id}")
async def get_processing_status(
    video_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get the processing status for a video including progress percentage.
    
    Args:
        video_id: ID of the video to check status for
        request: Request object (contains authenticated user_id)
        db: Database session
        
    Returns:
        dict: Contains video_id, status, current_step, progress_percentage, and timestamps
        
    Raises:
        HTTPException: If video not found or access denied
    """
    try:
        user_id = request.state.user_id
        logger.info(f"Getting processing status for video_id={video_id}, user_id={user_id}")
        
        result = video_upload_service.get_processing_status(video_id=video_id, db=db)
        
        # Optional: Verify user owns this video (security check)
        # Uncomment if you want to restrict access to video owner only
        # if result.get("user_id") and result["user_id"] != user_id:
        #     raise HTTPException(status_code=403, detail="Access denied")
        
        return result
        
    except ValueError as e:
        logger.error(f"Video not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get processing status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get processing status: {str(e)}"
        )


@router.get("/getClipsFromVideoId")
async def get_clips_from_video_id(
    request: Request,
    videoId: int = Query(..., description="Video ID"),
    db: Session = Depends(get_db),
):
    """Get all clips for a given video ID."""
    try:
        user_id = request.state.user_id
        result = video_upload_service.get_clips_from_video_id(
            video_id=videoId,
            user_id=user_id,
            db=db,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get clips for videoId={videoId}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get clips: {str(e)}")