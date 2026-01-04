from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from src.shared.core.config import settings
from src.shared.core.logger import get_logger
import uuid
from typing import Optional

logger = get_logger(__name__)


class VideoUploadService:
    """Service for handling video upload operations with Azure Blob Storage"""

    def __init__(self):
        self.account_name = settings.AZURE_STORAGE_ACCOUNT_NAME
        self.account_key = settings.AZURE_STORAGE_ACCOUNT_KEY
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME
        
        # Initialize BlobServiceClient
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.account_name}.blob.core.windows.net",
            credential=self.account_key
        )

    def generate_upload_sas_url(
        self, 
        file_extension: str = "mp4",
        expiry_hours: int = 1,
        user_id: Optional[int] = None,
    ) -> dict:
        """
        Generate a SAS URL for uploading a video to Azure Blob Storage
        
        Args:
            file_extension: File extension for the video (default: mp4)
            expiry_hours: Number of hours the SAS token is valid (default: 1)
            
        Returns:
            dict: Contains blob_name, sas_url, and expiry_time
        """
        try:
            # Generate unique blob name with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            # Include user_id folder if provided
            if user_id is not None:
                blob_name = f"videos/{user_id}/{timestamp}_{unique_id}.{file_extension}"
            else:
                blob_name = f"videos/{timestamp}_{unique_id}.{file_extension}"
            
            # Calculate expiry time
            expiry_time = datetime.utcnow() + timedelta(hours=expiry_hours)
            
            # Generate SAS token with write permissions
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                account_key=self.account_key,
                container_name=self.container_name,
                blob_name=blob_name,
                permission=BlobSasPermissions(
                    read=False,
                    write=True,
                    delete=False,
                    create=True
                ),
                expiry=expiry_time
            )
            
            # Construct the full SAS URL
            sas_url = (
                f"https://{self.account_name}.blob.core.windows.net/"
                f"{self.container_name}/{blob_name}?{sas_token}"
            )
            
            logger.info(f"Generated SAS URL for blob: {blob_name}")
            
            return {
                "blob_name": blob_name,
                "sas_url": sas_url,
                "container_name": self.container_name,
                "expiry_time": expiry_time.isoformat(),
                "blob_url": f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
            }
            
        except Exception as e:
            logger.error(f"Error generating SAS URL: {str(e)}")
            raise

    def verify_blob_exists(self, blob_name: str) -> bool:
        """
        Verify if a blob exists in the container
        
        Args:
            blob_name: Name of the blob to check (with or without 'videos/' prefix)
            
        Returns:
            bool: True if blob exists, False otherwise
        """
        try:
            # Ensure blob_name includes the videos/ prefix if not already present
            if not blob_name.startswith("videos/"):
                blob_name = f"videos/{blob_name}"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.exists()
        except Exception as e:
            logger.error(f"Error checking blob existence: {str(e)}")
            return False
    
    def get_blob_url(self, blob_name: str) -> str:
        """
        Get the full URL of a blob
        
        Args:
            blob_name: Name of the blob (with or without 'videos/' prefix)
            
        Returns:
            str: Full URL of the blob
        """
        # Ensure blob_name includes the videos/ prefix if not already present
        if not blob_name.startswith("videos/"):
            blob_name = f"videos/{blob_name}"
            
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"


# Singleton instance
video_upload_service = VideoUploadService()
