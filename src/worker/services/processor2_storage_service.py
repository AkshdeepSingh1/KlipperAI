import os
from azure.storage.blob import BlobServiceClient, ContentSettings
from src.shared.core.config import settings
from src.shared.core.logger import get_logger

logger = get_logger(__name__)

class Processor2StorageService:
    """Handles uploading Processor2 outputs to Azure Blob Storage."""

    def __init__(self):
        # The user mentioned the account is 'learningqueues'
        # We use the configured account details from settings.
        self.account_name = settings.AZURE_STORAGE_ACCOUNT_NAME
        self.account_key = settings.AZURE_STORAGE_ACCOUNT_KEY
        self.container_name = "processor2-outputs"

        self._blob_service_client = BlobServiceClient(
            account_url=f"https://{self.account_name}.blob.core.windows.net",
            credential=self.account_key,
        )

    def upload_video(
        self,
        local_path: str,
        request_id: int,
        filename: str
    ) -> str:
        """
        Upload the final video to Azure Blob Storage and return the URL.
        """
        blob_name = f"{request_id}/{filename}"
        
        blob_client = self._blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name,
        )

        logger.info(f"Uploading {local_path} to blob {blob_name} in container {self.container_name}")

        with open(local_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type="video/mp4"),
            )

        blob_url = (
            f"https://{self.account_name}.blob.core.windows.net/"
            f"{self.container_name}/{blob_name}"
        )
        logger.info(f"Uploaded video to {blob_url}")
        return blob_url

processor2_storage_service = Processor2StorageService()
