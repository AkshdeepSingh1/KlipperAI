import json
from azure.storage.queue import QueueClient
from src.shared.core.config import settings
from src.shared.core.logger import get_logger

logger = get_logger(__name__)


class QueueService:
    """Service for interacting with Azure Queue Storage"""
    
    def __init__(self):
        """Initialize Queue Service with connection string"""
        # Strip any whitespace from connection string
        self.connection_string = settings.AZURE_STORAGE_CONNECTION_STRING.strip() if settings.AZURE_STORAGE_CONNECTION_STRING else None
        self.queue_name = settings.AZURE_QUEUE_NAME
        self.account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", None)
        self.account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", None)
        self.queue_account_url = (
            f"https://{self.account_name}.queue.core.windows.net" if self.account_name else None
        )
        
        if not self.connection_string:
            logger.error("AZURE_STORAGE_CONNECTION_STRING is not configured")
        if not self.queue_name:
            logger.error("AZURE_QUEUE_NAME is not configured")
        
    def send_message(self, message_data: dict) -> bool:
        """
        Send a message to Azure Queue Storage
        
        Args:
            message_data: Dictionary containing the message data
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            # Validate configuration
            if not self.queue_name:
                logger.error("Cannot send message: AZURE_QUEUE_NAME is not configured")
                return False
            
            # Create queue client
            # Prefer account_url + account_key (more robust than connection string and avoids base64 padding issues)
            if self.queue_account_url and self.account_key:
                queue_client = QueueClient(
                    account_url=self.queue_account_url,
                    queue_name=self.queue_name,
                    credential=self.account_key,
                )
            elif self.connection_string:
                queue_client = QueueClient.from_connection_string(
                    self.connection_string,
                    self.queue_name,
                )
            else:
                logger.error(
                    "Cannot send message: neither AZURE_STORAGE_ACCOUNT_NAME/AZURE_STORAGE_ACCOUNT_KEY nor AZURE_STORAGE_CONNECTION_STRING is configured"
                )
                return False
            
            # Convert message data to JSON string
            message_content = json.dumps(message_data)
            
            # Send message to queue
            queue_client.send_message(message_content)
            
            logger.info(f"Message sent to queue '{self.queue_name}': {message_data}")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid connection string format: {str(e)}")
            logger.error("Please check your AZURE_STORAGE_CONNECTION_STRING in .env file")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to queue: {str(e)}", exc_info=True)
            return False
    
    def send_video_processing_message(
        self,
        video_id: int,
        blob_name: str,
        blob_url: str,
        user_id: int = None
    ) -> bool:
        """
        Send a video processing message to the queue
        
        Args:
            video_id: ID of the video record in database
            blob_name: Name of the blob in Azure Storage
            blob_url: Full URL of the blob
            user_id: Optional user ID who uploaded the video
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        message_data = {
            "video_id": video_id,
            "blob_name": blob_name,
            "blob_url": blob_url,
            "user_id": user_id,
            "action": "process_video"
        }
        
        return self.send_message(message_data)


# Create a singleton instance
queue_service = QueueService()
