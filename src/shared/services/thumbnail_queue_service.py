import json
import base64
from azure.storage.queue import QueueClient
from src.shared.core.config import settings
from src.shared.core.logger import get_logger
from src.shared.models.enums import GenerateThumbnailProcess

logger = get_logger(__name__)


class ThumbnailQueueService:
    """Service for interacting with Azure Queue Storage for thumbnail generation"""
    
    def __init__(self):
        """Initialize Thumbnail Queue Service with connection string"""
        # Strip any whitespace from connection string
        self.connection_string = settings.THUMBNAIL_STORAGE_CONNECTION_STRING.strip() if settings.THUMBNAIL_STORAGE_CONNECTION_STRING else None
        self.queue_name = settings.THUMBNAIL_QUEUE_NAME
        self.account_name = getattr(settings, "THUMBNAIL_STORAGE_ACCOUNT_NAME", None)
        
        # Extract account key from connection string if not directly provided
        self.account_key = None
        if self.connection_string and "AccountKey=" in self.connection_string:
            key_part = self.connection_string.split("AccountKey=")[1].split(";")[0]
            self.account_key = key_part
            
        self.queue_account_url = (
            f"https://{self.account_name}.queue.core.windows.net" if self.account_name else None
        )
        
        if not self.connection_string:
            logger.error("THUMBNAIL_STORAGE_CONNECTION_STRING is not configured")
        if not self.queue_name:
            logger.error("THUMBNAIL_QUEUE_NAME is not configured")
        
    def send_message(self, message_data: dict) -> bool:
        """
        Send a message to Azure Queue Storage for thumbnail generation
        
        Args:
            message_data: Dictionary containing the message data
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        try:
            # Validate configuration
            if not self.queue_name:
                logger.error("Cannot send message: THUMBNAIL_QUEUE_NAME is not configured")
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
                    "Cannot send message: neither THUMBNAIL_STORAGE_ACCOUNT_NAME nor THUMBNAIL_STORAGE_CONNECTION_STRING is configured"
                )
                return False
            
            # Convert message data to JSON string
            message_content = json.dumps(message_data)
            
            # Base64 encode the message for Azure Queue Storage compatibility
            message_bytes = message_content.encode('utf-8')
            encoded_message = base64.b64encode(message_bytes).decode('utf-8')
            
            # Send message to queue
            queue_client.send_message(encoded_message)
            
            logger.info(f"Message sent to thumbnail queue '{self.queue_name}': {message_data}")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid connection string format: {str(e)}")
            logger.error("Please check your THUMBNAIL_STORAGE_CONNECTION_STRING in .env file")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to thumbnail queue: {str(e)}", exc_info=True)
            return False
    
    def send_thumbnail_generation_message(self, entity_id: int, process_type: GenerateThumbnailProcess = GenerateThumbnailProcess.VIDEO_THUMBNAIL) -> bool:
        """
        Send a thumbnail generation message to the queue
        
        Args:
            entity_id: ID of the video or clip record in database
            process_type: Type of thumbnail generation process (VIDEO_THUMBNAIL or CLIP_THUMBNAIL)
            
        Returns:
            bool: True if message sent successfully, False otherwise
        """
        message_data = {
            "thumbnailProcess": process_type.value,
            "entityId": entity_id
        }
        
        return self.send_message(message_data)


# Create a singleton instance
thumbnail_queue_service = ThumbnailQueueService()
