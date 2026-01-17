"""
Background Worker for Kliper AI
Polls Azure Queue Storage and processes video jobs
"""
import time
import json
from azure.storage.queue import QueueClient
from src.shared.core.config import settings
from src.shared.core.logger import configure_application_logging, get_logger
from src.shared.core.database import SessionLocal
from src.worker.handlers.video_processor import VideoProcessor

# Configure logging
configure_application_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger(__name__)


class Worker:
    """Background worker that processes jobs from Azure Queue"""
    
    def __init__(self):
        self.queue_client = QueueClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.queue.core.windows.net",
            queue_name=settings.AZURE_QUEUE_NAME,
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
        )
        self.video_processor = VideoProcessor()
        self.running = True
        
    def process_message(self, message):
        """Process a single message from the queue"""
        try:
            # Parse message content
            message_data = json.loads(message.content)
            action = message_data.get("action")
            
            logger.info(f"Processing message: {action} - {message_data}")
            
            # Route to appropriate handler based on action
            if action == "process_video":
                video_id = message_data.get("video_id")
                blob_name = message_data.get("blob_name")
                blob_url = message_data.get("blob_url")
                user_id = message_data.get("user_id")
                
                # Process the video
                self.video_processor.process_video(
                    video_id=video_id,
                    blob_name=blob_name,
                    blob_url=blob_url,
                    user_id=user_id
                )
            else:
                logger.warning(f"Unknown action: {action}")
            
            # Delete message from queue after successful processing
            self.queue_client.delete_message(message)
            logger.info(f"Message processed and deleted: {message.id}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            # Delete malformed message
            self.queue_client.delete_message(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Message will remain in queue and be retried
    
    def run(self):
        """Main worker loop - polls queue and processes messages"""
        logger.info("Worker started - polling Azure Queue for messages...")
        logger.info(f"Queue: {settings.AZURE_QUEUE_NAME}")
        
        try:
            while self.running:
                try:
                    # Receive messages from queue (max 10 at a time)
                    messages = self.queue_client.receive_messages(
                        messages_per_page=10,
                        visibility_timeout=300  # 5 minutes to process
                    )
                    
                    message_count = 0
                    for message in messages:
                        message_count += 1
                        self.process_message(message)
                    
                    if message_count > 0:
                        logger.info(f"Processed {message_count} messages")
                    
                    # Sleep briefly before next poll
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    logger.info("Worker interrupted by user")
                    self.running = False
                    break
                    
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}", exc_info=True)
                    time.sleep(5)  # Wait before retrying
                    
        finally:
            logger.info("Worker stopped")


if __name__ == "__main__":
    worker = Worker()
    worker.run()
