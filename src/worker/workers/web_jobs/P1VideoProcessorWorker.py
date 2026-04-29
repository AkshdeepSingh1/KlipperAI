import time
import json
from azure.storage.queue import QueueClient
from src.shared.core.config import settings
from src.worker.handlers.video_processor import VideoProcessor
from src.worker.workers.base_worker import BaseWorker

class P1VideoProcessorWorker(BaseWorker):
    """
    Polls Azure Queue Storage and processes video jobs.
    Migrated from the original root worker.py logic.
    """
    name: str = "P1VideoProcessorWorker"

    def setup(self):
        super().setup()
        self.queue_client = QueueClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.queue.core.windows.net",
            queue_name=settings.AZURE_QUEUE_NAME,
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
        )
        self.video_processor = VideoProcessor()

    def run(self):
        self.logger.info(f"Worker started - polling Azure Queue: {settings.AZURE_QUEUE_NAME}")
        
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
                    self.logger.info(f"Processed {message_count} messages")
                
                # Sleep briefly before next poll
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying

    def process_message(self, message):
        """Process a single message from the queue"""
        try:
            message_data = json.loads(message.content)
            action = message_data.get("action")
            
            self.logger.info(f"Processing message: {action} - {message_data}")
            
            if action == "process_video":
                video_id = message_data.get("video_id")
                blob_name = message_data.get("blob_name")
                blob_url = message_data.get("blob_url")
                user_id = message_data.get("user_id")
                
                self.video_processor.process_video(
                    video_id=video_id,
                    blob_name=blob_name,
                    blob_url=blob_url,
                    user_id=user_id
                )
            else:
                self.logger.warning(f"Unknown action: {action}")
            
            self.queue_client.delete_message(message)
            self.logger.info(f"Message processed and deleted: {message.id}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message JSON: {e}")
            self.queue_client.delete_message(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
