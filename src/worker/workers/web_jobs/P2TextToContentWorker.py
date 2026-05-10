import time
import json
from azure.storage.queue import QueueClient, TextBase64EncodePolicy, TextBase64DecodePolicy
from src.shared.core.config import settings
from src.worker.workers.base_worker import BaseWorker
from src.worker.handlers.text_to_content_processor import TextToContentProcessor

class P2TextToContentWorker(BaseWorker):
    """
    Polls Azure Queue Storage and processes text-to-content jobs.
    Listens to the AZURE_QUEUE_P2_TEXT_TO_CONTENT_WORKER queue.
    """
    name: str = "P2TextToContentWorker"

    def setup(self):
        super().setup()
        self.queue_client = QueueClient(
            account_url=f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.queue.core.windows.net",
            queue_name=settings.AZURE_QUEUE_P2_TEXT_TO_CONTENT_WORKER,
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
            message_encode_policy=TextBase64EncodePolicy(),
            message_decode_policy=TextBase64DecodePolicy(),
        )
        self.processor = TextToContentProcessor()
        self.logger.info(f"Initialized QueueClient (with Base64 policy) and Processor for: {settings.AZURE_QUEUE_P2_TEXT_TO_CONTENT_WORKER}")

    def run(self):
        self.logger.info(f"Worker started - polling Azure Queue: {settings.AZURE_QUEUE_P2_TEXT_TO_CONTENT_WORKER}")
        
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
                
                # Sleep briefly before next poll to avoid high CPU/request costs
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying

    def process_message(self, message):
        """Process a single message from the queue"""
        try:
            self.logger.info(f"Raw message content: '{message.content}'")
            message_data = json.loads(message.content)
            request_id = message_data.get("requestId")
            
            if not request_id:
                self.logger.warning(f"Message missing requestId: {message_data}")
                self.queue_client.delete_message(message)
                return

            self.logger.info(f"Processing requestId: {request_id}")
            
            # Delegate to the orchestrator processor
            self.processor.process_request(request_id)
            
            # Only delete if successful (or if we want to ignore failures)
            self.queue_client.delete_message(message)
            self.logger.info(f"Message processed and deleted: {message.id}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message JSON: {e}")
            self.queue_client.delete_message(message)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
