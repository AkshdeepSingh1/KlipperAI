"""
Test Worker Functions
Allows local testing of worker processors without needing Azure Queue

Run this from the project root:
    python -m tests.test_worker
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.core.logger import configure_application_logging, get_logger
from src.shared.core.config import settings
from src.worker.handlers.video_processor import VideoProcessor

configure_application_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger(__name__)


def test_video_processor():
    """
    Test the video processor locally
    
    This simulates what the worker would do when it receives a message from the queue.
    You can run this directly to test video processing logic without deploying.
    
    Usage:
        python -m tests.test_worker
        
    Or with pytest:
        python -m pytest tests/test_worker.py::test_video_processor -v
    """
    logger.info("=" * 60)
    logger.info("Testing Video Processor Locally")
    logger.info("=" * 60)
    
    # Create processor instance
    processor = VideoProcessor()
    
    # Test data - replace with actual values from your database
    test_video_id = 9
    test_blob_name = "20260117_074655_9e218018.mp4"
    test_blob_url = "https://learningqueues.blob.core.windows.net/raw-videos/videos/1/20260117_074655_9e218018.mp4"
    test_user_id = 1
    
    logger.info(f"Processing test video:")
    logger.info(f"  Video ID: {test_video_id}")
    logger.info(f"  Blob Name: {test_blob_name}")
    logger.info(f"  Blob URL: {test_blob_url}")
    logger.info(f"  User ID: {test_user_id}")
    
    try:
        # Trigger the processor
        processor.process_video(
            video_id=test_video_id,
            blob_name=test_blob_name,
            blob_url=test_blob_url,
            user_id=test_user_id
        )
        
        logger.info("=" * 60)
        logger.info("✓ Video processing test completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ Video processing test failed: {e}", exc_info=True)
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    """
    Run this file directly to test the worker processor locally:
    
        python tests/test_worker.py
    
    This is great for development - you can test your video processing
    logic without needing to:
    - Deploy the worker
    - Send messages to Azure Queue
    - Run the full worker loop
    
    Just update the test data above and run this script!
    """
    test_video_processor()
