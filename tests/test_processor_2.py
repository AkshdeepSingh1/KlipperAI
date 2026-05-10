import logging
import sys
import os

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.worker.handlers.text_to_content_processor import TextToContentProcessor
from src.shared.core.logger import get_logger
from src.shared.core.database import SessionLocal
from src.shared.models import ContentJobRequest
from src.shared.enums import ContentJobRequestProcessingStatus

# Setup logger for the test script
logger = get_logger("test_processor")

def test_processor_run(request_id: int):
    """
    One method to run the processor logic for a specific request_id.
    """
    logger.info(f"Starting test for request_id={request_id}")
    
    # 1. Reset status to SCHEDULED to allow re-testing
    db = SessionLocal()
    try:
        job_request = db.query(ContentJobRequest).filter(ContentJobRequest.id == request_id).first()
        if job_request:
            if job_request.processing_status != ContentJobRequestProcessingStatus.SCHEDULED:
                logger.info(f"Resetting request_id={request_id} from {job_request.processing_status} to SCHEDULED for re-test.")
                job_request.processing_status = ContentJobRequestProcessingStatus.SCHEDULED
                db.commit()
        else:
            logger.error(f"Request ID {request_id} not found in database.")
            return
    except Exception as e:
        logger.error(f"Error resetting status: {e}")
        db.rollback()
    finally:
        db.close()

    # 2. Run processor
    try:
        processor = TextToContentProcessor()
        processor.process_request(request_id)
        logger.info(f"Successfully finished processing for request_id={request_id}")
    except Exception as e:
        logger.error(f"Failed to process request_id={request_id}: {e}", exc_info=True)

if __name__ == "__main__":
    # Set the request_id you want to test here
    # Based on current database state, 1 is a valid ID
    target_id = 11
    
    if len(sys.argv) > 1:
        try:
            target_id = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid request_id provided: {sys.argv[1]}")
            sys.exit(1)
            
    test_processor_run(target_id)
