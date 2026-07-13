import argparse
import sys
from src.shared.core.config import settings
from src.shared.core.logger import configure_application_logging, get_logger

# Configure logging centrally
configure_application_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger("WorkerLauncher")

# Registry of available workers (lazy loaded)
def get_p1_worker():
    from src.worker.workers.web_jobs.P1VideoProcessorWorker import P1VideoProcessorWorker
    return P1VideoProcessorWorker

def get_p2_worker():
    from src.worker.workers.web_jobs.P2TextToContentWorker import P2TextToContentWorker
    return P2TextToContentWorker

WORKER_REGISTRY = {
    "p1": get_p1_worker,
    "p2": get_p2_worker,
}

def main():
    parser = argparse.ArgumentParser(description="Kliper AI Background Worker Launcher")
    parser.add_argument(
        "--worker", "-w",
        type=str,
        default="p1",
        choices=WORKER_REGISTRY.keys(),
        help=f"Type of worker to run. Options: {', '.join(WORKER_REGISTRY.keys())} (default: p1)"
    )
    
    args = parser.parse_args()
    worker_getter = WORKER_REGISTRY.get(args.worker)
    
    if not worker_getter:
        logger.error(f"Unknown worker type: {args.worker}")
        sys.exit(1)
        
    logger.info(f"Starting worker of type: {args.worker}")
    worker_class = worker_getter()
    worker = worker_class()
    worker.start()

if __name__ == "__main__":
    main()
