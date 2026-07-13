"""
Cron Job Launcher
------------------
Entry point for all time-triggered cron jobs in the Kliper AI backend.

Usage:
    python cron.py --job daily_dispatcher

Available jobs are registered in CRON_REGISTRY below. Add new cron job
classes there to make them launchable without touching this file (OCP).
"""

import argparse
import sys

from src.shared.core.config import settings
from src.shared.core.logger import configure_application_logging, get_logger
from src.worker.workers.cron_jobs.DailyJobDispatcherCronJob import (
    DailyJobDispatcherCronJob,
)

# Configure logging centrally (mirrors worker.py pattern)
configure_application_logging(level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger("CronLauncher")

# -----------------------------------------------------------------------
# Registry — add new cron jobs here, nothing else needs to change (OCP)
# -----------------------------------------------------------------------
CRON_REGISTRY = {
    "daily_dispatcher": DailyJobDispatcherCronJob,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kliper AI Cron Job Launcher")
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        default="daily_dispatcher",
        choices=CRON_REGISTRY.keys(),
        help=(
            f"Cron job to run. Options: {', '.join(CRON_REGISTRY.keys())} "
            "(default: daily_dispatcher)"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run the job a single time and exit, instead of looping. Use this when "
            "an external scheduler owns the timing (e.g. a Kubernetes CronJob)."
        ),
    )

    args = parser.parse_args()
    job_class = CRON_REGISTRY.get(args.job)

    if not job_class:
        logger.error(f"Unknown cron job: {args.job}")
        sys.exit(1)

    job = job_class()

    if args.once:
        if not hasattr(job, "run_once"):
            logger.error(f"Cron job '{args.job}' does not support --once")
            sys.exit(1)
        logger.info(f"Running cron job once: {args.job}")
        job.run_once()
    else:
        logger.info(f"Starting cron job: {args.job}")
        job.start()


if __name__ == "__main__":
    main()
