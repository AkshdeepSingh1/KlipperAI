"""
Scheduled Jobs Dispatcher Service
----------------------------------
Responsible for:
  1. Querying the `content_job_requests` table for all rows whose
     `scheduled_at_utc` falls on today's date (UTC).
  2. Enqueuing each eligible row into the Azure Queue
     `p2-text-to-content-worker` so the P2 worker can pick them up.

Follows Single Responsibility Principle — this service does ONE thing:
fetch today's scheduled jobs and push them to the queue.
"""

import json
from datetime import datetime, timezone, date
from typing import List

from azure.storage.queue import QueueClient, TextBase64EncodePolicy
from sqlalchemy.orm import Session
from sqlalchemy import cast, Date

from src.shared.core.config import settings
from src.shared.core.database import SessionLocal
from src.shared.core.logger import get_logger
from src.shared.enums import ContentJobRequestProcessingStatus
from src.shared.models import ContentJobRequest

logger = get_logger(__name__)


class ScheduledJobsDispatcherService:
    """
    Fetches all content job requests scheduled for today (UTC) and
    dispatches them to the P2 text-to-content Azure Queue.

    Single Responsibility: query → filter → enqueue.
    Open/Closed: extend by subclassing or injecting a different queue client.
    """

    def __init__(self) -> None:
        self._queue_client: QueueClient = self._build_queue_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dispatch_todays_jobs(self) -> int:
        """
        Entry point called by the cron worker.

        Returns:
            int: Number of jobs successfully enqueued.
        """
        db: Session = SessionLocal()
        try:
            today_utc = datetime.now(timezone.utc).date()
            logger.info(
                f"[ScheduledJobsDispatcherService] Running dispatch for date: {today_utc}"
            )

            jobs = self._fetch_scheduled_jobs(db, today_utc)
            logger.info(
                f"[ScheduledJobsDispatcherService] Found {len(jobs)} job(s) scheduled for {today_utc}"
            )

            enqueued_count = 0
            for job in jobs:
                success = self._enqueue_job(job)
                if success:
                    enqueued_count += 1

            logger.info(
                f"[ScheduledJobsDispatcherService] Enqueued {enqueued_count}/{len(jobs)} job(s)"
            )
            return enqueued_count

        except Exception as exc:
            logger.error(
                f"[ScheduledJobsDispatcherService] Fatal error during dispatch: {exc}",
                exc_info=True,
            )
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_scheduled_jobs(
        self, db: Session, target_date: date
    ) -> List[ContentJobRequest]:
        """
        SELECT * FROM content_job_requests
        WHERE DATE(scheduled_at_utc) = :target_date
          AND processing_status = 'scheduled'
        ORDER BY id DESC

        Only rows still in SCHEDULED status are returned so we never
        accidentally re-dispatch a job that is already PROCESSING or COMPLETED.
        """
        jobs = (
            db.query(ContentJobRequest)
            .filter(
                cast(ContentJobRequest.scheduled_at_utc, Date) == target_date,
                ContentJobRequest.processing_status
                == ContentJobRequestProcessingStatus.SCHEDULED,
            )
            .order_by(ContentJobRequest.id.desc())
            .all()
        )
        return jobs

    def _enqueue_job(self, job: ContentJobRequest) -> bool:
        """
        Serialize the job request into a JSON message and push it to the
        Azure Queue.  Returns True on success, False on failure (so the
        caller can keep a count without crashing the whole batch).
        """
        try:
            payload = json.dumps({"requestId": job.id})
            self._queue_client.send_message(payload)
            logger.info(
                f"[ScheduledJobsDispatcherService] Enqueued job id={job.id} "
                f"(scheduled_at_utc={job.scheduled_at_utc})"
            )
            return True
        except Exception as exc:
            logger.error(
                f"[ScheduledJobsDispatcherService] Failed to enqueue job id={job.id}: {exc}",
                exc_info=True,
            )
            return False

    @staticmethod
    def _build_queue_client() -> QueueClient:
        """
        Factory method — isolates Azure SDK construction so it can be
        swapped out in tests (Dependency Inversion Principle).
        """
        return QueueClient(
            account_url=(
                f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}"
                ".queue.core.windows.net"
            ),
            queue_name=settings.AZURE_QUEUE_P2_TEXT_TO_CONTENT_WORKER,
            credential=settings.AZURE_STORAGE_ACCOUNT_KEY,
            message_encode_policy=TextBase64EncodePolicy(),
        )
