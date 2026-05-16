"""
Daily Job Dispatcher Cron Job
------------------------------
A time-triggered cron worker that fires at 06:00 UTC every day.

Responsibilities:
  - Wait until the next 06:00 UTC wall-clock moment.
  - Delegate all business logic to ScheduledJobsDispatcherService.
  - Loop forever (one dispatch per day).

This class intentionally contains NO database or queue logic — it only
orchestrates timing and delegates to the service (SRP + DIP).
"""

import time
from datetime import datetime, timezone

from src.worker.workers.base_worker import BaseWorker
from src.worker.services.scheduled_jobs_dispatcher_service import (
    ScheduledJobsDispatcherService,
)

# The hour (UTC) at which the cron job fires each day.
DISPATCH_HOUR_UTC: int = 6


class DailyJobDispatcherCronJob(BaseWorker):
    """
    Cron-style worker that wakes up at 06:00 UTC and dispatches all
    content job requests scheduled for that calendar day to the
    p2-text-to-content-worker Azure Queue.
    """

    name: str = "DailyJobDispatcherCronJob"

    # ------------------------------------------------------------------
    # Lifecycle hooks (BaseWorker protocol)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        super().setup()
        self._dispatcher = ScheduledJobsDispatcherService()
        self.logger.info(
            f"[{self.name}] Initialized. Will dispatch jobs every day at "
            f"{DISPATCH_HOUR_UTC:02d}:00 UTC."
        )

    def run(self) -> None:
        """
        Main loop:
          1. Sleep until 06:00 UTC.
          2. Run the dispatcher.
          3. Repeat.
        """
        self.logger.info(f"[{self.name}] Cron loop started.")

        while self.running:
            seconds_to_wait = self._seconds_until_next_dispatch()
            next_run = datetime.now(timezone.utc)
            self.logger.info(
                f"[{self.name}] Next dispatch in {seconds_to_wait:.0f}s "
                f"(≈ {seconds_to_wait / 3600:.2f}h). "
                f"Current UTC time: {next_run.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Sleep in short increments so KeyboardInterrupt is responsive.
            self._interruptible_sleep(seconds_to_wait)

            if not self.running:
                break

            self._execute_dispatch()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_dispatch(self) -> None:
        """Safely call the dispatcher and log the outcome."""
        try:
            self.logger.info(f"[{self.name}] Firing daily job dispatch...")
            enqueued = self._dispatcher.dispatch_todays_jobs()
            self.logger.info(
                f"[{self.name}] Dispatch complete. {enqueued} job(s) enqueued."
            )
        except Exception as exc:
            self.logger.error(
                f"[{self.name}] Dispatch failed with error: {exc}", exc_info=True
            )

    def _seconds_until_next_dispatch(self) -> float:
        """
        Calculate how many seconds remain until the next 06:00 UTC.
        If it is currently past 06:00, the result targets tomorrow's 06:00.
        """
        now = datetime.now(timezone.utc)
        # Build today's target time
        target = now.replace(
            hour=DISPATCH_HOUR_UTC, minute=0, second=0, microsecond=0
        )
        delta = (target - now).total_seconds()
        if delta <= 0:
            # We are already past 06:00 today — target tomorrow
            delta += 86_400  # 24 * 60 * 60
        return delta

    def _interruptible_sleep(self, total_seconds: float) -> None:
        """
        Sleep in 60-second chunks so the worker can respond to
        `self.running = False` (e.g. on SIGTERM) without a long delay.
        """
        slept = 0.0
        chunk = 60.0  # seconds per tick
        while self.running and slept < total_seconds:
            sleep_for = min(chunk, total_seconds - slept)
            time.sleep(sleep_for)
            slept += sleep_for
