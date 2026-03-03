"""
AI Service — Backward-Compatibility Façade

This module re-exports functions from their new homes in src/worker/services/
so that any existing code importing from src.ai.service continues to work.

New code should import directly from the dedicated service modules:
  - src.worker.services.video_download_service
  - src.worker.services.audio_service
  - src.worker.services.transcript_service
  - src.worker.services.clip_discovery_service
  - src.worker.services.video_editing_service
"""

# ── Video download ───────────────────────────────────────────────────
from src.worker.services.video_download_service import (
    download_from_youtube as download_youtube_video,
    download_from_azure as download_video_from_azure,
)

# ── Audio extraction ─────────────────────────────────────────────────
from src.worker.services.audio_service import extract_audio as get_audio_from_video

# ── Transcript ───────────────────────────────────────────────────────
from src.worker.services.transcript_service import generate_transcript

# ── Clip discovery ───────────────────────────────────────────────────
from src.worker.services.clip_discovery_service import (
    discover_clips as get_clips_from_transcript,
    resolve_timestamps as get_timestamps_from_clips,
)

# ── Video editing ────────────────────────────────────────────────────
from src.worker.services.video_editing_service import (
    cut_clips as cut_clips_from_video,
    add_subtitles as add_subtitles_to_clips,
)
