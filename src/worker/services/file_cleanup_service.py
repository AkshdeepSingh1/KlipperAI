"""
File Cleanup Service
Cleans up temporary download directories after processing.
"""
import shutil
from pathlib import Path

from src.shared.core.logger import get_logger

logger = get_logger(__name__)


def cleanup_downloads(user_id: str, video_id: str) -> None:
    """
    Remove the temporary download directory for a processed video.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction
    """
    download_dir = Path("downloads") / user_id / video_id
    if download_dir.exists():
        shutil.rmtree(download_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {download_dir}")
