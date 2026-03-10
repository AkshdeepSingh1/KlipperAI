"""
Video Download Service
Downloads raw video files to local disk from various sources.
"""
import os
import requests
import yt_dlp
from urllib.parse import urlparse, parse_qs

from src.shared.core.logger import get_logger

logger = get_logger(__name__)


def download_from_youtube(url: str, user_id: str, video_id: str) -> str:
    """
    Download a video from a URL using yt-dlp.

    Args:
        url: YouTube (or similar) video URL
        user_id: User identifier for path construction
        video_id: Video identifier for path construction

    Returns:
        Local file path of the downloaded video
    """
    logger.info(f"Starting download for URL: {url}")

    download_dir = os.path.join("downloads", user_id, video_id)
    os.makedirs(download_dir, exist_ok=True)

    ydl_opts = {
        "format": (
            "best[ext=mp4][protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]"
            "/best[protocol!=m3u8_native][protocol!=m3u8][protocol!=http_dash_segments]"
        ),
        "format_sort": ["lang:en"],
        "outtmpl": os.path.join(download_dir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
        "no_check_certificate": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found: {file_path}")

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")

            logger.info(
                f"Video downloaded successfully: {file_path} (size: {file_size} bytes)"
            )
            return file_path

    except Exception as e:
        error_msg = f"Error downloading video: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def download_from_azure(user_id: str, video_id: str, link: str) -> None:
    """
    Download a video from Azure Blob Storage (or any HTTP URL) to local disk.

    Args:
        user_id: User identifier for path construction
        video_id: Video identifier for path construction
        link: Full URL of the blob (may include SAS token)
    """
    parsed = urlparse(link)
    output_dir = os.path.join("downloads", user_id, video_id)
    os.makedirs(output_dir, exist_ok=True)

    dest_path = os.path.join("downloads", user_id, video_id, "video.mp4")

    # Checkpoint: skip download if video already exists
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info(f"Video already exists at {dest_path}, skipping download")
        return

    query = parsed.query or ""
    qs = parse_qs(query)
    has_sas = ("sig" in qs) or ("sv" in qs) or ("se" in qs and "sp" in qs)

    if has_sas or parsed.scheme.startswith("http"):
        try:
            with requests.get(link, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            logger.info(f"Downloaded to {dest_path}")
            return
        except requests.HTTPError as e:
            raise requests.HTTPError(
                f"HTTP error while downloading {link}: {e}"
            ) from e
        except requests.RequestException as e:
            raise Exception(
                f"Network error while downloading {link}: {e}"
            ) from e
