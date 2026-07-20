"""
Pipeline stage: URL-based video download.
Downloads a video from a URL (YouTube, direct MP4 link, or most platforms
yt-dlp supports) so it can be processed the same way as an uploaded file.
"""
from pathlib import Path
import yt_dlp


def download_video_from_url(url: str, output_dir: Path, video_id: str) -> Path:
    """
    Download a video from a URL and save it as <video_id>.<ext> in
    output_dir. Returns the local file path once the download (and any
    audio/video merge) is complete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # yt-dlp fills in %(ext)s with the actual container format at download time.
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        # Prefer a pre-merged mp4 if the source offers one; otherwise pull
        # the best available video+audio streams separately and merge them;
        # "best" is the final fallback for sources with only a single stream.
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",  # force ffmpeg to mux to .mp4 if streams were downloaded separately
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

        # prepare_filename() predicts the path BEFORE any post-processing
        # (like the video+audio merge) runs -- if the source needed
        # separate-stream merging, the actual output extension may end up
        # different (typically .mp4 per merge_output_format above) from
        # what prepare_filename() predicted. Fall back to checking the
        # .mp4 variant in that case.
        if not downloaded_path.exists():
            downloaded_path = downloaded_path.with_suffix(".mp4")

    if not downloaded_path.exists():
        raise FileNotFoundError(f"Download completed but file not found: {downloaded_path}")

    return downloaded_path
