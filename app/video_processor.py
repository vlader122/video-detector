"""Video processing module for keyframe extraction using PyAV."""

import tempfile
from pathlib import Path
from typing import BinaryIO

import av
from PIL import Image

# Supported video formats
SUPPORTED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/avi",
    "video/mov",
    "video/mkv",
}

# Maximum file size (100MB)
MAX_VIDEO_SIZE = 100 * 1024 * 1024

# Keyframe extraction settings
MIN_KEYFRAMES = 5       # Minimum frames for short videos
MAX_KEYFRAMES = None    # No limit - extract all keyframes
FRAMES_PER_SECOND = 1   # Target: 1 frame per second


def calculate_target_frames(duration_seconds: float) -> int | None:
    """
    Calculate the target number of frames based on video duration.

    Args:
        duration_seconds: Video duration in seconds

    Returns:
        Target number of frames to extract (None = no limit)
    """
    if MAX_KEYFRAMES is None:
        return None  # No limit, extract all keyframes

    if duration_seconds <= 0:
        return MIN_KEYFRAMES

    # Calculate based on duration
    target = int(duration_seconds * FRAMES_PER_SECOND)

    # Clamp between min and max
    return max(MIN_KEYFRAMES, min(target, MAX_KEYFRAMES))


def validate_video_content_type(content_type: str | None) -> bool:
    """
    Validate if the content type is a supported video format.

    Args:
        content_type: MIME type of the file

    Returns:
        True if supported, False otherwise
    """
    if content_type is None:
        return False
    return content_type.lower() in SUPPORTED_VIDEO_CONTENT_TYPES


def get_video_info(video_path: str | Path) -> dict:
    """
    Get video metadata information.

    Args:
        video_path: Path to the video file

    Returns:
        dict with duration, fps, codec, width, height
    """
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    duration = float(container.duration / av.time_base) if container.duration else 0.0
    fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0

    info = {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "codec": video_stream.codec_context.name,
        "width": video_stream.width,
        "height": video_stream.height,
    }

    container.close()
    return info


def extract_keyframes(video_path: str | Path, max_frames: int | None = None) -> list[Image.Image]:
    """
    Extract keyframes (I-frames) from a video file.

    Only decodes keyframes, which is much faster than decoding all frames.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of keyframes to extract (None = all)

    Returns:
        List of PIL Images (keyframes)
    """
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    # Only decode keyframes (I-frames)
    video_stream.codec_context.skip_frame = "NONKEY"

    keyframes = []
    frame_count = 0

    for frame in container.decode(video=0):
        # Convert to PIL Image
        img = frame.to_image()
        keyframes.append(img)
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    container.close()
    return keyframes


def extract_keyframes_from_bytes(
    video_bytes: bytes,
    max_frames: int | None = None,
) -> tuple[list[Image.Image], dict]:
    """
    Extract keyframes from video bytes (uploaded file).

    The number of frames extracted is proportional to video duration:
    - Short videos (< 10s): 5 frames
    - Medium videos (10-60s): ~1 frame per 2 seconds
    - Long videos (> 60s): up to 30 frames

    Args:
        video_bytes: Video file content as bytes
        max_frames: Override for maximum frames (None = auto-calculate)

    Returns:
        Tuple of (list of PIL Images, video info dict)
    """
    # Write to temporary file (PyAV needs file path for seeking)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # Get video info first
        info = get_video_info(tmp_path)

        # Extract frames at regular intervals (1 per second) for better splice detection
        # This is slower than keyframes but catches more manipulation
        frames = extract_frames_at_interval(
            tmp_path,
            interval_seconds=1.0,  # 1 frame per second
            max_frames=max_frames,
        )

        # Add frame info to metadata
        info["frames_extracted"] = len(frames)
        info["extraction_method"] = "interval"

        return frames, info

    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def extract_frames_at_interval(
    video_path: str | Path,
    interval_seconds: float = 1.0,
    max_frames: int | None = None,
) -> list[Image.Image]:
    """
    Extract frames at regular intervals (alternative to keyframes).

    Args:
        video_path: Path to the video file
        interval_seconds: Extract one frame every N seconds
        max_frames: Maximum number of frames to extract

    Returns:
        List of PIL Images
    """
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    frame_interval = int(fps * interval_seconds)

    frames = []
    frame_count = 0

    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_interval == 0:
            img = frame.to_image()
            frames.append(img)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

    container.close()
    return frames
