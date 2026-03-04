"""Utility functions for image processing."""

from io import BytesIO
from typing import BinaryIO

from PIL import Image

# Supported image formats
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

# Maximum image dimension (resize if larger)
# 512px is sufficient for AI detection and much faster to process
MAX_DIMENSION = 512


def validate_content_type(content_type: str | None) -> bool:
    """
    Validate if the content type is a supported image format.

    Args:
        content_type: MIME type of the file

    Returns:
        True if supported, False otherwise
    """
    if content_type is None:
        return False
    return content_type.lower() in SUPPORTED_CONTENT_TYPES


def validate_file_extension(filename: str | None) -> bool:
    """
    Validate if the file extension is supported.

    Args:
        filename: Name of the file

    Returns:
        True if supported, False otherwise
    """
    if filename is None:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in SUPPORTED_EXTENSIONS


def load_image(file: BinaryIO) -> Image.Image:
    """
    Load an image from a file-like object.

    Args:
        file: File-like object containing image data

    Returns:
        PIL Image object

    Raises:
        ValueError: If the image format is not supported
    """
    try:
        image = Image.open(file)
        image.load()  # Force load to catch corrupt images

        if image.format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {image.format}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        return image
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to load image: {str(e)}")


def preprocess_image(image: Image.Image, max_size: int = MAX_DIMENSION) -> Image.Image:
    """
    Preprocess image for analysis.

    - Convert to RGB if necessary
    - Resize if larger than max_size

    Args:
        image: PIL Image to preprocess
        max_size: Maximum dimension (width or height)

    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Convert a PIL Image to bytes.

    Args:
        image: PIL Image to convert
        format: Output format (JPEG, PNG, etc.)

    Returns:
        Image as bytes
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()
