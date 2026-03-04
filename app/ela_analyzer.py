"""Error Level Analysis (ELA) module for detecting image manipulation.

ELA works by re-saving the image at a known quality level and comparing
the difference. Areas that have been edited will have different error
levels compared to the rest of the image.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


# ELA settings (adjusted for video frames which have compression artifacts)
ELA_QUALITY = 90  # JPEG quality for re-compression
ELA_SCALE = 15    # Scale factor for visibility
MANIPULATION_THRESHOLD = 45  # Threshold for detecting manipulation (0-255) - was 25
MANIPULATION_AREA_THRESHOLD = 0.15  # 15% of pixels must be suspicious - was 5%


def perform_ela(image: Image.Image, quality: int = ELA_QUALITY) -> Image.Image:
    """
    Perform Error Level Analysis on an image.

    Args:
        image: PIL Image to analyze
        quality: JPEG quality for re-compression (default: 90)

    Returns:
        PIL Image showing the ELA result (differences highlighted)
    """
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Re-save at specified quality
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Calculate difference
    ela_image = ImageChops.difference(image, resaved)

    # Enhance the difference to make it more visible
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff * ELA_SCALE
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image


def analyze_ela(image: Image.Image) -> dict:
    """
    Analyze an image using Error Level Analysis to detect manipulation.

    Args:
        image: PIL Image to analyze

    Returns:
        dict with is_manipulated, confidence, suspicious_area_percentage, details
    """
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Perform ELA
    ela_image = perform_ela(image)

    # Convert to numpy for analysis
    ela_array = np.array(ela_image)

    # Calculate metrics
    # Get the maximum value across RGB channels for each pixel
    max_values = np.max(ela_array, axis=2)

    # Count pixels above threshold (suspicious areas)
    suspicious_pixels = np.sum(max_values > MANIPULATION_THRESHOLD)
    total_pixels = max_values.size
    suspicious_percentage = suspicious_pixels / total_pixels

    # Calculate average and max error levels
    avg_error = np.mean(max_values)
    max_error = np.max(max_values)

    # Calculate standard deviation (high std = inconsistent = likely edited)
    std_error = np.std(max_values)

    # Determine if manipulated based on multiple factors (stricter for video)
    is_manipulated = (
        suspicious_percentage > MANIPULATION_AREA_THRESHOLD or
        std_error > 50 or  # High variance indicates editing - was 30
        max_error > 230    # Very bright spots indicate heavy editing - was 200
    )

    # Calculate confidence based on how clear the manipulation is
    if is_manipulated:
        # Higher suspicious area = higher confidence
        confidence = min(0.95, suspicious_percentage * 5 + std_error / 100)
    else:
        # Lower suspicious area = higher confidence it's real
        confidence = min(0.95, 1 - suspicious_percentage * 3)

    # Generate details
    if is_manipulated:
        if suspicious_percentage > 0.2:
            details = f"Significant manipulation detected. {suspicious_percentage*100:.1f}% of the image shows signs of editing."
        elif std_error > 40:
            details = f"Inconsistent compression levels detected (std={std_error:.1f}), suggesting the image has been edited."
        else:
            details = f"Minor manipulation detected. Some areas ({suspicious_percentage*100:.1f}%) show different compression levels."
    else:
        details = f"No significant manipulation detected. Image appears to have consistent compression levels."

    return {
        "is_manipulated": is_manipulated,
        "confidence": round(confidence, 4),
        "suspicious_area_percentage": round(suspicious_percentage * 100, 2),
        "avg_error_level": round(avg_error, 2),
        "max_error_level": round(max_error, 2),
        "std_error_level": round(std_error, 2),
        "details": details,
    }


def get_ela_image(image: Image.Image) -> Image.Image:
    """
    Get the ELA visualization image.

    Args:
        image: PIL Image to analyze

    Returns:
        PIL Image showing ELA visualization (bright areas = potential edits)
    """
    return perform_ela(image)
