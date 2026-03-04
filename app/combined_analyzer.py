"""Combined analysis module that integrates all detection methods."""

from PIL import Image

from .detector import get_detector
from .deepfake_detector import get_deepfake_detector
from .ela_analyzer import analyze_ela
from .utils import preprocess_image


def analyze_image_comprehensive(image: Image.Image) -> dict:
    """
    Perform comprehensive analysis using all available detection methods.

    Combines:
    - AI-generated content detection
    - Deepfake detection
    - Error Level Analysis (ELA) for editing detection

    Args:
        image: PIL Image to analyze

    Returns:
        dict with combined analysis results
    """
    # Preprocess image
    processed = preprocess_image(image)

    # Run all detectors
    ai_result = get_detector().analyze(processed)
    deepfake_result = get_deepfake_detector().analyze(processed)
    ela_result = analyze_ela(image)  # Use original for ELA

    # Determine fraud types detected
    fraud_types = []

    if ai_result["is_ai_generated"]:
        fraud_types.append("ai_generated")

    if deepfake_result["is_deepfake"]:
        fraud_types.append("deepfake")

    if ela_result["is_manipulated"]:
        fraud_types.append("image_editing")

    # Calculate overall fraud detection
    fraud_detected = len(fraud_types) > 0

    # Calculate veracity score (0-100, higher = more authentic)
    # Weight: AI detection (40%), Deepfake (40%), ELA (20%)
    ai_penalty = ai_result["confidence"] * 40 if ai_result["is_ai_generated"] else 0
    deepfake_penalty = deepfake_result["confidence"] * 40 if deepfake_result["is_deepfake"] else 0
    ela_penalty = ela_result["confidence"] * 20 if ela_result["is_manipulated"] else 0

    veracity_score = int(max(0, 100 - ai_penalty - deepfake_penalty - ela_penalty))

    # Calculate overall confidence
    if fraud_detected:
        # Average of detected fraud confidences
        confidences = []
        if ai_result["is_ai_generated"]:
            confidences.append(ai_result["confidence"])
        if deepfake_result["is_deepfake"]:
            confidences.append(deepfake_result["confidence"])
        if ela_result["is_manipulated"]:
            confidences.append(ela_result["confidence"])
        confidence = int(sum(confidences) / len(confidences) * 100)
    else:
        # Average of "real" confidences
        confidence = int((
            (1 - ai_result["confidence"] if not ai_result["is_ai_generated"] else ai_result["confidence"]) +
            (1 - deepfake_result["confidence"] if not deepfake_result["is_deepfake"] else deepfake_result["confidence"])
        ) / 2 * 100)
        confidence = min(confidence, 95)  # Cap at 95% for real

    # Generate combined details
    details = _generate_combined_details(
        ai_result=ai_result,
        deepfake_result=deepfake_result,
        ela_result=ela_result,
        fraud_types=fraud_types,
    )

    return {
        "veracity_score": veracity_score,
        "fraud_detected": fraud_detected,
        "confidence": confidence,
        "fraud_types": fraud_types,
        "details": details,
        "analysis": {
            "ai_detection": {
                "detected": ai_result["is_ai_generated"],
                "confidence": round(ai_result["confidence"] * 100, 1),
            },
            "deepfake_detection": {
                "detected": deepfake_result["is_deepfake"],
                "confidence": round(deepfake_result["confidence"] * 100, 1),
            },
            "ela_analysis": {
                "manipulation_detected": ela_result["is_manipulated"],
                "confidence": round(ela_result["confidence"] * 100, 1),
                "suspicious_area": ela_result["suspicious_area_percentage"],
            },
        },
    }


def _generate_combined_details(
    ai_result: dict,
    deepfake_result: dict,
    ela_result: dict,
    fraud_types: list[str],
) -> str:
    """Generate a human-readable description of the combined analysis."""

    if not fraud_types:
        return (
            "Comprehensive analysis completed. No manipulation detected. "
            "The image passed AI-generation check, deepfake detection, and "
            "error level analysis. The image appears to be authentic."
        )

    issues = []

    if ai_result["is_ai_generated"]:
        issues.append(
            f"AI-generated content detected ({ai_result['confidence']*100:.0f}% confidence)"
        )

    if deepfake_result["is_deepfake"]:
        issues.append(
            f"Deepfake/face manipulation detected ({deepfake_result['confidence']*100:.0f}% confidence)"
        )

    if ela_result["is_manipulated"]:
        issues.append(
            f"Image editing detected via ELA ({ela_result['suspicious_area_percentage']:.1f}% suspicious area)"
        )

    return (
        f"Manipulation detected. Issues found: {'; '.join(issues)}. "
        f"The image shows signs of {', '.join(fraud_types).replace('_', ' ')}."
    )


# Minimum confidence threshold for AI detection
MIN_CONFIDENCE_AI = 0.70  # 70% confidence for AI detection


def analyze_frame_comprehensive(image: Image.Image, include_ela: bool = False) -> dict:
    """
    Analysis for video frames - uses only AI detection.

    Deepfake and ELA are disabled for video as they cause too many false positives
    due to video compression artifacts.

    Args:
        image: PIL Image to analyze
        include_ela: Ignored for video (always False)

    Returns:
        dict with is_fraud, confidence, fraud_types
    """
    processed = preprocess_image(image)

    # Only run AI detection for video frames
    ai_result = get_detector().analyze(processed)

    fraud_types = []

    # Only count as fraud if confidence meets minimum threshold
    ai_is_fraud = ai_result["is_ai_generated"] and ai_result["confidence"] >= MIN_CONFIDENCE_AI

    if ai_is_fraud:
        fraud_types.append("ai_generated")

    is_fraud = len(fraud_types) > 0
    confidence = ai_result["confidence"]

    return {
        "is_fraud": is_fraud,
        "confidence": confidence,
        "fraud_types": fraud_types,
    }
