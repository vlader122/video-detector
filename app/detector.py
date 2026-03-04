"""Core detector module using HuggingFace model."""

import time
from functools import lru_cache

import torch
from PIL import Image
from transformers import pipeline

from .utils import preprocess_image


def get_device() -> str:
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Batch size for processing multiple frames at once
# Adjust based on available memory (GPU: 8-16, CPU: 4-8)
BATCH_SIZE = 8


class AIDetector:
    """Singleton detector for AI-generated images."""

    _instance = None
    _classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "umm-maybe/AI-image-detector"):
        if self._classifier is None:
            self._model_name = model_name
            self._load_model()

    def _load_model(self):
        """Load the HuggingFace model."""
        device = get_device()
        self._classifier = pipeline(
            "image-classification",
            model=self._model_name,
            device=device,
            use_fast=True,
        )
        print(f"Model loaded on: {device.upper()}")

    def analyze(self, image: Image.Image) -> dict:
        """
        Analyze an image and return AI detection results.

        Args:
            image: PIL Image to analyze

        Returns:
            dict with is_ai_generated, confidence, label, and processing_time_ms
        """
        start_time = time.time()

        # Run classification
        results = self._classifier(image)

        # Parse results
        parsed = self._parse_result(results)

        processing_time = (time.time() - start_time) * 1000
        parsed["processing_time_ms"] = round(processing_time, 2)

        return parsed

    def analyze_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        Analyze multiple images in batch for better performance.

        Args:
            images: List of PIL Images to analyze

        Returns:
            List of result dicts
        """
        if not images:
            return []

        # Run batch classification
        results = self._classifier(images, batch_size=BATCH_SIZE)

        # Parse each result
        parsed_results = []
        for result in results:
            parsed = self._parse_result(result)
            parsed_results.append(parsed)

        return parsed_results

    def _parse_result(self, results: list[dict]) -> dict:
        """Parse classification results into standardized format."""
        # Model returns: [{'label': 'artificial', 'score': X}, {'label': 'human', 'score': Y}]
        ai_score = 0.0
        human_score = 0.0

        for result in results:
            label = result["label"].lower()
            if label in ("artificial", "ai", "fake"):
                ai_score = result["score"]
            elif label in ("human", "real"):
                human_score = result["score"]

        # Determine if AI generated (threshold: 0.5)
        is_ai_generated = ai_score > human_score
        confidence = ai_score if is_ai_generated else human_score

        return {
            "is_ai_generated": is_ai_generated,
            "confidence": round(confidence, 4),
            "label": "ai_generated" if is_ai_generated else "human_made",
        }


@lru_cache(maxsize=1)
def get_detector() -> AIDetector:
    """Get or create the singleton detector instance."""
    return AIDetector()


def analyze_image(image: Image.Image) -> dict:
    """
    Convenience function to analyze an image.

    Args:
        image: PIL Image to analyze

    Returns:
        Analysis result dict
    """
    detector = get_detector()
    return detector.analyze(image)


# Early stopping settings
EARLY_STOP_THRESHOLD = 0.85  # Stop if fraud confidence >= 85%
EARLY_STOP_MIN_FRAMES = 3    # Minimum frames before early stopping


def analyze_video_frames(
    frames: list[Image.Image],
    video_duration: float = 0.0,
    detailed: bool = False,
    early_stop: bool = True,
    comprehensive: bool = True,
    temporal_analysis: bool = True,
) -> dict:
    """
    Analyze multiple frames from a video and return aggregated results.

    Uses early stopping: stops analysis when fraud is detected with high confidence.
    Includes temporal analysis to detect abrupt changes between frames.

    Args:
        frames: List of PIL Images (keyframes from video)
        video_duration: Duration of the video in seconds
        detailed: If True, include additional metadata in response
        early_stop: If True, stop when fraud detected with high confidence
        comprehensive: If True, use combined analysis (AI + deepfake)
        temporal_analysis: If True, analyze for abrupt changes between frames

    Returns:
        dict with veracity_score, fraud_detected, confidence, fraud_types, details
    """
    start_time = time.time()

    if not frames:
        result = {
            "veracity_score": 100,
            "fraud_detected": False,
            "confidence": 0,
            "fraud_types": [],
            "details": "No frames could be extracted from the video for analysis.",
        }
        if detailed:
            result.update({
                "total_frames_analyzed": 0,
                "fraud_frames": 0,
                "video_duration_seconds": video_duration,
                "processing_time_ms": 0.0,
            })
        return result

    # Import here to avoid circular imports
    if comprehensive:
        from .combined_analyzer import analyze_frame_comprehensive
    if temporal_analysis:
        from .temporal_analyzer import analyze_temporal

    # First, run temporal analysis on all frames (fast, doesn't use AI)
    temporal_fraud_detected = False
    temporal_confidence = 0.0
    temporal_details = ""

    if temporal_analysis and len(frames) >= 3:
        temporal_result = analyze_temporal(frames)
        temporal_fraud_detected = temporal_result["is_manipulated"]
        temporal_confidence = temporal_result["confidence"]
        temporal_details = temporal_result["details"]

    # Analyze frames with early stopping
    fraud_count = 0
    total_fraud_confidence = 0.0
    frames_analyzed = 0
    early_stopped = False
    all_fraud_types = set()

    # If temporal analysis detected fraud, add it to fraud types
    if temporal_fraud_detected:
        all_fraud_types.add("temporal_inconsistency")

    for i, frame in enumerate(frames):
        frames_analyzed += 1

        if comprehensive:
            # Use combined analysis (AI + deepfake detection)
            frame_result = analyze_frame_comprehensive(frame)
            is_fraud = frame_result["is_fraud"]
            confidence = frame_result["confidence"]
            if frame_result["fraud_types"]:
                all_fraud_types.update(frame_result["fraud_types"])
        else:
            # Use only AI detection
            processed = preprocess_image(frame)
            frame_result = get_detector().analyze(processed)
            is_fraud = frame_result["is_ai_generated"]
            confidence = frame_result["confidence"]
            if is_fraud:
                all_fraud_types.add("ai_generated")

        if is_fraud:
            fraud_count += 1
            total_fraud_confidence += confidence

            # Early stopping: if high confidence fraud detected after minimum frames
            if early_stop and frames_analyzed >= EARLY_STOP_MIN_FRAMES:
                if confidence >= EARLY_STOP_THRESHOLD:
                    early_stopped = True
                    break

    # Calculate aggregates
    total_frames = len(frames)
    fraud_percentage = (fraud_count / frames_analyzed) * 100 if frames_analyzed > 0 else 0

    # Calculate average confidence for fraud frames
    if fraud_count > 0:
        avg_fraud_confidence = total_fraud_confidence / fraud_count
    else:
        avg_fraud_confidence = 0.0

    # Veracity score: 100 = completely real, 0 = completely fake
    # Include temporal fraud penalty
    temporal_penalty = temporal_confidence * 30 if temporal_fraud_detected else 0
    veracity_score = int(round(max(0, 100 - fraud_percentage - temporal_penalty)))

    # Fraud detected if veracity_score <= 50 (more fake than real)
    fraud_detected = veracity_score <= 50

    # Confidence: how sure we are about the verdict
    if fraud_detected:
        # Use the highest confidence from detected fraud types
        confidences = []
        if fraud_count > 0 and avg_fraud_confidence > 0:
            confidences.append(avg_fraud_confidence)
        if temporal_fraud_detected and temporal_confidence > 0:
            confidences.append(temporal_confidence)
        confidence = int(round(max(confidences) * 100)) if confidences else 50
    else:
        # Confidence in "real" verdict based on veracity score
        confidence = int(round(min(veracity_score, 95)))

    processing_time = (time.time() - start_time) * 1000

    # Convert fraud types set to sorted list (only if fraud detected)
    fraud_types = sorted(list(all_fraud_types)) if fraud_detected and all_fraud_types else []

    # Generate detailed description
    details = _generate_analysis_details(
        frames_analyzed=frames_analyzed,
        total_frames=total_frames,
        ai_count=fraud_count,
        ai_percentage=fraud_percentage,
        avg_confidence=avg_fraud_confidence,
        fraud_detected=fraud_detected,
        video_duration=video_duration,
        early_stopped=early_stopped,
        fraud_types=fraud_types,
        temporal_fraud_detected=temporal_fraud_detected,
        temporal_details=temporal_details,
    )

    result = {
        "veracity_score": veracity_score,
        "fraud_detected": fraud_detected,
        "confidence": confidence,
        "fraud_types": fraud_types,
        "details": details,
    }

    if detailed:
        result.update({
            "total_frames_analyzed": frames_analyzed,
            "fraud_frames": fraud_count,
            "video_duration_seconds": round(video_duration, 2),
            "processing_time_ms": round(processing_time, 2),
            "early_stopped": early_stopped,
        })

    return result


def _generate_analysis_details(
    frames_analyzed: int,
    total_frames: int,
    ai_count: int,
    ai_percentage: float,
    avg_confidence: float,
    fraud_detected: bool,
    video_duration: float,
    early_stopped: bool = False,
    fraud_types: list[str] = None,
    temporal_fraud_detected: bool = False,
    temporal_details: str = "",
) -> str:
    """Generate a human-readable description of the analysis results."""

    fraud_types = fraud_types or []

    if not fraud_detected:
        if ai_count == 0:
            return (
                f"Video analysis completed. Analyzed {frames_analyzed} keyframes "
                f"from {video_duration:.1f}s video. No manipulation detected. "
                f"The video appears to be authentic."
            )
        else:
            return (
                f"Video analysis completed. Analyzed {frames_analyzed} keyframes "
                f"from {video_duration:.1f}s video. Only {ai_count} frame(s) "
                f"({ai_percentage:.1f}%) showed minor signs of manipulation, "
                f"which is below the fraud threshold. Video is likely authentic."
            )

    # Fraud detected
    confidence_level = "High" if avg_confidence >= 0.8 else "Medium" if avg_confidence >= 0.6 else "Low"

    # Describe detected fraud types
    fraud_desc = []
    if "ai_generated" in fraud_types:
        fraud_desc.append("AI-generated content")
    if "deepfake" in fraud_types:
        fraud_desc.append("deepfake manipulation")
    if "image_editing" in fraud_types:
        fraud_desc.append("image editing")
    if "temporal_inconsistency" in fraud_types:
        fraud_desc.append("temporal inconsistencies (abrupt changes)")

    fraud_str = ", ".join(fraud_desc) if fraud_desc else "manipulation"

    # Build severity message based on what was detected
    if ai_percentage >= 80:
        severity = f"The video is highly likely to contain {fraud_str}."
    elif ai_percentage >= 50:
        severity = f"Significant portions of the video show {fraud_str}."
    elif ai_percentage > 0:
        severity = f"Some frames in the video show signs of {fraud_str}."
    elif temporal_fraud_detected:
        severity = f"Temporal inconsistencies detected: {fraud_str}."
    else:
        severity = f"The video shows signs of {fraud_str}."

    early_stop_msg = ""
    if early_stopped:
        early_stop_msg = f" Analysis stopped early after detecting fraud with {avg_confidence*100:.0f}% confidence."

    temporal_msg = ""
    if temporal_fraud_detected and temporal_details:
        temporal_msg = f" {temporal_details}"

    # Build main message
    if ai_count > 0:
        main_msg = (
            f"Video analysis detected {ai_count} of {frames_analyzed} frames "
            f"({ai_percentage:.1f}%) with manipulation. "
            f"{confidence_level} confidence ({avg_confidence*100:.0f}%) in detection. "
        )
    elif temporal_fraud_detected:
        main_msg = (
            f"Video analysis of {frames_analyzed} frames detected temporal manipulation. "
        )
    else:
        main_msg = f"Video analysis completed on {frames_analyzed} frames. "

    return f"{main_msg}{severity}{early_stop_msg}{temporal_msg}"
