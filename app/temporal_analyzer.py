"""Temporal analysis module for detecting abrupt changes between video frames.

Detects:
- Abrupt visual changes (jump cuts, splices)
- Inconsistent motion patterns
- Lip-sync manipulation (sudden mouth changes)
"""

import numpy as np
from PIL import Image, ImageFilter

# Thresholds for temporal analysis (balanced for splice detection)
ABRUPT_CHANGE_THRESHOLD = 0.20  # 20% pixel difference = abrupt change
MOTION_INCONSISTENCY_THRESHOLD = 0.35  # 35% = very inconsistent
MIN_FRAMES_FOR_ANALYSIS = 3  # Need at least 3 frames


def calculate_frame_difference(frame1: Image.Image, frame2: Image.Image) -> float:
    """
    Calculate the normalized difference between two frames.

    Args:
        frame1: First PIL Image
        frame2: Second PIL Image

    Returns:
        float: Normalized difference (0.0 = identical, 1.0 = completely different)
    """
    # Ensure same size
    if frame1.size != frame2.size:
        frame2 = frame2.resize(frame1.size, Image.Resampling.LANCZOS)

    # Convert to RGB
    if frame1.mode != "RGB":
        frame1 = frame1.convert("RGB")
    if frame2.mode != "RGB":
        frame2 = frame2.convert("RGB")

    # Convert to numpy arrays
    arr1 = np.array(frame1, dtype=np.float32)
    arr2 = np.array(frame2, dtype=np.float32)

    # Calculate absolute difference
    diff = np.abs(arr1 - arr2)

    # Normalize (0-255 range to 0-1)
    normalized_diff = np.mean(diff) / 255.0

    return normalized_diff


def calculate_center_region_difference(
    frame1: Image.Image,
    frame2: Image.Image,
    region_ratio: float = 0.4,
) -> float:
    """
    Calculate difference in the center region (where face/mouth usually is).

    Args:
        frame1: First PIL Image
        frame2: Second PIL Image
        region_ratio: Size of center region (0.4 = 40% of image)

    Returns:
        float: Normalized difference in center region
    """
    # Get center region
    width, height = frame1.size
    left = int(width * (1 - region_ratio) / 2)
    top = int(height * (1 - region_ratio) / 2)
    right = int(width * (1 + region_ratio) / 2)
    bottom = int(height * (1 + region_ratio) / 2)

    # Crop center regions
    center1 = frame1.crop((left, top, right, bottom))
    center2 = frame2.crop((left, top, right, bottom))

    return calculate_frame_difference(center1, center2)


def detect_abrupt_changes(frames: list[Image.Image]) -> dict:
    """
    Analyze a sequence of frames to detect abrupt changes.

    Args:
        frames: List of PIL Images (video frames)

    Returns:
        dict with analysis results
    """
    if len(frames) < MIN_FRAMES_FOR_ANALYSIS:
        return {
            "abrupt_changes_detected": False,
            "confidence": 0.0,
            "num_abrupt_changes": 0,
            "change_positions": [],
            "details": "Not enough frames for temporal analysis.",
        }

    # Calculate differences between consecutive frames
    frame_differences = []
    center_differences = []

    for i in range(len(frames) - 1):
        full_diff = calculate_frame_difference(frames[i], frames[i + 1])
        center_diff = calculate_center_region_difference(frames[i], frames[i + 1])

        frame_differences.append(full_diff)
        center_differences.append(center_diff)

    # Convert to numpy for analysis
    frame_diffs = np.array(frame_differences)
    center_diffs = np.array(center_differences)

    # Calculate statistics
    mean_diff = np.mean(frame_diffs)
    std_diff = np.std(frame_diffs)
    mean_center_diff = np.mean(center_diffs)

    # Detect abrupt changes (significantly higher than average)
    # Using z-score: if difference is more than 2 std deviations above mean
    threshold_dynamic = mean_diff + 2 * std_diff
    threshold = max(threshold_dynamic, ABRUPT_CHANGE_THRESHOLD)

    abrupt_positions = []
    for i, diff in enumerate(frame_diffs):
        if diff > threshold:
            abrupt_positions.append({
                "frame": i,
                "difference": round(diff * 100, 2),
                "type": "full_frame",
            })

    # Also check center region for mouth/face changes
    center_threshold = max(mean_center_diff + 2 * np.std(center_diffs), ABRUPT_CHANGE_THRESHOLD)
    for i, diff in enumerate(center_diffs):
        if diff > center_threshold and diff > frame_diffs[i] * 1.5:
            # Center changed more than full frame (suspicious)
            abrupt_positions.append({
                "frame": i,
                "difference": round(diff * 100, 2),
                "type": "center_region",
            })

    # Remove duplicates and sort
    seen_frames = set()
    unique_positions = []
    for pos in abrupt_positions:
        if pos["frame"] not in seen_frames:
            seen_frames.add(pos["frame"])
            unique_positions.append(pos)

    unique_positions.sort(key=lambda x: x["frame"])

    # Determine if manipulation detected
    num_abrupt = len(unique_positions)
    total_transitions = len(frames) - 1

    # Manipulation suspected if:
    # - At least 2 abrupt changes (splices)
    # - Or very high motion inconsistency
    abrupt_ratio = num_abrupt / total_transitions if total_transitions > 0 else 0
    motion_inconsistency = std_diff / mean_diff if mean_diff > 0 else 0

    is_manipulated = (
        num_abrupt >= 2 or  # At least 2 abrupt changes (likely splices)
        motion_inconsistency > MOTION_INCONSISTENCY_THRESHOLD
    )

    # Calculate confidence
    if is_manipulated:
        confidence = min(0.95, 0.5 + abrupt_ratio + motion_inconsistency * 0.5)
    else:
        confidence = min(0.95, 1 - abrupt_ratio - motion_inconsistency * 0.3)

    # Generate details
    if is_manipulated:
        if num_abrupt > 0:
            details = (
                f"Temporal analysis detected {num_abrupt} abrupt change(s) in {total_transitions} "
                f"frame transitions. Average frame difference: {mean_diff*100:.1f}%, "
                f"with sudden jumps up to {max(frame_diffs)*100:.1f}%. "
                f"This pattern suggests video splicing or manipulation."
            )
        else:
            details = (
                f"High motion inconsistency detected (variability: {motion_inconsistency:.2f}). "
                f"Frame differences vary significantly, suggesting potential manipulation."
            )
    else:
        details = (
            f"Temporal analysis completed. Analyzed {total_transitions} frame transitions. "
            f"Average difference: {mean_diff*100:.1f}%. No abrupt changes detected. "
            f"Video appears to have consistent temporal flow."
        )

    return {
        "abrupt_changes_detected": is_manipulated,
        "confidence": round(confidence, 4),
        "num_abrupt_changes": num_abrupt,
        "change_positions": unique_positions,
        "avg_frame_difference": round(mean_diff * 100, 2),
        "motion_inconsistency": round(motion_inconsistency, 4),
        "details": details,
    }


def analyze_temporal(frames: list[Image.Image]) -> dict:
    """
    Convenience function for temporal analysis.

    Args:
        frames: List of PIL Images (video frames)

    Returns:
        dict with is_manipulated, confidence, fraud_type, details
    """
    result = detect_abrupt_changes(frames)

    return {
        "is_manipulated": result["abrupt_changes_detected"],
        "confidence": result["confidence"],
        "fraud_type": "temporal_inconsistency" if result["abrupt_changes_detected"] else None,
        "num_abrupt_changes": result["num_abrupt_changes"],
        "details": result["details"],
    }
