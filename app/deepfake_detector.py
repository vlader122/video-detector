"""Deepfake detection module using HuggingFace model."""

from functools import lru_cache

import torch
from PIL import Image
from transformers import pipeline


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DeepfakeDetector:
    """Detector for deepfake/face-swap content."""

    _instance = None
    _classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "dima806/deepfake_vs_real_image_detection"):
        if self._classifier is None:
            self._model_name = model_name
            self._load_model()

    def _load_model(self):
        """Load the HuggingFace deepfake detection model."""
        device = get_device()
        self._classifier = pipeline(
            "image-classification",
            model=self._model_name,
            device=device,
            use_fast=True,
        )
        print(f"Deepfake detector loaded on: {device.upper()}")

    def analyze(self, image: Image.Image) -> dict:
        """
        Analyze an image for deepfake content.

        Args:
            image: PIL Image to analyze

        Returns:
            dict with is_deepfake, confidence, label
        """
        # Run classification
        results = self._classifier(image)

        # Parse results
        fake_score = 0.0
        real_score = 0.0

        for result in results:
            label = result["label"].lower()
            if label in ("fake", "deepfake", "manipulated"):
                fake_score = result["score"]
            elif label in ("real", "authentic", "original"):
                real_score = result["score"]

        # Determine if deepfake
        is_deepfake = fake_score > real_score
        confidence = fake_score if is_deepfake else real_score

        return {
            "is_deepfake": is_deepfake,
            "confidence": round(confidence, 4),
            "label": "deepfake" if is_deepfake else "authentic",
        }


@lru_cache(maxsize=1)
def get_deepfake_detector() -> DeepfakeDetector:
    """Get or create the singleton deepfake detector instance."""
    return DeepfakeDetector()


def analyze_deepfake(image: Image.Image) -> dict:
    """
    Convenience function to analyze an image for deepfakes.

    Args:
        image: PIL Image to analyze

    Returns:
        Analysis result dict
    """
    detector = get_deepfake_detector()
    return detector.analyze(image)
