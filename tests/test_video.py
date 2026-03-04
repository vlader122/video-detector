"""Tests for the video analysis functionality."""

import pytest
from PIL import Image

from app.video_processor import validate_video_content_type
from app.detector import analyze_video_frames


class TestVideoContentTypeValidation:
    """Tests for video content type validation."""

    def test_valid_mp4_content_type(self):
        """MP4 content type should be valid."""
        assert validate_video_content_type("video/mp4") is True

    def test_valid_webm_content_type(self):
        """WebM content type should be valid."""
        assert validate_video_content_type("video/webm") is True

    def test_invalid_content_type(self):
        """Image content types should be invalid for video."""
        assert validate_video_content_type("image/jpeg") is False
        assert validate_video_content_type("image/png") is False

    def test_none_content_type(self):
        """None content type should be invalid."""
        assert validate_video_content_type(None) is False

    def test_text_content_type(self):
        """Text content type should be invalid."""
        assert validate_video_content_type("text/plain") is False


class TestAnalyzeVideoFrames:
    """Tests for the analyze_video_frames function."""

    @pytest.fixture
    def sample_frames(self) -> list[Image.Image]:
        """Create sample test frames."""
        frames = []
        for i in range(5):
            # Create simple colored images
            color = (i * 50, 100, 150)
            img = Image.new("RGB", (100, 100), color=color)
            frames.append(img)
        return frames

    def test_empty_frames_returns_default_values(self):
        """Empty frame list should return default values."""
        result = analyze_video_frames([], video_duration=10.0)

        assert result["veracity_score"] == 100
        assert result["fraud_detected"] is False
        assert result["confidence"] == 0

    def test_result_has_required_fields(self, sample_frames):
        """Result should contain all required fields."""
        result = analyze_video_frames(sample_frames, video_duration=5.0)

        assert "veracity_score" in result
        assert "fraud_detected" in result
        assert "confidence" in result

    def test_detailed_result_has_metadata(self, sample_frames):
        """Detailed result should include metadata."""
        result = analyze_video_frames(sample_frames, video_duration=5.0, detailed=True)

        assert "veracity_score" in result
        assert "fraud_detected" in result
        assert "confidence" in result
        assert "total_frames_analyzed" in result
        assert "ai_generated_frames" in result
        assert "video_duration_seconds" in result
        assert "processing_time_ms" in result

    def test_veracity_score_in_valid_range(self, sample_frames):
        """Veracity score should be between 0 and 100."""
        result = analyze_video_frames(sample_frames, video_duration=5.0)
        assert 0 <= result["veracity_score"] <= 100

    def test_confidence_in_valid_range(self, sample_frames):
        """Confidence should be between 0 and 100."""
        result = analyze_video_frames(sample_frames, video_duration=5.0)
        assert 0 <= result["confidence"] <= 100

    def test_fraud_detected_is_boolean(self, sample_frames):
        """Fraud detected should be a boolean."""
        result = analyze_video_frames(sample_frames, video_duration=5.0)
        assert isinstance(result["fraud_detected"], bool)


class TestVideoEndpoint:
    """Tests for the video analysis endpoint."""

    def test_rejects_invalid_content_type(self):
        """Endpoint should reject non-video content types."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # Create a dummy file with image content type
        response = client.post(
            "/analyze/video",
            files={"file": ("test.jpg", b"fake video data", "image/jpeg")},
        )

        assert response.status_code == 400
        assert "Invalid content type" in response.json()["detail"]


class TestVideoURLEndpoint:
    """Tests for the video URL analysis endpoint."""

    def test_url_endpoint_exists(self):
        """URL endpoint should exist."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        # Test with invalid URL (will fail but endpoint should exist)
        response = client.post(
            "/analyze/video/url",
            json={"url": "http://invalid-url-that-does-not-exist.com/video.mp4"},
        )

        # Should return 502 (bad gateway) not 404 (not found)
        assert response.status_code != 404
