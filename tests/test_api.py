"""Tests for the API endpoints."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image() -> bytes:
    """Create a sample test image."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have correct structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_has_info(self, client):
        """Root response should have API info."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestAnalyzeEndpoint:
    """Tests for the analyze endpoint."""

    def test_analyze_accepts_jpeg(self, client, sample_image):
        """Analyze endpoint should accept JPEG images."""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_analyze_response_structure(self, client, sample_image):
        """Analyze response should have correct structure."""
        response = client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
        )
        data = response.json()

        assert "is_ai_generated" in data
        assert "confidence" in data
        assert "label" in data
        assert "processing_time_ms" in data

        assert isinstance(data["is_ai_generated"], bool)
        assert 0 <= data["confidence"] <= 1
        assert data["label"] in ("ai_generated", "human_made")
        assert data["processing_time_ms"] > 0

    def test_analyze_rejects_invalid_content_type(self, client):
        """Analyze endpoint should reject invalid content types."""
        response = client.post(
            "/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_analyze_accepts_png(self, client):
        """Analyze endpoint should accept PNG images."""
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/analyze",
            files={"file": ("test.png", buffer.read(), "image/png")},
        )
        assert response.status_code == 200
