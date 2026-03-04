"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    """Response model for image analysis endpoint."""

    is_ai_generated: bool = Field(
        description="Whether the image is detected as AI-generated"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the detection",
    )
    label: str = Field(
        description="Classification label: 'ai_generated' or 'human_made'"
    )
    processing_time_ms: float = Field(
        description="Time taken to process the image in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_ai_generated": True,
                "confidence": 0.87,
                "label": "ai_generated",
                "processing_time_ms": 234.5,
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether the AI model is loaded")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "model_loaded": True,
            }
        }
    }


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    detail: str = Field(description="Error message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Invalid image format. Supported formats: JPEG, PNG, WebP"
            }
        }
    }


class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis endpoint."""

    veracity_score: int = Field(
        ge=0,
        le=100,
        description="Veracity score (0-100). Higher = more likely to be authentic/real",
    )
    fraud_detected: bool = Field(
        description="Whether fraud/AI manipulation was detected in the video",
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence level of the detection (0-100)",
    )
    fraud_types: list[str] = Field(
        default_factory=list,
        description="List of fraud types detected (e.g., 'ai_generated', 'deepfake', 'face_swap')",
    )
    details: str = Field(
        description="Detailed description of the analysis results",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "veracity_score": 15,
                "fraud_detected": True,
                "confidence": 87,
                "fraud_types": ["ai_generated"],
                "details": "Video analysis detected 38 of 45 frames (84.4%) as AI-generated content. High confidence of artificial manipulation detected.",
            }
        }
    }


class VideoAnalysisDetailedResponse(BaseModel):
    """Detailed response model for video analysis (includes metadata)."""

    veracity_score: int = Field(
        ge=0,
        le=100,
        description="Veracity score (0-100). Higher = more likely to be authentic/real",
    )
    fraud_detected: bool = Field(
        description="Whether fraud/AI manipulation was detected in the video",
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence level of the detection (0-100)",
    )
    total_frames_analyzed: int = Field(
        description="Total number of keyframes analyzed"
    )
    ai_generated_frames: int = Field(
        description="Number of frames detected as AI-generated"
    )
    video_duration_seconds: float = Field(
        description="Duration of the video in seconds"
    )
    processing_time_ms: float = Field(
        description="Total time taken to process the video in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "veracity_score": 15,
                "fraud_detected": True,
                "confidence": 87,
                "total_frames_analyzed": 45,
                "ai_generated_frames": 38,
                "video_duration_seconds": 120.5,
                "processing_time_ms": 8543.2,
            }
        }
    }


class VideoURLRequest(BaseModel):
    """Request model for analyzing video from URL."""

    url: str = Field(
        description="URL to fetch the video from",
    )
    authorization: str | None = Field(
        default=None,
        description="Authorization header value (e.g., 'Bearer token123')",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "https://api.example.com/video/12345",
                "authorization": "Bearer 0dde98fe-779f-4e41-9bce-5c1d878e153d",
            }
        }
    }
