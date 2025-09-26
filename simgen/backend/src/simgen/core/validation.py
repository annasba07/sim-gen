"""
Request validation utilities and middleware.
Provides size limits, rate limiting, and input validation.
"""

import hashlib
import time
from typing import Dict, Optional, Any
from functools import wraps
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .exceptions import ValidationError, RateLimitError


# Request size limits
MAX_REQUEST_SIZE = 50 * 1024 * 1024  # 50MB total request
MAX_IMAGE_SIZE = 10 * 1024 * 1024    # 10MB for images
MAX_TEXT_LENGTH = 10000               # 10K characters for text
MAX_MJCF_SIZE = 5 * 1024 * 1024      # 5MB for MJCF XML


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 600
    burst_size: int = 10
    cooldown_seconds: int = 60


@dataclass
class RateLimitState:
    """State for rate limiting."""
    minute_requests: int = 0
    hour_requests: int = 0
    burst_requests: int = 0
    last_minute_reset: float = field(default_factory=time.time)
    last_hour_reset: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None


class RequestValidator:
    """Validates incoming requests."""

    @staticmethod
    def validate_image_data(image_data: str) -> bytes:
        """
        Validate and decode base64 image data.

        Args:
            image_data: Base64 encoded image string

        Returns:
            Decoded image bytes

        Raises:
            ValidationError: If validation fails
        """
        import base64

        if not image_data:
            raise ValidationError("Image data is required")

        # Handle data URL format
        if image_data.startswith('data:'):
            # Extract base64 part from data URL
            try:
                header, data = image_data.split(',', 1)
                if 'base64' not in header:
                    raise ValidationError("Invalid image data format")
                image_data = data
            except ValueError:
                raise ValidationError("Invalid data URL format")

        # Decode base64
        try:
            decoded = base64.b64decode(image_data)
        except Exception as e:
            raise ValidationError(f"Invalid base64 encoding: {str(e)}")

        # Check size
        if len(decoded) > MAX_IMAGE_SIZE:
            raise ValidationError(
                f"Image size ({len(decoded) / 1024 / 1024:.2f}MB) exceeds maximum allowed size ({MAX_IMAGE_SIZE / 1024 / 1024}MB)",
                details={"size": len(decoded), "max_size": MAX_IMAGE_SIZE}
            )

        # Validate image format
        if not (decoded.startswith(b'\\x89PNG') or
                decoded.startswith(b'\\xff\\xd8\\xff') or  # JPEG
                decoded.startswith(b'GIF')):
            raise ValidationError("Invalid image format. Supported formats: PNG, JPEG, GIF")

        return decoded

    @staticmethod
    def validate_text_prompt(prompt: str) -> str:
        """
        Validate text prompt.

        Args:
            prompt: User text prompt

        Returns:
            Validated prompt

        Raises:
            ValidationError: If validation fails
        """
        if not prompt:
            raise ValidationError("Text prompt is required")

        if len(prompt) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"Prompt length ({len(prompt)} characters) exceeds maximum allowed ({MAX_TEXT_LENGTH} characters)",
                details={"length": len(prompt), "max_length": MAX_TEXT_LENGTH}
            )

        # Strip and validate content
        prompt = prompt.strip()
        if not prompt:
            raise ValidationError("Prompt cannot be empty")

        return prompt

    @staticmethod
    def validate_mjcf_xml(mjcf_xml: str) -> str:
        """
        Validate MJCF XML content.

        Args:
            mjcf_xml: MJCF XML string

        Returns:
            Validated XML

        Raises:
            ValidationError: If validation fails
        """
        if not mjcf_xml:
            raise ValidationError("MJCF XML is required")

        if len(mjcf_xml) > MAX_MJCF_SIZE:
            raise ValidationError(
                f"MJCF XML size ({len(mjcf_xml) / 1024 / 1024:.2f}MB) exceeds maximum allowed ({MAX_MJCF_SIZE / 1024 / 1024}MB)",
                details={"size": len(mjcf_xml), "max_size": MAX_MJCF_SIZE}
            )

        # Basic XML validation
        if not mjcf_xml.strip().startswith('<'):
            raise ValidationError("Invalid XML format")

        if '<mujoco' not in mjcf_xml:
            raise ValidationError("Not a valid MuJoCo XML file")

        return mjcf_xml


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._cleanup_task = None

    async def start(self):
        """Start the rate limiter background cleanup."""
        self._cleanup_task = asyncio.create_task(self._cleanup_old_states())

    async def stop(self):
        """Stop the rate limiter."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Use IP address as client ID (in production, could use API key or user ID)
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(',')[0].strip()
        return client_ip

    async def check_rate_limit(self, request: Request) -> None:
        """
        Check if request exceeds rate limits.

        Args:
            request: FastAPI request object

        Raises:
            RateLimitError: If rate limit exceeded
        """
        client_id = self._get_client_id(request)
        state = self.states[client_id]
        current_time = time.time()

        # Check if client is blocked
        if state.blocked_until and current_time < state.blocked_until:
            retry_after = int(state.blocked_until - current_time)
            raise RateLimitError(
                f"Rate limit exceeded. Please retry after {retry_after} seconds",
                retry_after=retry_after
            )

        # Reset counters if needed
        if current_time - state.last_minute_reset > 60:
            state.minute_requests = 0
            state.last_minute_reset = current_time
            state.burst_requests = 0

        if current_time - state.last_hour_reset > 3600:
            state.hour_requests = 0
            state.last_hour_reset = current_time

        # Check burst limit
        time_since_last = current_time - state.last_request
        if time_since_last < 1:  # Less than 1 second since last request
            state.burst_requests += 1
            if state.burst_requests > self.config.burst_size:
                state.blocked_until = current_time + self.config.cooldown_seconds
                raise RateLimitError(
                    f"Burst limit exceeded. Blocked for {self.config.cooldown_seconds} seconds",
                    retry_after=self.config.cooldown_seconds
                )
        else:
            state.burst_requests = max(0, state.burst_requests - int(time_since_last))

        # Increment counters
        state.minute_requests += 1
        state.hour_requests += 1
        state.last_request = current_time

        # Check limits
        if state.minute_requests > self.config.requests_per_minute:
            retry_after = 60 - int(current_time - state.last_minute_reset)
            raise RateLimitError(
                f"Minute rate limit exceeded ({self.config.requests_per_minute} requests/minute)",
                retry_after=retry_after
            )

        if state.hour_requests > self.config.requests_per_hour:
            retry_after = 3600 - int(current_time - state.last_hour_reset)
            raise RateLimitError(
                f"Hourly rate limit exceeded ({self.config.requests_per_hour} requests/hour)",
                retry_after=retry_after
            )

    async def _cleanup_old_states(self):
        """Periodically clean up old client states."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                current_time = time.time()
                cutoff_time = current_time - 7200  # Remove states older than 2 hours

                to_remove = [
                    client_id for client_id, state in self.states.items()
                    if state.last_request < cutoff_time
                ]

                for client_id in to_remove:
                    del self.states[client_id]

                if to_remove:
                    print(f"Cleaned up {len(to_remove)} old rate limit states")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in rate limit cleanup: {e}")


# Global rate limiter instance
rate_limiter = RateLimiter()


async def validate_request_middleware(request: Request, call_next):
    """
    Middleware to validate incoming requests.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response object
    """
    # Check request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "error": "Request Too Large",
                "message": f"Request size exceeds maximum allowed ({MAX_REQUEST_SIZE / 1024 / 1024}MB)"
            }
        )

    # Check rate limits for API endpoints only
    if request.url.path.startswith("/api/"):
        try:
            await rate_limiter.check_rate_limit(request)
        except RateLimitError as e:
            return JSONResponse(
                status_code=429,
                content=e.to_dict(),
                headers={"Retry-After": str(e.retry_after)} if e.retry_after else {}
            )

    # Process request
    response = await call_next(request)
    return response


# Pydantic models with validation
class ValidatedSketchRequest(BaseModel):
    """Validated sketch analysis request."""
    sketch_data: str = Field(..., description="Base64 encoded image data")
    prompt: Optional[str] = Field(None, max_length=MAX_TEXT_LENGTH)
    include_actuators: bool = Field(default=True)
    include_sensors: bool = Field(default=True)

    @validator('sketch_data')
    def validate_sketch(cls, v):
        """Validate sketch data during parsing."""
        if not v:
            raise ValueError("Sketch data is required")
        # Basic size check (actual validation happens in RequestValidator)
        if len(v) > MAX_IMAGE_SIZE * 1.4:  # Base64 is ~1.37x larger
            raise ValueError(f"Sketch data too large")
        return v


class ValidatedPhysicsRequest(BaseModel):
    """Validated physics generation request."""
    prompt: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    sketch_data: Optional[str] = Field(None)
    use_multimodal: bool = Field(default=True)
    max_bodies: int = Field(default=10, ge=1, le=50)
    include_actuators: bool = Field(default=True)
    include_sensors: bool = Field(default=True)

    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt during parsing."""
        return RequestValidator.validate_text_prompt(v)