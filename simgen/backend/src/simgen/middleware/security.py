"""
Production-grade security middleware for SimGen AI
Implements authentication, rate limiting, input validation, and security headers
"""

import asyncio
import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import logging
import ipaddress
from dataclasses import dataclass
import json
import re

from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis
import jwt

from ..services.resilience import SimGenError, RateLimitError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 20  # Immediate burst allowance


@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    enable_cors: bool = True
    allowed_origins: List[str] = None
    max_request_size: int = 50 * 1024 * 1024  # 50MB
    rate_limit_redis_url: str = "redis://localhost:6379/1"


class APIKeyAuth:
    """API key authentication system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api_keys = {}  # In production, use database
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from configuration/database."""
        # For demo - in production, load from secure database
        self.api_keys = {
            "demo_key_12345": {
                "user_id": "demo_user",
                "tier": "free",
                "rate_limit": RateLimitConfig(requests_per_minute=30),
                "enabled": True,
                "created_at": datetime.now()
            },
            "premium_key_67890": {
                "user_id": "premium_user", 
                "tier": "premium",
                "rate_limit": RateLimitConfig(requests_per_minute=120),
                "enabled": True,
                "created_at": datetime.now()
            }
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        if not api_key:
            return None
        
        key_info = self.api_keys.get(api_key)
        if not key_info or not key_info.get("enabled"):
            return None
        
        return key_info
    
    def generate_api_key(self, user_id: str, tier: str = "free") -> str:
        """Generate new API key."""
        api_key = f"{tier}_{secrets.token_urlsafe(32)}"
        
        rate_config = RateLimitConfig()
        if tier == "premium":
            rate_config.requests_per_minute = 120
            rate_config.requests_per_hour = 5000
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "tier": tier,
            "rate_limit": rate_config,
            "enabled": True,
            "created_at": datetime.now()
        }
        
        return api_key


class RateLimiter:
    """Production-grade rate limiter with Redis backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_available = True
            logger.info("Rate limiter Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable for rate limiting: {e}")
            self.redis_available = False
            self.local_cache = {}  # Fallback to local memory
    
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        request_weight: int = 1
    ) -> Dict[str, Any]:
        """Check if request is within rate limits."""
        
        current_time = time.time()
        
        if self.redis_available:
            return await self._check_redis_rate_limit(key, config, request_weight, current_time)
        else:
            return await self._check_local_rate_limit(key, config, request_weight, current_time)
    
    async def _check_redis_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        weight: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Redis-based rate limiting with sliding window."""
        
        pipe = self.redis_client.pipeline()
        
        # Sliding window rate limiting
        minute_key = f"rate_limit:{key}:minute"
        hour_key = f"rate_limit:{key}:hour"
        day_key = f"rate_limit:{key}:day"
        
        # Remove old entries
        minute_start = current_time - 60
        hour_start = current_time - 3600
        day_start = current_time - 86400
        
        pipe.zremrangebyscore(minute_key, 0, minute_start)
        pipe.zremrangebyscore(hour_key, 0, hour_start)
        pipe.zremrangebyscore(day_key, 0, day_start)
        
        # Count current requests
        pipe.zcard(minute_key)
        pipe.zcard(hour_key)
        pipe.zcard(day_key)
        
        results = pipe.execute()
        
        minute_count = results[3]
        hour_count = results[4]
        day_count = results[5]
        
        # Check limits
        if minute_count + weight > config.requests_per_minute:
            reset_time = minute_start + 60
            raise RateLimitError(
                "Rate limit exceeded: too many requests per minute",
                "RATE_LIMIT_MINUTE",
                {
                    "limit": config.requests_per_minute,
                    "current": minute_count,
                    "reset_time": reset_time,
                    "retry_after": max(0, reset_time - current_time)
                }
            )
        
        if hour_count + weight > config.requests_per_hour:
            reset_time = hour_start + 3600
            raise RateLimitError(
                "Rate limit exceeded: too many requests per hour",
                "RATE_LIMIT_HOUR",
                {
                    "limit": config.requests_per_hour,
                    "current": hour_count,
                    "reset_time": reset_time,
                    "retry_after": max(0, reset_time - current_time)
                }
            )
        
        if day_count + weight > config.requests_per_day:
            reset_time = day_start + 86400
            raise RateLimitError(
                "Rate limit exceeded: too many requests per day",
                "RATE_LIMIT_DAY",
                {
                    "limit": config.requests_per_day,
                    "current": day_count,
                    "reset_time": reset_time,
                    "retry_after": max(0, reset_time - current_time)
                }
            )
        
        # Add current request
        request_id = f"{current_time}:{secrets.token_hex(8)}"
        pipe = self.redis_client.pipeline()
        pipe.zadd(minute_key, {request_id: current_time})
        pipe.zadd(hour_key, {request_id: current_time})
        pipe.zadd(day_key, {request_id: current_time})
        pipe.expire(minute_key, 60)
        pipe.expire(hour_key, 3600)
        pipe.expire(day_key, 86400)
        pipe.execute()
        
        return {
            "allowed": True,
            "minute_count": minute_count + weight,
            "hour_count": hour_count + weight,
            "day_count": day_count + weight,
            "limits": {
                "minute": config.requests_per_minute,
                "hour": config.requests_per_hour,
                "day": config.requests_per_day
            }
        }
    
    async def _check_local_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        weight: int,
        current_time: float
    ) -> Dict[str, Any]:
        """Fallback local memory rate limiting."""
        
        if key not in self.local_cache:
            self.local_cache[key] = {"requests": [], "last_cleanup": current_time}
        
        user_data = self.local_cache[key]
        
        # Cleanup old requests every minute
        if current_time - user_data["last_cleanup"] > 60:
            minute_start = current_time - 60
            user_data["requests"] = [
                req for req in user_data["requests"] 
                if req["timestamp"] > minute_start
            ]
            user_data["last_cleanup"] = current_time
        
        # Count recent requests
        minute_start = current_time - 60
        recent_requests = [
            req for req in user_data["requests"]
            if req["timestamp"] > minute_start
        ]
        
        if len(recent_requests) + weight > config.requests_per_minute:
            raise RateLimitError(
                "Rate limit exceeded (local): too many requests per minute",
                "RATE_LIMIT_MINUTE_LOCAL",
                {"limit": config.requests_per_minute, "current": len(recent_requests)}
            )
        
        # Add current request
        user_data["requests"].append({"timestamp": current_time, "weight": weight})
        
        return {
            "allowed": True,
            "minute_count": len(recent_requests) + weight,
            "limits": {"minute": config.requests_per_minute}
        }


class InputValidator:
    """Input validation and sanitization."""
    
    @staticmethod
    def validate_request_size(content_length: Optional[int], max_size: int):
        """Validate request size."""
        if content_length and content_length > max_size:
            raise ValidationError(
                f"Request too large: {content_length} bytes (max: {max_size})",
                "REQUEST_TOO_LARGE",
                {"size": content_length, "max_size": max_size}
            )
    
    @staticmethod
    def validate_sketch_data(sketch_data: str) -> bool:
        """Validate base64 sketch data."""
        if not sketch_data:
            raise ValidationError("Sketch data is required", "MISSING_SKETCH_DATA")
        
        # Basic base64 validation
        import base64
        try:
            decoded = base64.b64decode(sketch_data)
            
            # Check for reasonable image size (100 bytes to 10MB)
            if len(decoded) < 100:
                raise ValidationError("Sketch data too small", "INVALID_SKETCH_SIZE")
            
            if len(decoded) > 10 * 1024 * 1024:  # 10MB
                raise ValidationError("Sketch data too large", "INVALID_SKETCH_SIZE")
            
            return True
            
        except Exception as e:
            raise ValidationError(
                f"Invalid sketch data: {str(e)}",
                "INVALID_SKETCH_DATA"
            )
    
    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """Validate user prompt."""
        if not prompt or not prompt.strip():
            return True  # Optional field
        
        if len(prompt) > 2000:
            raise ValidationError(
                "Prompt too long (max 2000 characters)",
                "PROMPT_TOO_LONG",
                {"length": len(prompt), "max_length": 2000}
            )
        
        # Check for potential injection attacks
        suspicious_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise ValidationError(
                    "Prompt contains suspicious content",
                    "SUSPICIOUS_PROMPT_CONTENT"
                )
        
        return True
    
    @staticmethod
    def sanitize_user_input(text: str) -> str:
        """Sanitize user input."""
        if not text:
            return ""
        
        # Remove or escape dangerous characters
        sanitized = text.strip()
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Limit length
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000]
        
        return sanitized


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware."""
    
    def __init__(
        self,
        app,
        config: SecurityConfig,
        api_key_auth: APIKeyAuth,
        rate_limiter: RateLimiter
    ):
        super().__init__(app)
        self.config = config
        self.api_key_auth = api_key_auth
        self.rate_limiter = rate_limiter
        self.validator = InputValidator()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security pipeline."""
        
        try:
            # 1. Validate request size
            content_length = request.headers.get("content-length")
            if content_length:
                self.validator.validate_request_size(
                    int(content_length),
                    self.config.max_request_size
                )
            
            # 2. Extract client identifier
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "unknown")
            
            # 3. Authenticate request
            auth_info = await self._authenticate_request(request)
            
            # 4. Apply rate limiting
            rate_limit_key = auth_info.get("user_id", client_ip)
            rate_config = auth_info.get("rate_limit", RateLimitConfig())
            
            await self.rate_limiter.check_rate_limit(rate_limit_key, rate_config)
            
            # 5. Add security headers to request
            request.state.auth_info = auth_info
            request.state.client_ip = client_ip
            
            # 6. Process request
            response = await call_next(request)
            
            # 7. Add security headers to response
            self._add_security_headers(response)
            
            return response
            
        except (RateLimitError, ValidationError, SimGenError) as e:
            return self._create_error_response(e)
        
        except Exception as e:
            logger.error(f"Security middleware error: {e}", exc_info=True)
            return self._create_error_response(
                SimGenError("Internal security error", "SECURITY_ERROR")
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP address."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return getattr(request.client, "host", "unknown")
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate the request."""
        
        # Check for API key in header
        api_key = request.headers.get(self.config.api_key_header)
        
        if api_key:
            auth_info = self.api_key_auth.validate_api_key(api_key)
            if auth_info:
                return auth_info
            else:
                raise ValidationError("Invalid API key", "INVALID_API_KEY")
        
        # Check for JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = jwt.decode(
                    token,
                    self.config.jwt_secret_key,
                    algorithms=[self.config.jwt_algorithm]
                )
                return {
                    "user_id": payload.get("user_id"),
                    "tier": payload.get("tier", "free"),
                    "rate_limit": RateLimitConfig()
                }
            except jwt.InvalidTokenError:
                raise ValidationError("Invalid JWT token", "INVALID_TOKEN")
        
        # Anonymous access with basic rate limits
        return {
            "user_id": "anonymous",
            "tier": "anonymous", 
            "rate_limit": RateLimitConfig(requests_per_minute=10)
        }
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "X-SimGen-API-Version": "1.0"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
    
    def _create_error_response(self, error: SimGenError) -> JSONResponse:
        """Create standardized error response."""
        
        status_code = 429 if isinstance(error, RateLimitError) else 400
        
        response_data = {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add retry-after header for rate limits
        if isinstance(error, RateLimitError) and error.details.get("retry_after"):
            response = JSONResponse(
                content=response_data,
                status_code=status_code,
                headers={"Retry-After": str(int(error.details["retry_after"]))}
            )
        else:
            response = JSONResponse(content=response_data, status_code=status_code)
        
        self._add_security_headers(response)
        return response


# Dependency injection helpers
def get_current_user(request: Request) -> Dict[str, Any]:
    """Get authenticated user from request."""
    return getattr(request.state, "auth_info", {"user_id": "anonymous"})


def require_api_key(request: Request) -> Dict[str, Any]:
    """Require valid API key."""
    auth_info = get_current_user(request)
    if auth_info.get("user_id") == "anonymous":
        raise HTTPException(status_code=401, detail="API key required")
    return auth_info


# Global instances
_security_config = None
_api_key_auth = None
_rate_limiter = None

def initialize_security(
    jwt_secret_key: str,
    redis_url: str = "redis://localhost:6379/1"
) -> tuple:
    """Initialize security components."""
    global _security_config, _api_key_auth, _rate_limiter
    
    _security_config = SecurityConfig(
        jwt_secret_key=jwt_secret_key,
        rate_limit_redis_url=redis_url
    )
    
    _api_key_auth = APIKeyAuth(_security_config)
    _rate_limiter = RateLimiter(redis_url)
    
    return _security_config, _api_key_auth, _rate_limiter


def get_security_components():
    """Get initialized security components."""
    if not _security_config:
        raise RuntimeError("Security not initialized. Call initialize_security() first.")
    
    return _security_config, _api_key_auth, _rate_limiter