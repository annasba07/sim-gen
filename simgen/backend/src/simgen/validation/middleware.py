"""
API Validation Middleware
Provides comprehensive request/response validation, sanitization, and security checks
"""

import json
import logging
import re
import time
from typing import Dict, Any, Optional, List, Union
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, ValidationError
import bleach

from ..monitoring.observability import get_observability_manager


logger = logging.getLogger(__name__)


class ValidationConfig(BaseModel):
    """Configuration for validation middleware."""
    
    # Request validation
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_json_depth: int = 10
    max_array_length: int = 1000
    max_string_length: int = 10000
    
    # Content filtering
    enable_html_sanitization: bool = True
    enable_sql_injection_check: bool = True
    enable_xss_protection: bool = True
    enable_path_traversal_check: bool = True
    
    # Rate limiting per endpoint
    enable_endpoint_rate_limiting: bool = True
    default_endpoint_limit: int = 100  # requests per minute
    
    # Response validation
    validate_responses: bool = True
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    
    # Security headers
    add_security_headers: bool = True


class SecurityThreat(BaseModel):
    """Detected security threat information."""
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    detected_patterns: List[str]
    recommended_action: str


class ValidationResult(BaseModel):
    """Result of validation checks."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    security_threats: List[SecurityThreat] = []
    sanitized_data: Optional[Dict[str, Any]] = None


class APIValidationMiddleware(BaseHTTPMiddleware):
    """Comprehensive API validation and security middleware."""
    
    def __init__(self, app, config: Optional[ValidationConfig] = None):
        super().__init__(app)
        self.config = config or ValidationConfig()
        self.observability = get_observability_manager()
        
        # Security patterns
        self.sql_patterns = [
            r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b',
            r'[\'";].*[\'";]',
            r'--.*',
            r'/\*.*\*/',
        ]
        
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
        ]
        
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e\\',
        ]
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/simulation/generate": 10,  # Lower limit for expensive operations
            "/api/auth/login": 5,
            "/api/sketch/analyze": 20,
            "/api/monitoring/metrics": 60
        }
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method."""
        start_time = time.time()
        
        try:
            # Pre-request validation
            validation_result = await self._validate_request(request)
            
            if not validation_result.is_valid:
                return await self._create_validation_error_response(validation_result)
            
            # Process request with sanitized data if needed
            if validation_result.sanitized_data:
                request._body = json.dumps(validation_result.sanitized_data).encode()
            
            # Call the next middleware/endpoint
            response = await call_next(request)
            
            # Post-response validation
            if self.config.validate_responses:
                await self._validate_response(response, request.url.path)
            
            # Add security headers
            if self.config.add_security_headers:
                self._add_security_headers(response)
            
            # Track metrics
            processing_time = time.time() - start_time
            self.observability.metrics_collector.timer("api.validation.time", processing_time * 1000)
            self.observability.metrics_collector.increment("api.validation.success")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Validation middleware error: {e}")
            
            self.observability.metrics_collector.increment("api.validation.error")
            self.observability.metrics_collector.timer("api.validation.error_time", processing_time * 1000)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Validation middleware error",
                    "message": "Internal validation error occurred",
                    "timestamp": time.time()
                }
            )
    
    async def _validate_request(self, request: Request) -> ValidationResult:
        """Comprehensive request validation."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.config.max_request_size:
                result.is_valid = False
                result.errors.append(f"Request size exceeds maximum allowed ({self.config.max_request_size} bytes)")
                result.security_threats.append(SecurityThreat(
                    threat_type="resource_exhaustion",
                    severity="medium",
                    description="Request size exceeds safe limits",
                    detected_patterns=[f"size: {content_length}"],
                    recommended_action="Reject request and log incident"
                ))
            
            # Validate URL path
            path_validation = self._validate_path(str(request.url.path))
            if path_validation.security_threats:
                result.security_threats.extend(path_validation.security_threats)
                result.is_valid = False
            
            # Validate query parameters
            for key, value in request.query_params.items():
                param_validation = self._validate_parameter(key, value, "query")
                if param_validation.security_threats:
                    result.security_threats.extend(param_validation.security_threats)
                    if param_validation.severity in ["high", "critical"]:
                        result.is_valid = False
            
            # Validate request body if present
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    body_validation = await self._validate_body(body)
                    result.errors.extend(body_validation.errors)
                    result.warnings.extend(body_validation.warnings)
                    result.security_threats.extend(body_validation.security_threats)
                    
                    if not body_validation.is_valid:
                        result.is_valid = False
                    
                    if body_validation.sanitized_data:
                        result.sanitized_data = body_validation.sanitized_data
            
            # Check for critical security threats
            critical_threats = [t for t in result.security_threats if t.severity == "critical"]
            if critical_threats:
                result.is_valid = False
                
                # Log security incident
                logger.critical(f"Critical security threat detected: {critical_threats}")
                self.observability.metrics_collector.increment("security.threat.critical")
        
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            result.is_valid = False
            result.errors.append(f"Validation processing error: {str(e)}")
        
        return result
    
    def _validate_path(self, path: str) -> ValidationResult:
        """Validate URL path for security issues."""
        result = ValidationResult(is_valid=True)
        
        # Check for path traversal attempts
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                result.security_threats.append(SecurityThreat(
                    threat_type="path_traversal",
                    severity="high",
                    description="Path traversal attempt detected",
                    detected_patterns=[pattern],
                    recommended_action="Block request and log incident"
                ))
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&', '%00', '%2e%2e']
        for char in suspicious_chars:
            if char in path:
                result.security_threats.append(SecurityThreat(
                    threat_type="malformed_path",
                    severity="medium",
                    description=f"Suspicious character in path: {char}",
                    detected_patterns=[char],
                    recommended_action="Validate and sanitize path"
                ))
        
        return result
    
    def _validate_parameter(self, key: str, value: str, param_type: str) -> ValidationResult:
        """Validate query/form parameters."""
        result = ValidationResult(is_valid=True)
        result.severity = "low"
        
        # Check parameter length
        if len(value) > self.config.max_string_length:
            result.warnings.append(f"Parameter '{key}' exceeds maximum length")
            result.severity = "medium"
        
        # SQL injection check
        if self.config.enable_sql_injection_check:
            for pattern in self.sql_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    result.security_threats.append(SecurityThreat(
                        threat_type="sql_injection",
                        severity="high",
                        description=f"Potential SQL injection in parameter '{key}'",
                        detected_patterns=[pattern],
                        recommended_action="Sanitize or reject parameter"
                    ))
                    result.severity = "high"
        
        # XSS check
        if self.config.enable_xss_protection:
            for pattern in self.xss_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    result.security_threats.append(SecurityThreat(
                        threat_type="xss",
                        severity="high",
                        description=f"Potential XSS in parameter '{key}'",
                        detected_patterns=[pattern],
                        recommended_action="Sanitize HTML content"
                    ))
                    result.severity = "high"
        
        return result
    
    async def _validate_body(self, body: bytes) -> ValidationResult:
        """Validate request body content."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check if it's JSON
            if body.startswith(b'{') or body.startswith(b'['):
                body_str = body.decode('utf-8')
                
                # Check for JSON depth and complexity
                try:
                    data = json.loads(body_str)
                    
                    # Validate JSON structure
                    json_validation = self._validate_json_structure(data)
                    result.errors.extend(json_validation.errors)
                    result.warnings.extend(json_validation.warnings)
                    result.security_threats.extend(json_validation.security_threats)
                    
                    if not json_validation.is_valid:
                        result.is_valid = False
                    
                    # Sanitize if needed
                    if self.config.enable_html_sanitization:
                        sanitized_data = self._sanitize_data(data)
                        if sanitized_data != data:
                            result.sanitized_data = sanitized_data
                            result.warnings.append("Content was sanitized for security")
                
                except json.JSONDecodeError as e:
                    result.is_valid = False
                    result.errors.append(f"Invalid JSON format: {str(e)}")
            
            # Check for security threats in raw body
            body_str = body.decode('utf-8', errors='ignore')
            security_validation = self._check_security_threats(body_str)
            result.security_threats.extend(security_validation.security_threats)
            
            if security_validation.security_threats:
                critical_threats = [t for t in security_validation.security_threats if t.severity == "critical"]
                if critical_threats:
                    result.is_valid = False
        
        except Exception as e:
            logger.error(f"Body validation error: {e}")
            result.is_valid = False
            result.errors.append(f"Body validation processing error: {str(e)}")
        
        return result
    
    def _validate_json_structure(self, data: Any, depth: int = 0) -> ValidationResult:
        """Validate JSON structure for depth, size, and content."""
        result = ValidationResult(is_valid=True)
        
        # Check depth
        if depth > self.config.max_json_depth:
            result.is_valid = False
            result.errors.append(f"JSON nesting depth exceeds maximum ({self.config.max_json_depth})")
            return result
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Validate key
                if len(str(key)) > self.config.max_string_length:
                    result.warnings.append(f"JSON key '{key}' exceeds maximum length")
                
                # Recursively validate value
                sub_result = self._validate_json_structure(value, depth + 1)
                result.errors.extend(sub_result.errors)
                result.warnings.extend(sub_result.warnings)
                result.security_threats.extend(sub_result.security_threats)
                
                if not sub_result.is_valid:
                    result.is_valid = False
        
        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                result.warnings.append(f"Array length ({len(data)}) exceeds recommended maximum")
            
            for item in data:
                sub_result = self._validate_json_structure(item, depth + 1)
                result.errors.extend(sub_result.errors)
                result.warnings.extend(sub_result.warnings)
                result.security_threats.extend(sub_result.security_threats)
                
                if not sub_result.is_valid:
                    result.is_valid = False
        
        elif isinstance(data, str):
            if len(data) > self.config.max_string_length:
                result.warnings.append("String value exceeds maximum length")
            
            # Check for security threats in string
            threat_result = self._check_security_threats(data)
            result.security_threats.extend(threat_result.security_threats)
        
        return result
    
    def _check_security_threats(self, content: str) -> ValidationResult:
        """Check content for various security threats."""
        result = ValidationResult(is_valid=True)
        
        # SQL injection check
        if self.config.enable_sql_injection_check:
            for pattern in self.sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result.security_threats.append(SecurityThreat(
                        threat_type="sql_injection",
                        severity="high",
                        description="Potential SQL injection detected",
                        detected_patterns=[pattern],
                        recommended_action="Sanitize or reject content"
                    ))
        
        # XSS check
        if self.config.enable_xss_protection:
            for pattern in self.xss_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    result.security_threats.append(SecurityThreat(
                        threat_type="xss",
                        severity="high",
                        description="Potential XSS attack detected",
                        detected_patterns=[pattern],
                        recommended_action="Sanitize HTML content"
                    ))
        
        return result
    
    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize data structure."""
        if isinstance(data, dict):
            return {key: self._sanitize_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Use bleach to sanitize HTML content
            return bleach.clean(data, tags=[], attributes={}, strip=True)
        else:
            return data
    
    async def _validate_response(self, response: Response, endpoint: str) -> None:
        """Validate response content and size."""
        try:
            # Check response size if it's JSON
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.config.max_response_size:
                logger.warning(f"Response size ({content_length}) exceeds maximum for endpoint {endpoint}")
                self.observability.metrics_collector.increment("api.response.oversized")
        
        except Exception as e:
            logger.error(f"Response validation error: {e}")
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        for header, value in security_headers.items():
            if header not in response.headers:
                response.headers[header] = value
    
    async def _create_validation_error_response(self, result: ValidationResult) -> JSONResponse:
        """Create error response for validation failures."""
        
        # Determine status code based on threat severity
        status_code = 400  # Bad Request
        
        if any(t.severity == "critical" for t in result.security_threats):
            status_code = 403  # Forbidden
        elif any(t.severity == "high" for t in result.security_threats):
            status_code = 422  # Unprocessable Entity
        
        # Log security threats
        if result.security_threats:
            threat_summary = {
                "threats": len(result.security_threats),
                "critical": sum(1 for t in result.security_threats if t.severity == "critical"),
                "high": sum(1 for t in result.security_threats if t.severity == "high"),
                "patterns": [t.threat_type for t in result.security_threats]
            }
            logger.warning(f"Security threats detected: {threat_summary}")
            self.observability.metrics_collector.increment("security.threats.detected")
        
        # Create sanitized error response (don't expose internal details)
        response_content = {
            "error": "Validation failed",
            "message": "Request validation failed due to security or format issues",
            "timestamp": time.time()
        }
        
        # Add non-sensitive error details
        if result.errors:
            response_content["validation_errors"] = [
                error for error in result.errors 
                if "internal" not in error.lower()
            ]
        
        return JSONResponse(
            status_code=status_code,
            content=response_content
        )