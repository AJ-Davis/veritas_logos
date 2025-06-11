"""
Enhanced middleware for the Veritas Logos API Gateway.

Provides logging, request validation, security enhancements, and monitoring.
"""

import time
import uuid
import logging
import re
import json
from typing import Callable, Any, Dict, Optional
from datetime import datetime

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request/response logging middleware with performance metrics."""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()
        
        # Log request details
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")
        
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {client_ip} - Agent: {user_agent[:100]}"
        )
        
        # Log query parameters (excluding sensitive data)
        if request.query_params:
            safe_params = self._sanitize_params(dict(request.query_params))
            logger.debug(f"[{request_id}] Query params: {safe_params}")
        
        # Log headers (excluding sensitive data)
        if logger.level <= logging.DEBUG:
            safe_headers = self._sanitize_headers(dict(request.headers))
            logger.debug(f"[{request_id}] Headers: {safe_headers}")
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] Exception occurred: {str(e)} - "
                f"Took {process_time:.4f}s"
            )
            raise
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"[{request_id}] Response: {response.status_code} - "
            f"Took {process_time:.4f}s"
        )
        
        # Add performance and tracing headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{process_time:.4f}"
        response.headers["X-Timestamp"] = start_timestamp
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxy headers."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from query parameters."""
        sensitive_keys = {"password", "token", "secret", "key", "auth"}
        return {
            k: "***REDACTED***" if any(sens in k.lower() for sens in sensitive_keys) else v
            for k, v in params.items()
        }
    
    def _sanitize_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from headers."""
        sensitive_headers = {"authorization", "cookie", "x-api-key", "x-auth-token"}
        return {
            k: "***REDACTED***" if k.lower() in sensitive_headers else v
            for k, v in headers.items()
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with request validation and sanitization."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_request_size = max_request_size
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(f"Request too large: {content_length} bytes")
            return JSONResponse(
                status_code=413,
                content={"error": "Request entity too large"}
            )
        
        # Validate and sanitize request path
        if not self._validate_path(request.url.path):
            logger.warning(f"Suspicious request path: {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request path"}
            )
        
        # Check for common attack patterns
        if self._detect_attack_patterns(request):
            logger.warning(f"Potential attack detected from {request.client.host}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"}
            )
        
        return await call_next(request)
    
    def _validate_path(self, path: str) -> bool:
        """Validate request path for security issues."""
        # Check for path traversal attempts
        if ".." in path or "~" in path:
            return False
        
        # Check for suspicious characters
        suspicious_chars = ["<", ">", "\"", "'", "&", ";", "|"]
        if any(char in path for char in suspicious_chars):
            return False
        
        return True
    
    def _detect_attack_patterns(self, request: Request) -> bool:
        """Detect common attack patterns in requests."""
        # SQL injection patterns
        sql_patterns = [
            r"union\s+select", r"drop\s+table", r"insert\s+into",
            r"delete\s+from", r"update\s+set", r"--", r"/\*"
        ]
        
        # XSS patterns
        xss_patterns = [
            r"<script", r"javascript:", r"onload=", r"onerror=",
            r"eval\(", r"alert\(", r"document\."
        ]
        
        # Check query parameters
        query_string = str(request.query_params)
        for pattern in sql_patterns + xss_patterns:
            if re.search(pattern, query_string, re.IGNORECASE):
                return True
        
        return False


class RequestValidator:
    """Request validation and sanitization utilities."""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input by removing dangerous characters."""
        if not isinstance(input_str, str):
            return str(input_str)
        
        # Truncate if too long
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&;"\'\\]', '', input_str)
        
        # Remove control characters except tab, newline, carriage return
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """Validate UUID format."""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, uuid_str, re.IGNORECASE))
    
    @staticmethod
    def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize request data."""
        if isinstance(data, dict):
            return {
                key: RequestValidator.sanitize_request_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [RequestValidator.sanitize_request_data(item) for item in data]
        elif isinstance(data, str):
            return RequestValidator.sanitize_string(data)
        else:
            return data


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.response_times = []
        self.status_codes = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Increment request counter
        self.request_count += 1
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Count status codes
        status_code = response.status_code
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            max_response_time = max(self.response_times)
            min_response_time = min(self.response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        return {
            "total_requests": self.request_count,
            "average_response_time": round(avg_response_time, 4),
            "max_response_time": round(max_response_time, 4),
            "min_response_time": round(min_response_time, 4),
            "status_codes": self.status_codes,
            "recent_requests": len(self.response_times)
        }


# Global metrics instance
metrics_middleware = None

def get_metrics() -> Dict[str, Any]:
    """Get API metrics."""
    if metrics_middleware:
        return metrics_middleware.get_metrics()
    return {"error": "Metrics not available"} 