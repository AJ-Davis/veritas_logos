"""
API Versioning module for Veritas Logos.

Provides support for multiple API versions with backward compatibility.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re
from datetime import datetime


class VersionInfo(BaseModel):
    """API version information."""
    version: str
    description: str
    deprecated: bool = False
    deprecated_date: Optional[str] = None
    sunset_date: Optional[str] = None
    changelog_url: Optional[str] = None


class APIVersionManager:
    """Manages API versions and routing."""
    
    def __init__(self):
        self.versions: Dict[str, VersionInfo] = {}
        self.routers: Dict[str, APIRouter] = {}
        self.current_version = "v1"
        
        # Register initial versions
        self._register_default_versions()
    
    def _register_default_versions(self):
        """Register default API versions."""
        self.register_version(
            "v1",
            VersionInfo(
                version="1.0.0",
                description="Initial API version with document verification capabilities",
                deprecated=False
            )
        )
        
        # Prepare for future v2
        self.register_version(
            "v2",
            VersionInfo(
                version="2.0.0",
                description="Enhanced API with improved verification algorithms and batch processing",
                deprecated=False
            )
        )
    
    def register_version(self, version_key: str, version_info: VersionInfo):
        """Register a new API version."""
        self.versions[version_key] = version_info
        self.routers[version_key] = APIRouter(prefix=f"/api/{version_key}")
    
    def get_router(self, version: str) -> APIRouter:
        """Get router for specific version."""
        if version not in self.routers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API version {version} not found"
            )
        return self.routers[version]
    
    def get_version_info(self, version: str) -> VersionInfo:
        """Get information about a specific version."""
        if version not in self.versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API version {version} not found"
            )
        return self.versions[version]
    
    def list_versions(self) -> Dict[str, VersionInfo]:
        """List all available API versions."""
        return self.versions
    
    def extract_version_from_path(self, path: str) -> Optional[str]:
        """Extract version from request path."""
        match = re.match(r'/api/(v\d+)/', path)
        return match.group(1) if match else None
    
    def extract_version_from_header(self, request: Request) -> Optional[str]:
        """Extract version from Accept header."""
        accept_header = request.headers.get("accept", "")
        version_match = re.search(r'application/vnd\.veritaslogos\.([^+]+)', accept_header)
        return version_match.group(1) if version_match else None
    
    def get_requested_version(self, request: Request) -> str:
        """Determine which API version is requested."""
        # 1. Try to get version from path
        path_version = self.extract_version_from_path(request.url.path)
        if path_version and path_version in self.versions:
            return path_version
        
        # 2. Try to get version from Accept header
        header_version = self.extract_version_from_header(request)
        if header_version and header_version in self.versions:
            return header_version
        
        # 3. Fall back to current version
        return self.current_version
    
    def deprecate_version(self, version: str, deprecated_date: str, sunset_date: str):
        """Mark a version as deprecated."""
        if version in self.versions:
            self.versions[version].deprecated = True
            self.versions[version].deprecated_date = deprecated_date
            self.versions[version].sunset_date = sunset_date
    
    def create_version_response(self, data: Any, version: str, request: Request) -> JSONResponse:
        """Create a response with version information."""
        response_data = {
            "data": data,
            "meta": {
                "version": version,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
        
        # Add deprecation warnings if applicable
        version_info = self.versions.get(version)
        if version_info and version_info.deprecated:
            response_data["meta"]["deprecated"] = True
            response_data["meta"]["deprecated_date"] = version_info.deprecated_date
            response_data["meta"]["sunset_date"] = version_info.sunset_date
            response_data["meta"]["warning"] = f"API version {version} is deprecated"
        
        response = JSONResponse(content=response_data)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        if version_info:
            response.headers["X-API-Version-Info"] = version_info.version
            if version_info.deprecated:
                response.headers["X-API-Deprecated"] = "true"
                response.headers["X-API-Sunset-Date"] = version_info.sunset_date or ""
        
        return response


# Global version manager instance
version_manager = APIVersionManager()


def version_middleware(request: Request, call_next):
    """Middleware to handle API versioning."""
    async def middleware():
        # Extract and validate version
        requested_version = version_manager.get_requested_version(request)
        request.state.api_version = requested_version
        
        # Check if version is deprecated
        version_info = version_manager.get_version_info(requested_version)
        if version_info.deprecated:
            # Log deprecation warning
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Deprecated API version {requested_version} accessed by "
                f"{request.client.host if request.client else 'unknown'}"
            )
        
        response = await call_next(request)
        return response
    
    return middleware()


# Version-specific response models
class V1Response(BaseModel):
    """Standard v1 API response format."""
    data: Any
    status: str = "success"
    timestamp: str


class V2Response(BaseModel):
    """Enhanced v2 API response format."""
    data: Any
    meta: Dict[str, Any]
    status: str = "success"
    timestamp: str
    pagination: Optional[Dict[str, Any]] = None


def create_versioned_endpoint(versions: list):
    """Decorator to create endpoints that support multiple versions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if request:
                version = getattr(request.state, 'api_version', 'v1')
                kwargs['api_version'] = version
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage for version-specific implementations
class VersionedVerificationResponse:
    """Version-specific verification response formats."""
    
    @staticmethod
    def format_v1(verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format verification result for v1 API."""
        return {
            "task_id": verification_result.get("task_id"),
            "status": verification_result.get("status"),
            "result": verification_result.get("result", {}),
            "created_at": verification_result.get("created_at"),
            "completed_at": verification_result.get("completed_at")
        }
    
    @staticmethod
    def format_v2(verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format verification result for v2 API with enhanced fields."""
        v1_result = VersionedVerificationResponse.format_v1(verification_result)
        
        # Add v2 enhancements
        v1_result.update({
            "verification_chain": verification_result.get("chain_id"),
            "confidence_score": verification_result.get("confidence", 0.0),
            "processing_time_ms": verification_result.get("processing_time", 0),
            "metadata": verification_result.get("metadata", {}),
            "annotations": verification_result.get("annotations", []),
            "quality_metrics": verification_result.get("quality_metrics", {})
        })
        
        return v1_result
    
    @classmethod
    def format_for_version(cls, verification_result: Dict[str, Any], version: str) -> Dict[str, Any]:
        """Format verification result based on API version."""
        if version == "v2":
            return cls.format_v2(verification_result)
        else:
            return cls.format_v1(verification_result) 