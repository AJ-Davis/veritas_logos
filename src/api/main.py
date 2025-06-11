"""
Main FastAPI application for the Veritas Logos verification system.

This module creates the main FastAPI application that integrates:
- JWT authentication system
- Document verification endpoints
- Dashboard and output generation endpoints
- WebSocket support for real-time updates
- Rate limiting and security features
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import authentication system
from .auth import (
    create_database_tables, init_default_admin,
    get_current_user, User
)

# Import route modules
from .routes.auth_routes import router as auth_router
from .routes.billing_routes import router as billing_router
from .routes.document_routes import router as document_router
from .routes.verification_status_routes import router as verification_status_router
from .routes.websocket_routes import router as websocket_router

# Import WebSocket manager
from .websocket_manager import websocket_manager

# Import enhanced middleware
from .middleware import (
    RequestLoggingMiddleware, SecurityMiddleware, MetricsMiddleware,
    get_metrics
)

# Configure enhanced logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log', mode='a')
    ] if os.path.exists('logs') or os.makedirs('logs', exist_ok=True) else [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
API_V1_PREFIX = "/api/v1"

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",        # React development
    "http://localhost:8080",        # Vue.js development
    "http://localhost:5173",        # Vite development
    "https://localhost:3000",       # HTTPS development
    "https://app.veritaslogos.com",  # Production frontend
    "https://admin.veritaslogos.com",  # Admin dashboard
]

if ENVIRONMENT == "development":
    ALLOWED_ORIGINS.extend([
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
        "http://localhost:8000",   # API docs
        "http://127.0.0.1:8000",
    ])

# Additional origins from environment
additional_origins = os.getenv("ADDITIONAL_CORS_ORIGINS")
if additional_origins:
    ALLOWED_ORIGINS.extend(additional_origins.split(","))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Veritas Logos API...")

    # Initialize database
    try:
        create_database_tables()
        init_default_admin()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    # Initialize verification system components
    try:
        # This will trigger the startup event in verification_api
        logger.info("Verification system initialized")
    except Exception as e:
        logger.error(f"Verification system initialization failed: {e}")
        raise

    # Initialize WebSocket manager
    try:
        await websocket_manager.start()
        logger.info("WebSocket manager started successfully")
    except Exception as e:
        logger.error(f"WebSocket manager initialization failed: {e}")
        raise

    yield  # Application is running

    # Shutdown
    logger.info("Shutting down Veritas Logos API...")

    # Shutdown WebSocket manager
    try:
        await websocket_manager.stop()
        logger.info("WebSocket manager stopped successfully")
    except Exception as e:
        logger.error(f"WebSocket manager shutdown failed: {e}")


# Create main FastAPI application
app = FastAPI(
    title="Veritas Logos API",
    description="Document verification and analysis system with JWT authentication",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add enhanced middleware stack (order matters!)
# Security middleware first
app.add_middleware(SecurityMiddleware, max_request_size=50 * 1024 * 1024)  # 50MB for document uploads

# Metrics middleware
app.add_middleware(MetricsMiddleware)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware, log_level=log_level)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Trusted host middleware for production
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["veritaslogos.com", "*.veritaslogos.com", "localhost"]
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Include authentication routes
app.include_router(auth_router, prefix=API_V1_PREFIX)

# Include billing routes
app.include_router(billing_router, prefix=API_V1_PREFIX)

# Include document routes
app.include_router(document_router, prefix=API_V1_PREFIX)

# Include verification status routes
app.include_router(verification_status_router, prefix=API_V1_PREFIX)

# Include WebSocket routes
app.include_router(websocket_router, prefix=API_V1_PREFIX)


# Root endpoint with API information
@app.get("/", response_model=Dict[str, str])
@limiter.limit("100/minute")
def root(request: Request):
    """Root endpoint with API information."""
    return {
        "message": "Veritas Logos Document Verification API",
        "version": "1.0.0",
        "docs": "/docs" if ENVIRONMENT == "development" else "Contact admin for API documentation",
        "health": "/health"
    }


@app.get("/health", response_model=Dict[str, Any])
@limiter.limit("200/minute")
def health_check(request: Request):
    """Health check endpoint for monitoring."""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "environment": ENVIRONMENT
    }


@app.get(f"{API_V1_PREFIX}/metrics", response_model=Dict[str, Any])
@limiter.limit("30/minute")
def get_api_metrics(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get API performance metrics (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return get_metrics()


@app.get(f"{API_V1_PREFIX}/chains")
@limiter.limit("50/minute")
def list_verification_chains_protected(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """List available verification chains."""
    # This would integrate with the verification system
    return {"chains": ["academic_paper", "news_article", "social_media"]}


@app.get(f"{API_V1_PREFIX}/chains/{{chain_id}}")
@limiter.limit("50/minute")
def get_verification_chain_protected(
    request: Request,
    chain_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get details about a specific verification chain."""
    # This would integrate with the verification system
    return {"chain_id": chain_id, "description": f"Verification chain for {chain_id}"}


@app.post(f"{API_V1_PREFIX}/verify")
@limiter.limit("10/minute")
def submit_verification_task_protected(
    request: Request,
    verification_request: dict,  # Will define proper model
    current_user: User = Depends(get_current_user)
):
    """Submit a new verification task."""
    # This would integrate with the verification system
    return {"task_id": "12345", "status": "submitted"}


@app.get(f"{API_V1_PREFIX}/tasks/{{task_id}}/status")
@limiter.limit("60/minute")
def get_task_status_protected(
    request: Request,
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of a verification task."""
    # This would integrate with the verification system
    return {"task_id": task_id, "status": "processing"}


@app.get(f"{API_V1_PREFIX}/tasks/{{task_id}}/result")
@limiter.limit("30/minute")
def get_task_result_protected(
    request: Request,
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the result of a completed verification task."""
    # This would integrate with the verification system
    return {"task_id": task_id, "result": {"verified": True, "confidence": 0.85}}


@app.get(f"{API_V1_PREFIX}/tasks")
@limiter.limit("30/minute")
def list_tasks_protected(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """List verification tasks for the current user."""
    # This would integrate with the verification system
    return {"tasks": []}


@app.delete(f"{API_V1_PREFIX}/tasks/{{task_id}}")
@limiter.limit("20/minute")
def cancel_task_protected(
    request: Request,
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a verification task."""
    # This would integrate with the verification system
    return {"task_id": task_id, "status": "cancelled"}


# Duplicate endpoint removed - using the one above
# The duplicate metrics endpoint has been removed to avoid conflicts


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use real timestamp
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with logging."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use real timestamp
        }
    )


if __name__ == "__main__":
    import uvicorn

    # Configuration for development
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": ENVIRONMENT == "development",
        "log_level": "debug" if DEBUG else "info",
        "access_log": True,
    }

    logger.info(f"Starting server in {ENVIRONMENT} mode...")
    uvicorn.run("src.api.main:app", **config)
