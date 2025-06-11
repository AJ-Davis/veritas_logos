"""
Configuration settings for the FastAPI application.

Handles environment-based configuration, security settings, and application parameters.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Basic application settings
    APP_NAME: str = "Veritas Logos API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./veritas_logos.db"
    DATABASE_ECHO: bool = False
    
    # JWT Authentication settings
    JWT_SECRET_KEY: str = "your-secret-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Security settings
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*.veritaslogos.com"]
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:5173",
        "https://app.veritaslogos.com"
    ]
    ADDITIONAL_CORS_ORIGINS: Optional[str] = None
    
    # Rate limiting settings
    RATE_LIMIT_ENABLED: bool = True
    DEFAULT_RATE_LIMIT: str = "100/minute"
    AUTH_RATE_LIMIT: str = "5/minute"
    UPLOAD_RATE_LIMIT: str = "10/minute"
    API_RATE_LIMIT: str = "60/minute"
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 150 * 1024 * 1024  # 150MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".md", ".txt"]
    UPLOAD_DIRECTORY: str = "./uploads"
    
    # Stripe settings
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_PRICE_IDS: Dict[str, str] = {
        "basic": "price_basic",
        "pro": "price_pro", 
        "enterprise": "price_enterprise"
    }
    
    # WebSocket settings
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    WEBSOCKET_CLEANUP_INTERVAL: int = 300
    WEBSOCKET_MAX_CONNECTIONS: int = 1000
    
    # Monitoring and logging settings
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8090
    ENABLE_TRACING: bool = False
    JAEGER_ENDPOINT: Optional[str] = None
    
    # External service settings
    ANTHROPIC_API_KEY: Optional[str] = None
    PERPLEXITY_API_KEY: Optional[str] = None
    
    # LLM settings
    MODEL: Optional[str] = None
    MAX_TOKENS: Optional[int] = None
    TEMPERATURE: Optional[float] = None
    PERPLEXITY_MODEL: Optional[str] = None
    
    # Task Master settings
    DEFAULT_SUBTASKS: Optional[int] = None
    DEFAULT_PRIORITY: Optional[str] = None
    PROJECT_NAME: Optional[str] = None
    
    # Cache settings
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600
    
    # Health check settings
    HEALTH_CHECK_INTERVAL: int = 30
    ENABLE_DEEP_HEALTH_CHECKS: bool = True
    
    # API documentation settings
    ENABLE_DOCS: bool = True
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @field_validator("STRIPE_PRICE_IDS", mode="before")
    @classmethod
    def parse_stripe_price_ids(cls, v):
        """Parse Stripe price IDs from environment."""
        if isinstance(v, str):
            # Format: "basic:price_123,pro:price_456,enterprise:price_789"
            prices = {}
            for pair in v.split(","):
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    prices[key.strip()] = value.strip()
            return prices
        return v
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v):
        """Parse DEBUG from string boolean."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def docs_enabled(self) -> bool:
        """Check if API documentation should be enabled."""
        return self.ENABLE_DOCS and (self.is_development or self.DEBUG)
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get complete list of CORS origins including additional ones."""
        origins = self.CORS_ORIGINS.copy()
        if self.ADDITIONAL_CORS_ORIGINS:
            additional = [origin.strip() for origin in self.ADDITIONAL_CORS_ORIGINS.split(",")]
            origins.extend(additional)
        return origins
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"  # Ignore extra environment variables
    }


class SecuritySettings:
    """Security-related configuration and constants."""
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    }
    
    # Sensitive endpoints that require extra protection
    SENSITIVE_ENDPOINTS = [
        "/api/v1/auth/token",
        "/api/v1/auth/refresh",
        "/api/v1/auth/register",
        "/api/v1/billing/webhooks/stripe",
        "/api/v1/admin/*"
    ]
    
    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = [
        "/",
        "/health",
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/api/v1/auth/token",
        "/api/v1/auth/register"
    ]


class MonitoringSettings:
    """Monitoring and observability configuration."""
    
    # Metrics to track
    METRICS_CONFIG = {
        "request_duration_seconds": {
            "type": "histogram",
            "description": "HTTP request duration in seconds",
            "buckets": [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        },
        "request_count_total": {
            "type": "counter", 
            "description": "Total number of HTTP requests"
        },
        "active_connections": {
            "type": "gauge",
            "description": "Number of active WebSocket connections"
        },
        "verification_tasks_total": {
            "type": "counter",
            "description": "Total number of verification tasks"
        },
        "verification_duration_seconds": {
            "type": "histogram",
            "description": "Verification task duration in seconds"
        }
    }
    
    # Health check configuration
    HEALTH_CHECKS = {
        "database": {"timeout": 5, "critical": True},
        "redis": {"timeout": 3, "critical": False},
        "stripe": {"timeout": 10, "critical": False},
        "anthropic": {"timeout": 15, "critical": False}
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()

# Export commonly used settings
__all__ = [
    "Settings",
    "SecuritySettings", 
    "MonitoringSettings",
    "get_settings",
    "settings"
] 