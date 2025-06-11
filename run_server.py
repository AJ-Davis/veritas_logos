#!/usr/bin/env python3
"""
Simple startup script for the Veritas Logos API server.
"""

import os
import sys
import uvicorn

# Clean up any problematic environment variables FIRST
problem_vars = [
    "MAX_TOKENS", "TEMPERATURE", "DEFAULT_SUBTASKS", 
    "MODEL", "PERPLEXITY_MODEL", "DEFAULT_PRIORITY",
    "PROJECT_NAME", "PERPLEXITY_API_KEY", "LOG_LEVEL"
]

for var in problem_vars:
    if var in os.environ and "#" in os.environ[var]:
        # Clean the value by removing comments
        clean_value = os.environ[var].split("#")[0].strip()
        os.environ[var] = clean_value
        print(f"Cleaned {var}: '{clean_value}'")

# Set up minimal environment configuration
os.environ.setdefault("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")
os.environ.setdefault("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
os.environ.setdefault("DATABASE_URL", "sqlite:///./veritas_logos.db")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "info")

# Set up mock Stripe keys for testing (prevents billing errors)
os.environ.setdefault("STRIPE_SECRET_KEY", "sk_test_fake_key_for_testing")
os.environ.setdefault("STRIPE_PUBLISHABLE_KEY", "pk_test_fake_key_for_testing")
os.environ.setdefault("STRIPE_WEBHOOK_SECRET", "whsec_fake_webhook_secret_for_testing")

print("Environment variables configured for development/testing")

def main():
    """Start the API server."""
    try:
        print("Starting Veritas Logos API server...")
        print("Environment: development")
        print("Server will be available at: http://localhost:8000")
        print("API docs will be available at: http://localhost:8000/docs")
        
        # Start the server using import string for reload support
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Failed to import application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 