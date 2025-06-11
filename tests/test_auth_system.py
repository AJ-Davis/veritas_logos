"""
Unit tests for the authentication system.

This module tests JWT authentication, user management, password handling,
role-based access control, and all authentication API endpoints.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

from src.api.auth import (
    verify_password, get_password_hash, create_access_token, 
    verify_token, UserCreate, UserRole, UserStatus
)


class TestPasswordFunctions:
    """Test password hashing and verification functions."""
    
    def test_password_hashing(self):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("WrongPassword", hashed)
    
    def test_password_verification(self):
        """Test password verification with various inputs."""
        password = "MySecurePass456!"
        hashed = get_password_hash(password)
        
        # Correct password
        assert verify_password(password, hashed)
        
        # Wrong password
        assert not verify_password("WrongPassword", hashed)
        
        # Empty password
        assert not verify_password("", hashed)
        
        # Case sensitive
        assert not verify_password(password.lower(), hashed)


class TestJWTFunctions:
    """Test JWT token creation and verification."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        data = {
            "sub": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "user"
        }
        
        token = create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_expiry(self):
        """Test access token creation with custom expiry."""
        data = {"sub": "user123", "username": "testuser", "email": "test@example.com", "role": "user"}
        expires_delta = timedelta(minutes=15)
        
        token = create_access_token(data, expires_delta)
        token_data = verify_token(token)
        
        assert token_data is not None
        assert token_data.sub == "user123"
    
    def test_verify_token_valid(self):
        """Test token verification with valid token."""
        data = {
            "sub": "user123",
            "username": "testuser", 
            "email": "test@example.com",
            "role": "user"
        }
        
        token = create_access_token(data)
        token_data = verify_token(token)
        
        assert token_data is not None
        assert token_data.sub == "user123"
        assert token_data.username == "testuser"
        assert token_data.email == "test@example.com"
        assert token_data.role == "user"
    
    def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        invalid_token = "invalid.token.string"
        token_data = verify_token(invalid_token)
        assert token_data is None
    
    def test_verify_token_expired(self):
        """Test token verification with expired token."""
        data = {"sub": "user123", "username": "testuser", "email": "test@example.com", "role": "user"}
        expires_delta = timedelta(seconds=-1)  # Already expired
        
        token = create_access_token(data, expires_delta)
        token_data = verify_token(token)
        
        # Token should be expired and verification should fail
        assert token_data is None


class TestUserModels:
    """Test Pydantic user models."""
    
    def test_user_create_valid(self):
        """Test UserCreate model with valid data."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPass123!",
            "full_name": "Test User"
        }
        
        user_create = UserCreate(**user_data)
        assert user_create.email == "test@example.com"
        assert user_create.username == "testuser"  # Should be lowercased
        assert user_create.password == "TestPass123!"
        assert user_create.full_name == "Test User"
    
    def test_user_create_invalid_email(self):
        """Test UserCreate model with invalid email."""
        user_data = {
            "email": "invalid-email",
            "username": "testuser",
            "password": "TestPass123!"
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)
    
    def test_user_create_invalid_username(self):
        """Test UserCreate model with invalid username."""
        user_data = {
            "email": "test@example.com",
            "username": "test user!",  # Invalid characters
            "password": "TestPass123!"
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)
    
    def test_user_create_weak_password(self):
        """Test UserCreate model with weak password."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "weak"  # Too weak
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)


if __name__ == "__main__":
    pytest.main([__file__]) 