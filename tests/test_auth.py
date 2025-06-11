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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.auth import (
    Base, User, RefreshToken, UserRole, UserStatus,
    UserCreate, UserResponse, UserUpdate, PasswordChange, LoginRequest, Token,
    verify_password, get_password_hash, create_access_token, create_refresh_token,
    verify_token, create_user, authenticate_user, get_user_by_email,
    get_user_by_username, get_user_by_id, create_database_tables, init_default_admin,
    SessionLocal, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
)
from src.api.routes.auth_routes import router
from src.api.main import app


# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def test_db():
    """Create test database session."""
    Base.metadata.create_all(bind=test_engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def test_client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "TestPass123!",
        "full_name": "Test User"
    }


@pytest.fixture
def admin_user_data():
    """Sample admin user data for testing."""
    return {
        "email": "admin@example.com", 
        "username": "adminuser",
        "password": "AdminPass123!",
        "full_name": "Admin User"
    }


@pytest.fixture
def test_user(test_db, sample_user_data):
    """Create test user in database."""
    user_create = UserCreate(**sample_user_data)
    user = create_user(test_db, user_create)
    return user


@pytest.fixture
def test_admin_user(test_db, admin_user_data):
    """Create test admin user in database."""
    user_create = UserCreate(**admin_user_data)
    user = create_user(test_db, user_create)
    user.role = UserRole.ADMIN
    test_db.commit()
    test_db.refresh(user)
    return user


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
        data = {"sub": "user123", "username": "testuser"}
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
        data = {"sub": "user123", "username": "testuser"}
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
    
    def test_password_change_validation(self):
        """Test PasswordChange model validation."""
        # Valid password change
        password_change = PasswordChange(
            current_password="OldPass123!",
            new_password="NewPass456!"
        )
        assert password_change.current_password == "OldPass123!"
        assert password_change.new_password == "NewPass456!"
        
        # Invalid new password
        with pytest.raises(ValueError):
            PasswordChange(
                current_password="OldPass123!",
                new_password="weak"
            )


class TestUserCRUD:
    """Test user CRUD operations."""
    
    def test_create_user_success(self, test_db, sample_user_data):
        """Test successful user creation."""
        user_create = UserCreate(**sample_user_data)
        user = create_user(test_db, user_create)
        
        assert user.email == sample_user_data["email"]
        assert user.username == sample_user_data["username"]
        assert user.full_name == sample_user_data["full_name"]
        assert user.role == UserRole.USER
        assert user.status == UserStatus.ACTIVE
        assert verify_password(sample_user_data["password"], user.hashed_password)
    
    def test_create_user_duplicate_email(self, test_db, sample_user_data):
        """Test user creation with duplicate email."""
        user_create = UserCreate(**sample_user_data)
        create_user(test_db, user_create)
        
        # Try to create another user with same email
        with pytest.raises(Exception):  # Should raise HTTPException
            create_user(test_db, user_create)
    
    def test_create_user_duplicate_username(self, test_db, sample_user_data):
        """Test user creation with duplicate username."""
        user_create = UserCreate(**sample_user_data)
        create_user(test_db, user_create)
        
        # Try to create another user with same username
        duplicate_data = sample_user_data.copy()
        duplicate_data["email"] = "different@example.com"
        duplicate_create = UserCreate(**duplicate_data)
        
        with pytest.raises(Exception):  # Should raise HTTPException
            create_user(test_db, duplicate_create)
    
    def test_get_user_by_email(self, test_db, test_user):
        """Test getting user by email."""
        found_user = get_user_by_email(test_db, test_user.email)
        assert found_user is not None
        assert found_user.id == test_user.id
        
        # Non-existent email
        not_found = get_user_by_email(test_db, "nonexistent@example.com")
        assert not_found is None
    
    def test_get_user_by_username(self, test_db, test_user):
        """Test getting user by username."""
        found_user = get_user_by_username(test_db, test_user.username)
        assert found_user is not None
        assert found_user.id == test_user.id
        
        # Non-existent username
        not_found = get_user_by_username(test_db, "nonexistent")
        assert not_found is None
    
    def test_get_user_by_id(self, test_db, test_user):
        """Test getting user by ID."""
        found_user = get_user_by_id(test_db, test_user.id)
        assert found_user is not None
        assert found_user.email == test_user.email
        
        # Non-existent ID
        not_found = get_user_by_id(test_db, str(uuid.uuid4()))
        assert not_found is None
    
    def test_authenticate_user_success(self, test_db, test_user, sample_user_data):
        """Test successful user authentication."""
        authenticated_user = authenticate_user(
            test_db, 
            test_user.username, 
            sample_user_data["password"]
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == test_user.id
    
    def test_authenticate_user_by_email(self, test_db, test_user, sample_user_data):
        """Test user authentication by email."""
        authenticated_user = authenticate_user(
            test_db,
            test_user.email,
            sample_user_data["password"]
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == test_user.id
    
    def test_authenticate_user_wrong_password(self, test_db, test_user):
        """Test authentication with wrong password."""
        authenticated_user = authenticate_user(
            test_db,
            test_user.username,
            "WrongPassword"
        )
        
        assert authenticated_user is None
    
    def test_authenticate_user_nonexistent(self, test_db):
        """Test authentication with non-existent user."""
        authenticated_user = authenticate_user(
            test_db,
            "nonexistent",
            "password"
        )
        
        assert authenticated_user is None
    
    def test_authenticate_user_inactive(self, test_db, test_user, sample_user_data):
        """Test authentication with inactive user."""
        test_user.status = UserStatus.INACTIVE
        test_db.commit()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            authenticate_user(
                test_db,
                test_user.username,
                sample_user_data["password"]
            )


class TestDatabaseFunctions:
    """Test database initialization functions."""
    
    @patch('src.api.auth.SessionLocal')
    def test_init_default_admin_no_existing(self, mock_session):
        """Test default admin creation when none exists."""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        init_default_admin()
        
        # Verify admin user was created
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @patch('src.api.auth.SessionLocal')
    def test_init_default_admin_existing(self, mock_session):
        """Test default admin creation when admin already exists."""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = Mock()
        
        init_default_admin()
        
        # Verify no admin user was created
        mock_db.add.assert_not_called()
        mock_db.commit.assert_not_called()


class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""
    
    @patch('src.api.routes.auth_routes.get_db')
    def test_register_user_success(self, mock_get_db, test_client, sample_user_data):
        """Test successful user registration."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock user creation
        mock_user = Mock()
        mock_user.id = str(uuid.uuid4())
        mock_user.email = sample_user_data["email"]
        mock_user.username = sample_user_data["username"]
        mock_user.full_name = sample_user_data["full_name"]
        mock_user.role = UserRole.USER
        mock_user.status = UserStatus.ACTIVE
        mock_user.created_at = datetime.utcnow()
        mock_user.last_login = None
        mock_user.is_verified = False
        mock_user.subscription_tier = "free"
        mock_user.subscription_status = "inactive"
        
        with patch('src.api.routes.auth_routes.create_user', return_value=mock_user):
            response = test_client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == sample_user_data["email"]
        assert data["username"] == sample_user_data["username"]
    
    @patch('src.api.routes.auth_routes.get_db')
    def test_register_user_duplicate_email(self, mock_get_db, test_client, sample_user_data):
        """Test user registration with duplicate email."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        from fastapi import HTTPException
        with patch('src.api.routes.auth_routes.create_user', 
                   side_effect=HTTPException(status_code=400, detail="Email already registered")):
            response = test_client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_register_user_invalid_data(self, test_client):
        """Test user registration with invalid data."""
        invalid_data = {
            "email": "invalid-email",
            "username": "test user!",
            "password": "weak"
        }
        
        response = test_client.post("/api/v1/auth/register", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @patch('src.api.routes.auth_routes.get_db')
    @patch('src.api.routes.auth_routes.authenticate_user')
    @patch('src.api.routes.auth_routes.create_access_token')
    @patch('src.api.routes.auth_routes.create_refresh_token')
    def test_login_success(self, mock_create_refresh, mock_create_access, 
                          mock_authenticate, mock_get_db, test_client):
        """Test successful login."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        # Mock user
        mock_user = Mock()
        mock_user.id = str(uuid.uuid4())
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.role.value = "user"
        mock_authenticate.return_value = mock_user
        
        # Mock token creation
        mock_create_access.return_value = "mock_access_token"
        mock_create_refresh.return_value = "mock_refresh_token"
        
        login_data = {
            "username": "testuser",
            "password": "TestPass123!"
        }
        
        with patch('src.api.routes.auth_routes.UserResponse.from_orm') as mock_from_orm:
            mock_from_orm.return_value = {"id": mock_user.id, "username": "testuser"}
            response = test_client.post("/api/v1/auth/login", json=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @patch('src.api.routes.auth_routes.get_db')
    @patch('src.api.routes.auth_routes.authenticate_user')
    def test_login_invalid_credentials(self, mock_authenticate, mock_get_db, test_client):
        """Test login with invalid credentials."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_authenticate.return_value = None
        
        login_data = {
            "username": "testuser",
            "password": "WrongPassword"
        }
        
        response = test_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('src.api.routes.auth_routes.get_current_user')
    def test_get_current_user_profile(self, mock_get_current_user, test_client):
        """Test getting current user profile."""
        # Mock current user
        mock_user = Mock()
        mock_user.id = str(uuid.uuid4())
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_get_current_user.return_value = mock_user
        
        headers = {"Authorization": "Bearer mock_token"}
        
        with patch('src.api.routes.auth_routes.UserResponse.from_orm') as mock_from_orm:
            mock_from_orm.return_value = {"id": mock_user.id, "username": "testuser"}
            response = test_client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
    
    @patch('src.api.routes.auth_routes.get_current_user')
    @patch('src.api.routes.auth_routes.get_db')
    def test_change_password_success(self, mock_get_db, mock_get_current_user, test_client):
        """Test successful password change."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        mock_user = Mock()
        mock_user.hashed_password = get_password_hash("OldPass123!")
        mock_get_current_user.return_value = mock_user
        
        password_data = {
            "current_password": "OldPass123!",
            "new_password": "NewPass456!"
        }
        
        headers = {"Authorization": "Bearer mock_token"}
        response = test_client.post("/api/v1/auth/change-password", 
                                  json=password_data, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Password updated successfully"
    
    @patch('src.api.routes.auth_routes.get_current_user')
    def test_change_password_wrong_current(self, mock_get_current_user, test_client):
        """Test password change with wrong current password."""
        mock_user = Mock()
        mock_user.hashed_password = get_password_hash("OldPass123!")
        mock_get_current_user.return_value = mock_user
        
        password_data = {
            "current_password": "WrongPassword",
            "new_password": "NewPass456!"
        }
        
        headers = {"Authorization": "Bearer mock_token"}
        response = test_client.post("/api/v1/auth/change-password",
                                  json=password_data, headers=headers)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestAdminEndpoints:
    """Test admin-only endpoints."""
    
    @patch('src.api.routes.auth_routes.get_current_admin_user')
    @patch('src.api.routes.auth_routes.get_db')
    def test_list_users_admin(self, mock_get_db, mock_get_admin, test_client):
        """Test listing users as admin."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        mock_admin = Mock()
        mock_admin.role = UserRole.ADMIN
        mock_get_admin.return_value = mock_admin
        
        # Mock query result
        mock_users = [Mock(), Mock()]
        mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_users
        
        headers = {"Authorization": "Bearer admin_token"}
        
        with patch('src.api.routes.auth_routes.UserResponse.from_orm') as mock_from_orm:
            mock_from_orm.return_value = {"id": "user_id", "username": "user"}
            response = test_client.get("/api/v1/auth/admin/users", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
    
    @patch('src.api.routes.auth_routes.get_current_admin_user')
    @patch('src.api.routes.auth_routes.get_db')
    @patch('src.api.routes.auth_routes.get_user_by_id')
    def test_update_user_status_admin(self, mock_get_user, mock_get_db, 
                                     mock_get_admin, test_client):
        """Test updating user status as admin."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        
        mock_admin = Mock()
        mock_admin.id = "admin_id"
        mock_admin.role = UserRole.ADMIN
        mock_get_admin.return_value = mock_admin
        
        mock_user = Mock()
        mock_user.id = "user_id"
        mock_user.status = UserStatus.ACTIVE
        mock_get_user.return_value = mock_user
        
        headers = {"Authorization": "Bearer admin_token"}
        response = test_client.put("/api/v1/auth/admin/users/user_id/status",
                                 params={"new_status": "suspended"}, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK 