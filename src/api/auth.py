"""
Authentication system for the Veritas Logos API.

This module implements JWT-based authentication with user management,
role-based access control, and secure password handling.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Enum as SQLEnum, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./veritas_logos.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserRole(str, Enum):
    """User roles for access control."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # API usage tracking
    api_calls_today = Column(Integer, default=0, nullable=False)
    last_api_call_date = Column(DateTime, nullable=True)
    
    # Billing information
    stripe_customer_id = Column(String, nullable=True)
    subscription_tier = Column(String, default="free", nullable=False)
    subscription_status = Column(String, default="inactive", nullable=False)


class RefreshToken(Base):
    """Refresh token model for secure token renewal."""
    __tablename__ = "refresh_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    token_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    device_info = Column(Text, nullable=True)


# Pydantic models for API
class UserCreate(BaseModel):
    """Model for user creation."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must contain only letters and numbers')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserResponse(BaseModel):
    """Model for user information in responses."""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime]
    is_verified: bool
    subscription_tier: str
    subscription_status: str
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Model for user updates."""
    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    
    class Config:
        from_attributes = True


class PasswordChange(BaseModel):
    """Model for password change requests."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token payload data."""
    sub: str  # user_id
    username: str
    email: str
    role: str
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for tracking


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str
    device_info: Optional[str] = None


# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: str, device_info: Optional[str] = None) -> str:
    """Create refresh token and store in database."""
    db = SessionLocal()
    try:
        # Generate refresh token
        token_data = {
            "sub": user_id,
            "type": "refresh",
            "jti": str(uuid.uuid4())
        }
        
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        token_data["exp"] = expire
        
        refresh_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        token_hash = pwd_context.hash(refresh_token)
        
        # Store in database
        db_token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expire,
            device_info=device_info
        )
        db.add(db_token)
        db.commit()
        
        return refresh_token
    finally:
        db.close()


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        email: str = payload.get("email")
        role: str = payload.get("role")
        
        if user_id is None or username is None:
            return None
            
        token_data = TokenData(
            sub=user_id,
            username=username,
            email=email,
            role=role,
            exp=datetime.fromtimestamp(payload.get("exp")),
            iat=datetime.fromtimestamp(payload.get("iat")),
            jti=payload.get("jti")
        )
        return token_data
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, user_create: UserCreate) -> User:
    """Create new user."""
    # Check if user already exists
    if get_user_by_email(db, user_create.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if get_user_by_username(db, user_create.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    hashed_password = get_password_hash(user_create.password)
    db_user = User(
        email=user_create.email,
        username=user_create.username.lower(),
        full_name=user_create.full_name,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user credentials."""
    user = get_user_by_username(db, username.lower())
    if not user:
        user = get_user_by_email(db, username.lower())
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.value}"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    
    user = get_user_by_id(db, token_data.sub)
    if user is None:
        raise credentials_exception
    
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.value}"
        )
    
    return user


def require_role(required_roles: List[UserRole]):
    """Decorator to require specific user roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


def get_current_admin_user(current_user: User = Depends(require_role([UserRole.ADMIN]))) -> User:
    """Get current admin user."""
    return current_user


def create_database_tables():
    """Create database tables."""
    Base.metadata.create_all(bind=engine)


def init_default_admin():
    """Initialize default admin user if none exists."""
    db = SessionLocal()
    try:
        admin_exists = db.query(User).filter(User.role == UserRole.ADMIN).first()
        if not admin_exists:
            # Create default admin
            admin_user = User(
                email="admin@veritaslogos.com",
                username="admin",
                full_name="System Administrator",
                hashed_password=get_password_hash("AdminPass123!"),
                role=UserRole.ADMIN,
                status=UserStatus.ACTIVE,
                is_verified=True
            )
            db.add(admin_user)
            db.commit()
            logger.info("Default admin user created: admin@veritaslogos.com")
    finally:
        db.close() 