"""
Authentication API routes for the Veritas Logos system.

This module implements all authentication-related endpoints including
user registration, login, token management, profile updates, and
administrative user management functions.
"""

from datetime import timedelta
from typing import List, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..auth import (
    User, UserRole, UserStatus, UserCreate, UserResponse, UserUpdate,
    PasswordChange, LoginRequest, Token, TokenData,
    get_db, get_current_user, get_current_admin_user,
    authenticate_user, create_user, create_access_token, create_refresh_token,
    get_password_hash, verify_password, verify_token,
    get_user_by_id, get_user_by_email, get_user_by_username,
    security, ACCESS_TOKEN_EXPIRE_MINUTES
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a new user with the provided information and returns
    the user details without sensitive information.
    """
    try:
        user = create_user(db, user_data)
        return UserResponse.from_orm(user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access and refresh tokens.
    
    Accepts username/email and password, validates credentials,
    and returns JWT tokens for API access.
    """
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value
        },
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token = create_refresh_token(user.id, login_data.device_info)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_orm(user)
    )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_token: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Refresh access token using a valid refresh token.
    
    Validates the refresh token and issues a new access token
    if the refresh token is valid and not expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify refresh token
    token_data = verify_token(refresh_token)
    if not token_data or token_data.sub is None:
        raise credentials_exception
    
    # Get user
    user = get_user_by_id(db, token_data.sub)
    if not user:
        raise credentials_exception
    
    if user.status != UserStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user.status.value}"
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,  # Keep the same refresh token
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.from_orm(user)
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's profile information.
    
    Returns the authenticated user's profile data including
    role, status, and subscription information.
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's profile information.
    
    Allows users to update their profile data such as
    full name and email address.
    """
    try:
        # Check if email is being changed and is already taken
        if user_update.email and user_update.email != current_user.email:
            existing_user = get_user_by_email(db, user_update.email)
            if existing_user and existing_user.id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            current_user.email = user_update.email
        
        if user_update.full_name is not None:
            current_user.full_name = user_update.full_name
        
        db.commit()
        db.refresh(current_user)
        
        return UserResponse.from_orm(current_user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user's password.
    
    Validates current password and updates to new password
    if validation passes.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    try:
        # Update password
        current_user.hashed_password = get_password_hash(password_data.new_password)
        db.commit()
        
        return {"message": "Password updated successfully"}
    except Exception as e:
        logger.error(f"Password change error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: User = Depends(get_current_user)
):
    """
    Logout user (client-side token invalidation).
    
    While JWT tokens can't be server-side invalidated without
    a blacklist, this endpoint confirms successful logout
    and the client should discard tokens.
    """
    # In a production system, you might want to add token to a blacklist
    # For now, we just confirm the logout
    return {"message": "Successfully logged out"}


# Admin endpoints
@router.get("/admin/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[UserStatus] = None,
    role_filter: Optional[UserRole] = None,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    List all users (admin only).
    
    Returns a paginated list of users with optional filtering
    by status and role.
    """
    query = db.query(User)
    
    if status_filter:
        query = query.filter(User.status == status_filter)
    
    if role_filter:
        query = query.filter(User.role == role_filter)
    
    users = query.offset(skip).limit(limit).all()
    return [UserResponse.from_orm(user) for user in users]


@router.get("/admin/users/{user_id}", response_model=UserResponse)
async def get_user_by_admin(
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get user details by ID (admin only).
    
    Returns detailed information about a specific user.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)


@router.put("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    new_status: UserStatus,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update user status (admin only).
    
    Allows admins to activate, suspend, or deactivate user accounts.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from suspending themselves
    if user.id == admin_user.id and new_status in [UserStatus.SUSPENDED, UserStatus.INACTIVE]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend or deactivate your own account"
        )
    
    try:
        user.status = new_status
        db.commit()
        
        return {"message": f"User status updated to {new_status.value}"}
    except Exception as e:
        logger.error(f"User status update error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )


@router.put("/admin/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    new_role: UserRole,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update user role (admin only).
    
    Allows admins to change user roles and permissions.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from demoting themselves if they're the last admin
    if user.id == admin_user.id and new_role != UserRole.ADMIN:
        admin_count = db.query(User).filter(User.role == UserRole.ADMIN).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the last admin user"
            )
    
    try:
        user.role = new_role
        db.commit()
        
        return {"message": f"User role updated to {new_role.value}"}
    except Exception as e:
        logger.error(f"User role update error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )


@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account (admin only).
    
    Permanently removes a user account. This action cannot be undone.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent admin from deleting themselves if they're the last admin
    if user.id == admin_user.id:
        if user.role == UserRole.ADMIN:
            admin_count = db.query(User).filter(User.role == UserRole.ADMIN).count()
            if admin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last admin user"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
    
    try:
        db.delete(user)
        db.commit()
        
        return {"message": "User account deleted successfully"}
    except Exception as e:
        logger.error(f"User deletion error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        ) 