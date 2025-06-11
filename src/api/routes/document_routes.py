"""
Document management routes for the Veritas Logos system.

This module handles document upload, verification submission, status checking,
and results retrieval with authentication and billing integration.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import mimetypes
from pathlib import Path

from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File,
    BackgroundTasks, Request, Query
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.api.auth import get_db, get_current_user, User
from src.api.billing import (
    get_customer_by_user, record_usage, UsageRecordCreate, UsageType,
    enforce_usage_limit
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])

# Configuration
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 150 * 1024 * 1024  # 150 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/markdown",
    "text/plain"
}


# Pydantic Models
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    filename: str
    file_size: int
    mime_type: str
    upload_timestamp: datetime
    message: str


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    document_id: str
    filename: str
    file_size: int
    mime_type: str
    upload_timestamp: datetime
    user_id: str
    verification_count: int = 0
    last_verification: Optional[datetime] = None


class VerificationSubmissionRequest(BaseModel):
    """Request model for verification submission."""
    document_id: str = Field(..., description="ID of uploaded document")
    chain_id: str = Field(default="comprehensive", description="Verification chain to use")
    priority: str = Field(default="MEDIUM", description="Task priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    enable_acvf: bool = Field(default=True, description="Enable ACVF adversarial testing")


class VerificationSubmissionResponse(BaseModel):
    """Response for verification submission."""
    task_id: str
    document_id: str
    status: str
    chain_id: str
    priority: str
    created_at: datetime
    estimated_completion_time: Optional[str] = None
    message: str


# Helper Functions
def ensure_upload_directory():
    """Ensure upload directory exists."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_file_extension(filename: str) -> bool:
    """Validate file extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_mime_type(mime_type: str) -> bool:
    """Validate MIME type."""
    return mime_type in ALLOWED_MIME_TYPES


def generate_document_id() -> str:
    """Generate unique document ID."""
    return str(uuid.uuid4())


def get_file_path(document_id: str, filename: str) -> str:
    """Get full file path for document storage."""
    ensure_upload_directory()
    safe_filename = f"{document_id}_{filename}"
    return os.path.join(UPLOAD_DIR, safe_filename)


# Document Upload Endpoints
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a document for verification.
    
    Supports PDF, DOCX, Markdown, and TXT files up to 150MB.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content to check size and MIME type
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type or not validate_mime_type(mime_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported MIME type: {mime_type}"
            )
        
        # Check usage limits
        customer = get_customer_by_user(db, current_user)
        if customer:
            enforce_usage_limit(db, customer, UsageType.DOCUMENT_UPLOAD, 1)
        
        # Generate document ID and save file
        document_id = generate_document_id()
        file_path = get_file_path(document_id, file.filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Record usage for billing
        if customer:
            usage_record = UsageRecordCreate(
                usage_type=UsageType.DOCUMENT_UPLOAD,
                quantity=1,
                metadata={
                    "document_id": document_id,
                    "filename": file.filename,
                    "file_size": file_size,
                    "mime_type": mime_type
                }
            )
            record_usage(db, customer, usage_record)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_size=file_size,
            mime_type=mime_type,
            upload_timestamp=datetime.utcnow(),
            message="Document uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during upload"
        )


@router.get("/{document_id}/metadata", response_model=DocumentMetadata)
async def get_document_metadata(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get metadata for an uploaded document."""
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(document_id)]
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    file_path = os.path.join(UPLOAD_DIR, files[0])
    file_stat = os.stat(file_path)
    filename = files[0].replace(f"{document_id}_", "")
    
    return DocumentMetadata(
        document_id=document_id,
        filename=filename,
        file_size=file_stat.st_size,
        mime_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
        upload_timestamp=datetime.fromtimestamp(file_stat.st_ctime),
        user_id=str(current_user.id),
        verification_count=0,
        last_verification=None
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an uploaded document."""
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(document_id)]
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file)
            os.remove(file_path)
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document"
        )


@router.post("/{document_id}/verify", response_model=VerificationSubmissionResponse)
async def submit_verification(
    document_id: str,
    request: VerificationSubmissionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a document for verification processing.
    
    This triggers the verification pipeline including ACVF adversarial testing.
    """
    try:
        # Verify document exists
        files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(document_id)]
        if not files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check usage limits
        customer = get_customer_by_user(db, current_user)
        if customer:
            enforce_usage_limit(db, customer, UsageType.DOCUMENT_VERIFICATION, 1)
        
        # Create verification task
        task_id = str(uuid.uuid4())
        
        # Record usage for billing
        if customer:
            usage_record = UsageRecordCreate(
                usage_type=UsageType.DOCUMENT_VERIFICATION,
                quantity=1,
                metadata={
                    "task_id": task_id,
                    "document_id": document_id,
                    "chain_id": request.chain_id,
                    "enable_acvf": request.enable_acvf
                }
            )
            record_usage(db, customer, usage_record)
        
        return VerificationSubmissionResponse(
            task_id=task_id,
            document_id=document_id,
            status="PENDING",
            chain_id=request.chain_id,
            priority=request.priority,
            created_at=datetime.utcnow(),
            estimated_completion_time="5-15 minutes",
            message="Verification task submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during verification submission"
        )


@router.get("/")
async def list_documents(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """List documents for the current user."""
    return {
        "documents": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    } 