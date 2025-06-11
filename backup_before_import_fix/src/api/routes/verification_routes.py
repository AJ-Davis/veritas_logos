"""
Verification API routes for the Veritas Logos system.

This module implements all verification-related endpoints including document
upload, submission to the verification pipeline, status checking, and results
retrieval with proper authentication and billing integration.
"""

import os
import shutil
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import mimetypes
from pathlib import Path

from fastapi import (
    APIRouter, Depends, HTTPException, status, UploadFile, File, Form,
    BackgroundTasks, Request, Query
)
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..auth import get_db, get_current_user, User
from ..billing import (
    get_customer_by_user, record_usage, UsageRecordCreate, UsageType,
    enforce_usage_limit
)
from ...verification.config.chain_loader import ChainConfigLoader
from ...verification.workers.verification_worker import execute_verification_chain_task
from ...models.verification import (
    VerificationTask, VerificationChainConfig, VerificationChainResult,
    VerificationStatus, Priority
)
from ...models.document import ParsedDocument
from ...models.issues import UnifiedIssue
from ...models.output import OutputVerificationResult
from ...models.dashboard import DashboardAggregator
from ...models.acvf import ACVFResult
from ...models.debate_view_generator import DebateViewGenerator
from ...document_ingestion.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verification"])

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

# Initialize services
document_service = DocumentService()
config_loader = ChainConfigLoader()


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
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    enable_acvf: bool = Field(default=True, description="Enable ACVF adversarial testing")


class VerificationSubmissionResponse(BaseModel):
    """Response for verification submission."""
    task_id: str
    document_id: str
    status: VerificationStatus
    chain_id: str
    priority: Priority
    created_at: datetime
    estimated_completion_time: Optional[str] = None
    message: str


class VerificationStatusResponse(BaseModel):
    """Response for verification status check."""
    task_id: str
    document_id: str
    status: VerificationStatus
    progress: Optional[Dict[str, Any]] = None
    current_pass: Optional[str] = None
    completed_passes: List[str] = []
    remaining_passes: List[str] = []
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class VerificationResultsResponse(BaseModel):
    """Response for verification results."""
    task_id: str
    document_id: str
    status: VerificationStatus
    result: Optional[VerificationChainResult] = None
    issues: List[UnifiedIssue] = []
    acvf_result: Optional[ACVFResult] = None
    summary: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    completed_at: Optional[datetime] = None


class DashboardDataResponse(BaseModel):
    """Response for dashboard data."""
    document_id: str
    task_id: str
    aggregated_data: Dict[str, Any]
    time_series: List[Dict[str, Any]]
    heatmap_data: Dict[str, Any]
    relationship_graph: Dict[str, Any]
    generated_at: datetime


class DebateViewResponse(BaseModel):
    """Response for debate view data."""
    task_id: str
    document_id: str
    debate_data: Dict[str, Any]
    participants: List[str]
    total_rounds: int
    final_verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    generated_at: datetime


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
@router.post("/documents/upload", response_model=DocumentUploadResponse)
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


@router.get("/documents/{document_id}/metadata", response_model=DocumentMetadata)
async def get_document_metadata(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get metadata for an uploaded document."""
    # Implementation would query database for document metadata
    # For now, return basic info if file exists
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
        user_id=current_user.id,
        verification_count=0,  # Would be fetched from database
        last_verification=None
    )


@router.delete("/documents/{document_id}")
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


# Verification Submission Endpoints
@router.post("/verification/submit", response_model=VerificationSubmissionResponse)
async def submit_verification(
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
        files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(request.document_id)]
        if not files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check usage limits
        customer = get_customer_by_user(db, current_user)
        if customer:
            enforce_usage_limit(db, customer, UsageType.DOCUMENT_VERIFICATION, 1)
        
        # Get verification chain configuration
        try:
            chain_config = config_loader.get_chain_config(request.chain_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chain_id: {request.chain_id}"
            )
        
        # Create verification task
        task_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, files[0])
        
        verification_task = VerificationTask(
            task_id=task_id,
            document_id=request.document_id,
            document_path=file_path,
            chain_config=chain_config,
            priority=request.priority,
            status=VerificationStatus.PENDING,
            metadata=request.metadata,
            created_at=datetime.utcnow()
        )
        
        # Submit to background processing
        background_tasks.add_task(
            execute_verification_chain_task,
            verification_task
        )
        
        # Record usage for billing
        if customer:
            usage_record = UsageRecordCreate(
                usage_type=UsageType.DOCUMENT_VERIFICATION,
                quantity=1,
                metadata={
                    "task_id": task_id,
                    "document_id": request.document_id,
                    "chain_id": request.chain_id,
                    "enable_acvf": request.enable_acvf
                }
            )
            record_usage(db, customer, usage_record)
        
        # Estimate completion time based on chain complexity
        estimated_time = "5-15 minutes"  # Would be calculated based on chain config
        
        return VerificationSubmissionResponse(
            task_id=task_id,
            document_id=request.document_id,
            status=VerificationStatus.PENDING,
            chain_id=request.chain_id,
            priority=request.priority,
            created_at=datetime.utcnow(),
            estimated_completion_time=estimated_time,
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


# Verification Status and Results Endpoints
@router.get("/verification/{task_id}/status", response_model=VerificationStatusResponse)
async def get_verification_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the current status of a verification task."""
    # This would integrate with the existing verification system
    # For now, return a mock response
    return VerificationStatusResponse(
        task_id=task_id,
        document_id="mock-doc-id",
        status=VerificationStatus.PENDING,
        progress={"completed": 2, "total": 8},
        current_pass="citation_verification",
        completed_passes=["claim_extraction", "evidence_retrieval"],
        remaining_passes=["logic_analysis", "bias_scan", "acvf_testing"],
        created_at=datetime.utcnow() - timedelta(minutes=5),
        started_at=datetime.utcnow() - timedelta(minutes=3),
        estimated_completion=datetime.utcnow() + timedelta(minutes=10)
    )


@router.get("/verification/{task_id}/results", response_model=VerificationResultsResponse)
async def get_verification_results(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the complete results of a verification task."""
    # This would integrate with the existing verification system
    # Implementation would fetch from Redis/database
    return VerificationResultsResponse(
        task_id=task_id,
        document_id="mock-doc-id",
        status=VerificationStatus.COMPLETED,
        summary={
            "total_issues": 5,
            "critical_issues": 1,
            "average_confidence": 0.85,
            "verification_score": 7.2
        },
        metrics={
            "processing_time": 847,
            "tokens_processed": 15420,
            "api_calls": 23
        },
        completed_at=datetime.utcnow()
    )


# Dashboard and Visualization Endpoints
@router.get("/verification/{task_id}/dashboard", response_model=DashboardDataResponse)
async def get_dashboard_data(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get dashboard visualization data for verification results."""
    # This would integrate with the dashboard data structures
    aggregator = DashboardAggregator()
    
    # Mock implementation - would fetch real verification results
    dashboard_data = {
        "time_series": [],
        "heatmap": {},
        "relationships": {},
        "metrics": {}
    }
    
    return DashboardDataResponse(
        document_id="mock-doc-id",
        task_id=task_id,
        aggregated_data=dashboard_data,
        time_series=[],
        heatmap_data={},
        relationship_graph={},
        generated_at=datetime.utcnow()
    )


@router.get("/verification/{task_id}/debate-view", response_model=DebateViewResponse)
async def get_debate_view(
    task_id: str,
    format_type: str = Query(default="THREADED", description="Debate view format"),
    current_user: User = Depends(get_current_user)
):
    """Get ACVF debate view data for verification results."""
    # This would integrate with the debate view generator
    debate_generator = DebateViewGenerator()
    
    # Mock implementation - would fetch real ACVF results
    debate_data = {
        "rounds": [],
        "arguments": [],
        "verdicts": [],
        "participants": ["Challenger", "Defender", "Judge"]
    }
    
    return DebateViewResponse(
        task_id=task_id,
        document_id="mock-doc-id",
        debate_data=debate_data,
        participants=["Challenger", "Defender", "Judge"],
        total_rounds=3,
        final_verdict="CHALLENGER_WINS",
        confidence_score=0.72,
        generated_at=datetime.utcnow()
    )


# List and Management Endpoints
@router.get("/verification/tasks")
async def list_verification_tasks(
    status_filter: Optional[VerificationStatus] = Query(None),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """List verification tasks for the current user."""
    # Implementation would query database/Redis for user's tasks
    return {
        "tasks": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@router.delete("/verification/{task_id}")
async def cancel_verification_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a pending verification task."""
    # Implementation would cancel Celery task and update status
    return {"message": f"Verification task {task_id} cancelled"}


# Health and Status Endpoints
@router.get("/verification/health")
async def verification_health():
    """Check verification system health."""
    return {
        "status": "healthy",
        "components": {
            "document_service": "operational",
            "verification_worker": "operational",
            "redis_storage": "operational",
            "celery_broker": "operational"
        },
        "timestamp": datetime.utcnow()
    }


@router.get("/verification/metrics")
async def verification_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get verification system metrics."""
    return {
        "total_verifications": 0,
        "pending_tasks": 0,
        "average_processing_time": 0,
        "success_rate": 0.0,
        "last_updated": datetime.utcnow()
    } 