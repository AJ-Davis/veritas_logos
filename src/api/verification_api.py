"""
FastAPI application for verification chain management.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import uuid
import json
import redis
import os

from src.models.verification import (
    VerificationTask,
    VerificationChainConfig,
    VerificationChainResult,
    VerificationStatus,
    Priority,
    VerificationMetrics
)
from src.verification.config.chain_loader import ChainConfigLoader, create_default_chain_configs
from src.verification.workers.verification_worker import execute_verification_chain_task, celery_app
from src.document_ingestion import document_service


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VeritasLogos Verification API",
    description="API for document verification chain management",
    version="1.0.0"
)

# Configure CORS with secure origins
# Define trusted origins that are allowed to access this API
ALLOWED_ORIGINS = [
    "http://localhost:3000",        # React development server
    "http://localhost:8080",        # Vue.js development server
    "http://localhost:5173",        # Vite development server
    "https://your-frontend-domain.com",  # Production frontend domain
    "https://admin.your-domain.com",     # Admin dashboard domain
    # Add your specific trusted domains here
]

app.add_middleware(
    CORSMiddleware,
    # Only allow requests from trusted origins to prevent CSRF attacks
    # and unauthorized access from malicious websites
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,  # Required for authentication cookies/headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=["*"],     # Can be restricted further if needed
)

# Global configuration loader
config_loader = ChainConfigLoader()

# Redis connection for persistent storage across multiple workers/pods
# Reuse the same Redis instance that Celery uses for consistency
redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(redis_url, decode_responses=True)

# Storage key prefixes for Redis
TASK_STORAGE_PREFIX = "verification_task:"
RESULT_STORAGE_PREFIX = "verification_result:"


class RedisTaskStorage:
    """Redis-backed task storage for multi-worker safety."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def store_task(self, task: VerificationTask) -> None:
        """Store a verification task in Redis."""
        key = f"{TASK_STORAGE_PREFIX}{task.task_id}"
        # Convert task to dict and serialize to JSON
        task_data = task.dict()
        # Handle datetime serialization
        task_data['created_at'] = task_data['created_at'].isoformat()
        self.redis.setex(key, 86400, json.dumps(task_data))  # 24 hour TTL
    
    def get_task(self, task_id: str) -> Optional[VerificationTask]:
        """Retrieve a verification task from Redis."""
        key = f"{TASK_STORAGE_PREFIX}{task_id}"
        task_data = self.redis.get(key)
        if not task_data:
            return None
        
        try:
            data = json.loads(task_data)
            # Convert datetime string back to datetime object
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            # Reconstruct chain_config from dict
            data['chain_config'] = VerificationChainConfig(**data['chain_config'])
            return VerificationTask(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to deserialize task {task_id}: {e}")
            return None
    
    def update_task_status(self, task_id: str, status: VerificationStatus) -> bool:
        """Update task status in Redis."""
        task = self.get_task(task_id)
        if not task:
            return False
        
        task.status = status
        self.store_task(task)
        return True
    
    def get_all_tasks(self, limit: int = 100) -> List[VerificationTask]:
        """Get all tasks from Redis."""
        pattern = f"{TASK_STORAGE_PREFIX}*"
        keys = self.redis.keys(pattern)
        tasks = []
        
        for key in keys[:limit]:
            task_id = key.replace(TASK_STORAGE_PREFIX, "")
            task = self.get_task(task_id)
            if task:
                tasks.append(task)
        
        return tasks
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists in Redis."""
        key = f"{TASK_STORAGE_PREFIX}{task_id}"
        return bool(self.redis.exists(key))
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task from Redis."""
        key = f"{TASK_STORAGE_PREFIX}{task_id}"
        return bool(self.redis.delete(key))


class RedisResultStorage:
    """Redis-backed result storage for multi-worker safety."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def store_result(self, task_id: str, result: VerificationChainResult) -> None:
        """Store a verification result in Redis."""
        key = f"{RESULT_STORAGE_PREFIX}{task_id}"
        # Convert result to dict and serialize to JSON
        result_data = result.dict()
        # Handle datetime serialization
        if result_data.get('execution_start_time'):
            result_data['execution_start_time'] = result_data['execution_start_time'].isoformat()
        if result_data.get('execution_end_time'):
            result_data['execution_end_time'] = result_data['execution_end_time'].isoformat()
        
        self.redis.setex(key, 86400, json.dumps(result_data))  # 24 hour TTL
    
    def get_result(self, task_id: str) -> Optional[VerificationChainResult]:
        """Retrieve a verification result from Redis."""
        key = f"{RESULT_STORAGE_PREFIX}{task_id}"
        result_data = self.redis.get(key)
        if not result_data:
            return None
        
        try:
            data = json.loads(result_data)
            # Convert datetime strings back to datetime objects
            if data.get('execution_start_time'):
                data['execution_start_time'] = datetime.fromisoformat(data['execution_start_time'])
            if data.get('execution_end_time'):
                data['execution_end_time'] = datetime.fromisoformat(data['execution_end_time'])
            
            return VerificationChainResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to deserialize result {task_id}: {e}")
            return None
    
    def get_all_results(self) -> List[VerificationChainResult]:
        """Get all results from Redis."""
        pattern = f"{RESULT_STORAGE_PREFIX}*"
        keys = self.redis.keys(pattern)
        results = []
        
        for key in keys:
            task_id = key.replace(RESULT_STORAGE_PREFIX, "")
            result = self.get_result(task_id)
            if result:
                results.append(result)
        
        return results
    
    def result_exists(self, task_id: str) -> bool:
        """Check if a result exists in Redis."""
        key = f"{RESULT_STORAGE_PREFIX}{task_id}"
        return bool(self.redis.exists(key))


# Initialize Redis-backed storage
task_storage = RedisTaskStorage(redis_client)
result_storage = RedisResultStorage(redis_client)


# Dependency injection
def get_config_loader() -> ChainConfigLoader:
    """Get configuration loader instance."""
    return config_loader


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting VeritasLogos Verification API")
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info(f"Successfully connected to Redis at {redis_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_url}: {str(e)}")
        raise RuntimeError(f"Redis connection failed: {str(e)}")
    
    # Initialize default configurations
    try:
        default_configs = create_default_chain_configs()
        for chain_id, config_data in default_configs.items():
            # Convert dict to VerificationChainConfig
            chain_config = VerificationChainConfig(**config_data)
            config_loader.loaded_chains[chain_id] = chain_config
        
        logger.info(f"Loaded {len(config_loader.loaded_chains)} default chain configurations")
        
    except Exception as e:
        logger.error(f"Failed to load default configurations: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down VeritasLogos Verification API")


# API Models for requests/responses
from pydantic import BaseModel, Field

class SubmitVerificationRequest(BaseModel):
    """Request model for submitting verification tasks."""
    document_id: str = Field(..., description="Document identifier or file path")
    chain_id: str = Field(..., description="Verification chain to use")
    priority: Priority = Field(Priority.MEDIUM, description="Task priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerificationTaskResponse(BaseModel):
    """Response model for verification task submission."""
    task_id: str
    status: VerificationStatus
    created_at: datetime
    estimated_completion_time: Optional[str] = None
    message: str


class TaskStatusResponse(BaseModel):
    """Response model for task status queries."""
    task_id: str
    status: VerificationStatus
    progress: Optional[Dict[str, Any]] = None
    result: Optional[VerificationChainResult] = None
    error_message: Optional[str] = None


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "VeritasLogos Verification API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    try:
        # Check Celery worker health
        celery_health = celery_app.control.ping(timeout=1.0)
        celery_status = "healthy" if celery_health else "unhealthy"
        
        # Check Redis connectivity
        try:
            redis_client.ping()
            redis_status = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_status = "unhealthy"
        
        # Count active tasks from Redis
        try:
            active_task_count = len(task_storage.get_all_tasks())
        except Exception as e:
            logger.error(f"Failed to count active tasks: {e}")
            active_task_count = 0
        
        overall_status = "healthy" if all([
            celery_status == "healthy",
            redis_status == "healthy"
        ]) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "api": "healthy",
                "celery": celery_status,
                "redis": redis_status,
                "document_service": "healthy",
                "config_loader": "healthy"
            },
            "loaded_chains": len(config_loader.loaded_chains),
            "active_tasks": active_task_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/chains", response_model=Dict[str, VerificationChainConfig])
async def list_verification_chains(
    config_loader: ChainConfigLoader = Depends(get_config_loader)
):
    """List all available verification chains."""
    return config_loader.loaded_chains


@app.get("/chains/{chain_id}", response_model=VerificationChainConfig)
async def get_verification_chain(
    chain_id: str,
    config_loader: ChainConfigLoader = Depends(get_config_loader)
):
    """Get a specific verification chain configuration."""
    chain_config = config_loader.get_chain_config(chain_id)
    if not chain_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification chain '{chain_id}' not found"
        )
    return chain_config


@app.post("/verify", response_model=VerificationTaskResponse)
async def submit_verification_task(
    request: SubmitVerificationRequest,
    background_tasks: BackgroundTasks,
    config_loader: ChainConfigLoader = Depends(get_config_loader)
):
    """Submit a document for verification."""
    
    # Validate chain configuration
    chain_config = config_loader.get_chain_config(request.chain_id)
    if not chain_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown verification chain: {request.chain_id}"
        )
    
    # Validate document
    try:
        validation = document_service.validate_document(request.document_id)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document validation failed: {validation['error']}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to validate document: {str(e)}"
        )
    
    # Create verification task
    verification_task = VerificationTask(
        document_id=request.document_id,
        chain_config=chain_config,
        priority=request.priority,
        metadata=request.metadata
    )
    
    # Store task in Redis
    task_storage.store_task(verification_task)
    
    # Submit to Celery
    try:
        # Use the SAME id for both layers
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(verification_task)
        celery_task = execute_verification_chain_task.apply_async(
            args=[payload], 
            task_id=verification_task.task_id  # keep ids in sync
        )
        
        logger.info(f"Submitted verification task {verification_task.task_id} to Celery")
        
        # Estimate completion time (simplified)
        estimated_time = f"{chain_config.global_timeout_seconds} seconds"
        
        return VerificationTaskResponse(
            task_id=verification_task.task_id,
            status=VerificationStatus.PENDING,
            created_at=verification_task.created_at,
            estimated_completion_time=estimated_time,
            message="Verification task submitted successfully"
        )
        
    except Exception as e:
        # Remove from storage if submission failed
        task_storage.delete_task(verification_task.task_id)
        logger.error(f"Failed to submit task to Celery: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit verification task: {str(e)}"
        )


@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a verification task."""
    
    # Check if task exists in Redis
    task = task_storage.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found"
        )
    
    # Check Celery task status
    try:
        celery_task = celery_app.AsyncResult(task_id)
        
        if celery_task.state == 'PENDING':
            status_response = TaskStatusResponse(
                task_id=task_id,
                status=VerificationStatus.PENDING,
                progress={"message": "Task is queued for processing"}
            )
        elif celery_task.state == 'STARTED':
            status_response = TaskStatusResponse(
                task_id=task_id,
                status=VerificationStatus.RUNNING,
                progress={"message": "Task is currently running"}
            )
        elif celery_task.state == 'SUCCESS':
            result_data = celery_task.result
            chain_result = VerificationChainResult(**result_data)
            result_storage.store_result(task_id, chain_result)
            
            status_response = TaskStatusResponse(
                task_id=task_id,
                status=VerificationStatus.COMPLETED,
                result=chain_result
            )
        elif celery_task.state == 'FAILURE':
            status_response = TaskStatusResponse(
                task_id=task_id,
                status=VerificationStatus.FAILED,
                error_message=str(celery_task.info)
            )
        else:
            status_response = TaskStatusResponse(
                task_id=task_id,
                status=VerificationStatus.PENDING,
                progress={"state": celery_task.state}
            )
        
        return status_response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task status: {str(e)}"
        )


@app.get("/tasks/{task_id}/result", response_model=VerificationChainResult)
async def get_task_result(task_id: str):
    """Get the detailed result of a completed verification task."""
    
    # Check if result exists in Redis
    chain_result = result_storage.get_result(task_id)
    if not chain_result:
        # Try to get from Celery
        try:
            celery_task = celery_app.AsyncResult(task_id)
            if celery_task.state == 'SUCCESS':
                result_data = celery_task.result
                chain_result = VerificationChainResult(**result_data)
                result_storage.store_result(task_id, chain_result)
                return chain_result
        except Exception as e:
            logger.error(f"Failed to retrieve result from Celery: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result for task '{task_id}' not found or not ready"
        )
    
    return chain_result


@app.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(
    status_filter: Optional[VerificationStatus] = None,
    limit: int = 100
):
    """List verification tasks with optional status filtering."""
    
    tasks = []
    all_tasks = task_storage.get_all_tasks(limit)
    
    for task in all_tasks:
        task_info = {
            "task_id": task.task_id,
            "document_id": task.document_id,
            "chain_id": task.chain_config.chain_id,
            "priority": task.priority,
            "created_at": task.created_at,
            "status": task.status
        }
        
        # Apply status filter
        if status_filter is None or task.status == status_filter:
            tasks.append(task_info)
    
    return tasks


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running verification task."""
    
    if not task_storage.task_exists(task_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found"
        )
    
    try:
        # Revoke the Celery task
        celery_app.control.revoke(task_id, terminate=True)
        
        # Update task status in Redis
        task_storage.update_task_status(task_id, VerificationStatus.CANCELLED)
        
        logger.info(f"Cancelled verification task {task_id}")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@app.get("/metrics", response_model=VerificationMetrics)
async def get_metrics():
    """Get verification system metrics."""
    
    # Calculate metrics from stored tasks in Redis
    all_tasks = task_storage.get_all_tasks()
    total_tasks = len(all_tasks)
    completed_tasks = len([t for t in all_tasks if t.status == VerificationStatus.COMPLETED])
    failed_tasks = len([t for t in all_tasks if t.status == VerificationStatus.FAILED])
    successful_tasks = completed_tasks - failed_tasks
    
    # Calculate average execution time from results in Redis
    execution_times = []
    all_results = result_storage.get_all_results()
    for result in all_results:
        if result.total_execution_time_seconds:
            execution_times.append(result.total_execution_time_seconds)
    
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
    
    return VerificationMetrics(
        total_tasks_processed=total_tasks,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        average_execution_time_seconds=avg_execution_time
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)