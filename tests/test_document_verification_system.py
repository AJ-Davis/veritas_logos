"""
Test suite for document submission and verification system.

This module tests document upload, verification submission, status checking,
results retrieval, and integration with authentication and billing systems.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import io

# Import the models and functions to test
from src.api.routes.document_routes import (
    router as document_router,
    ensure_upload_directory, validate_file_extension, validate_mime_type,
    generate_document_id, get_file_path, UPLOAD_DIR, MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES
)
from src.api.routes.verification_status_routes import router as verification_status_router
from src.api.auth import User, create_database_tables, Base
from src.api.billing import Customer, SubscriptionTier


class TestDocumentManagement:
    """Test document upload and management functionality."""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    @pytest.fixture
    def test_user(self, db_session):
        """Create test user."""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
            role="USER",
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        return user
    
    @pytest.fixture
    def test_customer(self, db_session, test_user):
        """Create test customer."""
        customer = Customer(
            id=1,
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            subscription_tier=SubscriptionTier.STARTER,
            email=test_user.email
        )
        db_session.add(customer)
        db_session.commit()
        return customer
    
    def test_ensure_upload_directory(self):
        """Test upload directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                ensure_upload_directory()
                assert os.path.exists(temp_dir)
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        assert validate_file_extension("document.pdf") is True
        assert validate_file_extension("document.docx") is True
        assert validate_file_extension("document.md") is True
        assert validate_file_extension("document.txt") is True
        assert validate_file_extension("document.exe") is False
        assert validate_file_extension("document.php") is False
        assert validate_file_extension("document") is False
    
    def test_validate_mime_type(self):
        """Test MIME type validation."""
        assert validate_mime_type("application/pdf") is True
        assert validate_mime_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document") is True
        assert validate_mime_type("text/markdown") is True
        assert validate_mime_type("text/plain") is True
        assert validate_mime_type("application/exe") is False
        assert validate_mime_type("text/html") is False
    
    def test_generate_document_id(self):
        """Test document ID generation."""
        doc_id = generate_document_id()
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36  # UUID4 format
        
        # Test uniqueness
        doc_id2 = generate_document_id()
        assert doc_id != doc_id2
    
    def test_get_file_path(self):
        """Test file path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                doc_id = "test-doc-id"
                filename = "test.pdf"
                path = get_file_path(doc_id, filename)
                
                expected_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
                assert path == expected_path
                assert os.path.exists(temp_dir)  # Directory should be created


class TestDocumentUpload:
    """Test document upload functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.document_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock authentication and billing dependencies."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_user, \
             patch('src.api.routes.document_routes.get_db') as mock_db, \
             patch('src.api.routes.document_routes.get_customer_by_user') as mock_customer, \
             patch('src.api.routes.document_routes.enforce_usage_limit') as mock_limit, \
             patch('src.api.routes.document_routes.record_usage') as mock_usage:
            
            # Mock user
            mock_user.return_value = User(id=1, username="test", email="test@example.com")
            
            # Mock database session
            mock_db.return_value = MagicMock()
            
            # Mock customer
            mock_customer.return_value = Customer(id=1, user_id=1, subscription_tier=SubscriptionTier.STARTER)
            
            yield {
                'user': mock_user,
                'db': mock_db,
                'customer': mock_customer,
                'limit': mock_limit,
                'usage': mock_usage
            }
    
    def test_upload_document_success(self, client, mock_dependencies):
        """Test successful document upload."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                # Create test file content
                file_content = b"This is a test PDF content"
                
                # Mock file upload
                files = {
                    "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
                }
                
                with patch('src.api.routes.document_routes.generate_document_id') as mock_id:
                    mock_id.return_value = "test-doc-123"
                    
                    response = client.post("/api/v1/upload", files=files)
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["document_id"] == "test-doc-123"
                    assert data["filename"] == "test.pdf"
                    assert data["file_size"] == len(file_content)
                    assert data["mime_type"] == "application/pdf"
                    assert "upload_timestamp" in data
                    assert data["message"] == "Document uploaded successfully"
    
    def test_upload_document_no_filename(self, client, mock_dependencies):
        """Test upload with no filename."""
        files = {"file": ("", io.BytesIO(b"content"), "application/pdf")}
        
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 400
        assert "No filename provided" in response.json()["detail"]
    
    def test_upload_document_invalid_extension(self, client, mock_dependencies):
        """Test upload with invalid file extension."""
        files = {"file": ("test.exe", io.BytesIO(b"content"), "application/exe")}
        
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_document_too_large(self, client, mock_dependencies):
        """Test upload with file too large."""
        large_content = b"x" * (MAX_FILE_SIZE + 1)
        files = {"file": ("test.pdf", io.BytesIO(large_content), "application/pdf")}
        
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
    
    def test_upload_document_invalid_mime_type(self, client, mock_dependencies):
        """Test upload with invalid MIME type."""
        with patch('mimetypes.guess_type') as mock_mime:
            mock_mime.return_value = ("application/exe", None)
            
            files = {"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")}
            
            response = client.post("/api/v1/upload", files=files)
            assert response.status_code == 400
            assert "Unsupported MIME type" in response.json()["detail"]


class TestDocumentMetadata:
    """Test document metadata functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.document_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock authentication dependencies."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_user:
            mock_user.return_value = User(id=1, username="test", email="test@example.com")
            yield {'user': mock_user}
    
    def test_get_document_metadata_success(self, client, mock_dependencies):
        """Test successful metadata retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            doc_id = "test-doc-123"
            filename = "test.pdf"
            file_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
            
            with open(file_path, "wb") as f:
                f.write(b"test content")
            
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                response = client.get(f"/api/v1/{doc_id}/metadata")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["document_id"] == doc_id
                assert data["filename"] == filename
                assert data["file_size"] > 0
                assert "upload_timestamp" in data
                assert data["user_id"] == "1"
    
    def test_get_document_metadata_not_found(self, client, mock_dependencies):
        """Test metadata retrieval for non-existent document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                response = client.get("/api/v1/nonexistent-doc/metadata")
                
                assert response.status_code == 404
                assert "Document not found" in response.json()["detail"]


class TestDocumentDeletion:
    """Test document deletion functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.document_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock authentication dependencies."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_user:
            mock_user.return_value = User(id=1, username="test", email="test@example.com")
            yield {'user': mock_user}
    
    def test_delete_document_success(self, client, mock_dependencies):
        """Test successful document deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            doc_id = "test-doc-123"
            filename = "test.pdf"
            file_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
            
            with open(file_path, "wb") as f:
                f.write(b"test content")
            
            assert os.path.exists(file_path)
            
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                response = client.delete(f"/api/v1/{doc_id}")
                
                assert response.status_code == 200
                assert f"Document {doc_id} deleted successfully" in response.json()["message"]
                assert not os.path.exists(file_path)
    
    def test_delete_document_not_found(self, client, mock_dependencies):
        """Test deletion of non-existent document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                response = client.delete("/api/v1/nonexistent-doc")
                
                assert response.status_code == 404
                assert "Document not found" in response.json()["detail"]


class TestVerificationSubmission:
    """Test verification submission functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.document_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock authentication and billing dependencies."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_user, \
             patch('src.api.routes.document_routes.get_db') as mock_db, \
             patch('src.api.routes.document_routes.get_customer_by_user') as mock_customer, \
             patch('src.api.routes.document_routes.enforce_usage_limit') as mock_limit, \
             patch('src.api.routes.document_routes.record_usage') as mock_usage:
            
            mock_user.return_value = User(id=1, username="test", email="test@example.com")
            mock_db.return_value = MagicMock()
            mock_customer.return_value = Customer(id=1, user_id=1, subscription_tier=SubscriptionTier.STARTER)
            
            yield {
                'user': mock_user,
                'db': mock_db,
                'customer': mock_customer,
                'limit': mock_limit,
                'usage': mock_usage
            }
    
    def test_submit_verification_success(self, client, mock_dependencies):
        """Test successful verification submission."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            doc_id = "test-doc-123"
            filename = "test.pdf"
            file_path = os.path.join(temp_dir, f"{doc_id}_{filename}")
            
            with open(file_path, "wb") as f:
                f.write(b"test content")
            
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                request_data = {
                    "document_id": doc_id,
                    "chain_id": "comprehensive",
                    "priority": "HIGH",
                    "enable_acvf": True,
                    "metadata": {"test": "data"}
                }
                
                response = client.post(f"/api/v1/{doc_id}/verify", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                
                assert "task_id" in data
                assert data["document_id"] == doc_id
                assert data["status"] == "PENDING"
                assert data["chain_id"] == "comprehensive"
                assert data["priority"] == "HIGH"
                assert "created_at" in data
                assert data["message"] == "Verification task submitted successfully"
    
    def test_submit_verification_document_not_found(self, client, mock_dependencies):
        """Test verification submission for non-existent document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.api.routes.document_routes.UPLOAD_DIR', temp_dir):
                request_data = {
                    "document_id": "nonexistent-doc",
                    "chain_id": "comprehensive",
                    "priority": "MEDIUM",
                    "enable_acvf": True
                }
                
                response = client.post("/api/v1/nonexistent-doc/verify", json=request_data)
                
                assert response.status_code == 404
                assert "Document not found" in response.json()["detail"]


class TestVerificationStatus:
    """Test verification status and results functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.verification_status_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock authentication dependencies."""
        with patch('src.api.routes.verification_status_routes.get_current_user') as mock_user:
            mock_user.return_value = User(id=1, username="test", email="test@example.com")
            yield {'user': mock_user}
    
    def test_get_verification_status(self, client, mock_dependencies):
        """Test verification status retrieval."""
        task_id = "test-task-123"
        
        response = client.get(f"/api/v1/verification/{task_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == task_id
        assert "document_id" in data
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data
    
    def test_get_verification_results(self, client, mock_dependencies):
        """Test verification results retrieval."""
        task_id = "test-task-123"
        
        response = client.get(f"/api/v1/verification/{task_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == task_id
        assert "document_id" in data
        assert "status" in data
        assert "summary" in data
        assert "metrics" in data
    
    def test_get_dashboard_data(self, client, mock_dependencies):
        """Test dashboard data retrieval."""
        task_id = "test-task-123"
        
        response = client.get(f"/api/v1/verification/{task_id}/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == task_id
        assert "document_id" in data
        assert "aggregated_data" in data
        assert "generated_at" in data
    
    def test_get_debate_view(self, client, mock_dependencies):
        """Test debate view data retrieval."""
        task_id = "test-task-123"
        
        response = client.get(f"/api/v1/verification/{task_id}/debate-view")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["task_id"] == task_id
        assert "document_id" in data
        assert "debate_data" in data
        assert "participants" in data
        assert "generated_at" in data
    
    def test_list_verification_tasks(self, client, mock_dependencies):
        """Test verification tasks listing."""
        response = client.get("/api/v1/verification/tasks")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tasks" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_cancel_verification_task(self, client, mock_dependencies):
        """Test verification task cancellation."""
        task_id = "test-task-123"
        
        response = client.delete(f"/api/v1/verification/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert f"Verification task {task_id} cancelled" in data["message"]
    
    def test_verification_health(self, client, mock_dependencies):
        """Test verification system health check."""
        response = client.get("/api/v1/verification/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
    
    def test_verification_metrics(self, client, mock_dependencies):
        """Test verification metrics retrieval."""
        response = client.get("/api/v1/verification/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_verifications" in data
        assert "pending_tasks" in data
        assert "average_processing_time" in data
        assert "success_rate" in data
        assert "last_updated" in data


if __name__ == "__main__":
    pytest.main([__file__]) 