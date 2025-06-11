"""
Test suite for document verification API endpoints.

Tests document upload, verification submission, and status checking.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import io

from src.api.routes.document_routes import (
    validate_file_extension, validate_mime_type, generate_document_id
)


class TestDocumentValidation:
    """Test document validation functions."""
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        assert validate_file_extension("document.pdf") is True
        assert validate_file_extension("document.docx") is True
        assert validate_file_extension("document.md") is True
        assert validate_file_extension("document.txt") is True
        assert validate_file_extension("document.exe") is False
        assert validate_file_extension("document.php") is False
    
    def test_validate_mime_type(self):
        """Test MIME type validation."""
        assert validate_mime_type("application/pdf") is True
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


class TestDocumentRoutes:
    """Test document API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.document_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        
        # Mock dependencies
        from src.api.auth import User
        
        def mock_get_current_user():
            return User(id=1, username="test", email="test@example.com")
        
        def mock_get_db():
            return MagicMock()
        
        app.dependency_overrides.update({
            "src.api.routes.document_routes.get_current_user": mock_get_current_user,
            "src.api.routes.document_routes.get_db": mock_get_db,
        })
        
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_list_documents(self, client):
        """Test document listing endpoint."""
        with patch('src.api.routes.document_routes.get_customer_by_user') as mock_customer:
            mock_customer.return_value = None
            
            response = client.get("/api/v1/")
            
            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data


class TestVerificationStatusRoutes:
    """Test verification status API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from src.api.routes.verification_status_routes import router
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        
        # Mock dependencies
        from src.api.auth import User
        
        def mock_get_current_user():
            return User(id=1, username="test", email="test@example.com")
        
        app.dependency_overrides.update({
            "src.api.routes.verification_status_routes.get_current_user": mock_get_current_user,
        })
        
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_verification_health(self, client):
        """Test verification health endpoint."""
        response = client.get("/api/v1/verification/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
    
    def test_verification_metrics(self, client):
        """Test verification metrics endpoint."""
        response = client.get("/api/v1/verification/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_verifications" in data
        assert "pending_tasks" in data
        assert "success_rate" in data
    
    def test_get_verification_status(self, client):
        """Test verification status endpoint."""
        task_id = "test-task-123"
        response = client.get(f"/api/v1/verification/{task_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
    
    def test_list_verification_tasks(self, client):
        """Test verification tasks listing."""
        response = client.get("/api/v1/verification/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data


if __name__ == "__main__":
    pytest.main([__file__]) 