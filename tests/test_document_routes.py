"""
Tests for document submission and verification endpoints.

This module tests the API endpoints for:
- Document upload and validation
- Verification request submission
- Progress monitoring
- Results retrieval
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
from io import BytesIO

from src.api.main import app
from src.api.auth import User


class TestDocumentRoutes:
    """Test document submission and verification endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock authenticated user."""
        user = Mock(spec=User)
        user.id = "test_user_123"
        user.username = "testuser"
        user.email = "test@example.com"
        return user
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a sample PDF file for testing."""
        content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000204 00000 n\ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n299\n%%EOF"
        return BytesIO(content)
    
    @pytest.fixture
    def sample_docx_file(self):
        """Create a sample DOCX file for testing."""
        # Minimal DOCX content (not a real DOCX but enough for testing)
        content = b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00Mock DOCX content for testing"
        return BytesIO(content)
    
    @patch('src.api.routes.document_routes.get_current_user')
    @patch('src.api.routes.document_routes.get_customer_by_user')
    @patch('src.api.routes.document_routes.enforce_usage_limit')
    @patch('src.api.routes.document_routes.record_usage')
    @patch('src.api.routes.document_routes.DocumentIngestionService')
    def test_upload_document_success(self, mock_ingestion, mock_record_usage, 
                                   mock_enforce_limit, mock_get_customer, 
                                   mock_get_user, client, mock_user, sample_pdf_file):
        """Test successful document upload."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_customer = Mock()
        mock_customer.id = "customer_123"
        mock_get_customer.return_value = mock_customer
        mock_enforce_limit.return_value = None  # No exception = allowed
        
        mock_parsed_doc = Mock()
        mock_parsed_doc.id = "doc_123"
        mock_parsed_doc.title = "Test Document"
        mock_parsed_doc.total_pages = 1
        mock_ingestion.return_value.ingest_document.return_value = mock_parsed_doc
        
        # Make request
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        
        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        response_data = response.json()
        assert response_data["document"]["id"] == "doc_123"
        assert response_data["document"]["title"] == "Test Document"
        assert response_data["message"] == "Document uploaded successfully"
        
        # Verify usage tracking
        mock_enforce_limit.assert_called_once()
        mock_record_usage.assert_called_once()
    
    @patch('src.api.routes.document_routes.get_current_user')
    @patch('src.api.routes.document_routes.get_customer_by_user')
    @patch('src.api.routes.document_routes.enforce_usage_limit')
    def test_upload_document_usage_limit_exceeded(self, mock_enforce_limit, 
                                                 mock_get_customer, mock_get_user, 
                                                 client, mock_user, sample_pdf_file):
        """Test document upload when usage limit is exceeded."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_customer = Mock()
        mock_get_customer.return_value = mock_customer
        
        from fastapi import HTTPException
        mock_enforce_limit.side_effect = HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Usage limit exceeded"
        )
        
        # Make request
        files = {"file": ("test.pdf", sample_pdf_file, "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        
        # Assertions
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    
    def test_upload_document_invalid_file_type(self, client):
        """Test document upload with invalid file type."""
        # Create a text file
        text_content = BytesIO(b"This is not a PDF or DOCX file")
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        # Mock authentication
        with patch('src.api.routes.document_routes.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="user_123")
            response = client.post("/api/v1/documents/upload", files=files)
        
        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_document_file_too_large(self, client):
        """Test document upload with file too large."""
        # Create a large file (mock)
        large_content = BytesIO(b"x" * (51 * 1024 * 1024))  # 51MB
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        
        # Mock authentication
        with patch('src.api.routes.document_routes.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="user_123")
            response = client.post("/api/v1/documents/upload", files=files)
        
        # Assertions
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "File too large" in response.json()["detail"]
    
    @patch('src.api.routes.document_routes.get_current_user')
    @patch('src.api.routes.document_routes.get_customer_by_user')
    @patch('src.api.routes.document_routes.enforce_usage_limit')
    @patch('src.api.routes.document_routes.record_usage')
    @patch('src.api.routes.document_routes.VerificationWorker')
    @patch('src.api.routes.document_routes.get_parsed_document_by_id')
    def test_submit_verification_success(self, mock_get_doc, mock_worker, 
                                       mock_record_usage, mock_enforce_limit, 
                                       mock_get_customer, mock_get_user, 
                                       client, mock_user):
        """Test successful verification submission."""
        # Setup mocks
        mock_get_user.return_value = mock_user
        mock_customer = Mock()
        mock_get_customer.return_value = mock_customer
        mock_enforce_limit.return_value = None
        
        mock_doc = Mock()
        mock_doc.id = "doc_123"
        mock_get_doc.return_value = mock_doc
        
        mock_worker_instance = Mock()
        mock_worker_instance.submit_verification_task.return_value = {
            "task_id": "task_123",
            "status": "pending"
        }
        mock_worker.return_value = mock_worker_instance
        
        # Make request
        request_data = {
            "document_id": "doc_123",
            "chain_ids": ["comprehensive_verification"],
            "priority": "normal"
        }
        response = client.post("/api/v1/documents/verify", json=request_data)
        
        # Assertions
        assert response.status_code == status.HTTP_202_ACCEPTED
        response_data = response.json()
        assert response_data["task_id"] == "task_123"
        assert response_data["status"] == "pending"
    
    def test_submit_verification_document_not_found(self, client, mock_user):
        """Test verification submission with non-existent document."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            with patch('src.api.routes.document_routes.get_parsed_document_by_id') as mock_get_doc:
                mock_get_doc.return_value = None
                
                request_data = {
                    "document_id": "nonexistent_doc",
                    "chain_ids": ["comprehensive_verification"],
                    "priority": "normal"
                }
                response = client.post("/api/v1/documents/verify", json=request_data)
        
        # Assertions
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Document not found" in response.json()["detail"]
    
    @patch('src.api.routes.document_routes.get_current_user')
    @patch('src.api.routes.document_routes.get_task_status')
    def test_get_verification_status_success(self, mock_get_status, mock_get_user, 
                                           client, mock_user):
        """Test successful verification status retrieval."""
        mock_get_user.return_value = mock_user
        mock_get_status.return_value = {
            "task_id": "task_123",
            "status": "completed",
            "progress": 100,
            "current_pass": None,
            "estimated_completion": None
        }
        
        response = client.get("/api/v1/documents/verify/task_123/status")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["task_id"] == "task_123"
        assert response_data["status"] == "completed"
    
    @patch('src.api.routes.document_routes.get_current_user')
    @patch('src.api.routes.document_routes.get_task_result')
    def test_get_verification_result_success(self, mock_get_result, mock_get_user, 
                                           client, mock_user):
        """Test successful verification result retrieval."""
        mock_get_user.return_value = mock_user
        mock_result = {
            "task_id": "task_123",
            "document_id": "doc_123",
            "status": "completed",
            "results": {
                "overall_score": 0.85,
                "issues_found": 2,
                "verification_passes": ["claim_extraction", "citation_check"]
            }
        }
        mock_get_result.return_value = mock_result
        
        response = client.get("/api/v1/documents/verify/task_123/result")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["task_id"] == "task_123"
        assert response_data["status"] == "completed"
        assert "results" in response_data
    
    @patch('src.api.routes.document_routes.get_current_user')
    def test_list_user_documents_success(self, mock_get_user, client, mock_user):
        """Test successful user documents listing."""
        mock_get_user.return_value = mock_user
        
        with patch('src.api.routes.document_routes.get_documents_by_user') as mock_get_docs:
            mock_docs = [
                Mock(id="doc1", title="Document 1", created_at="2024-01-01T00:00:00Z"),
                Mock(id="doc2", title="Document 2", created_at="2024-01-02T00:00:00Z")
            ]
            mock_get_docs.return_value = mock_docs
            
            response = client.get("/api/v1/documents/")
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert len(response_data["documents"]) == 2
    
    @patch('src.api.routes.document_routes.get_current_user')
    def test_get_document_details_success(self, mock_get_user, client, mock_user):
        """Test successful document details retrieval."""
        mock_get_user.return_value = mock_user
        
        with patch('src.api.routes.document_routes.get_parsed_document_by_id') as mock_get_doc:
            mock_doc = Mock()
            mock_doc.id = "doc_123"
            mock_doc.title = "Test Document"
            mock_doc.total_pages = 5
            mock_doc.created_at = "2024-01-01T00:00:00Z"
            mock_get_doc.return_value = mock_doc
            
            response = client.get("/api/v1/documents/doc_123")
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data["id"] == "doc_123"
            assert response_data["title"] == "Test Document"
    
    def test_get_document_details_not_found(self, client, mock_user):
        """Test document details retrieval with non-existent document."""
        with patch('src.api.routes.document_routes.get_current_user') as mock_auth:
            mock_auth.return_value = mock_user
            with patch('src.api.routes.document_routes.get_parsed_document_by_id') as mock_get_doc:
                mock_get_doc.return_value = None
                
                response = client.get("/api/v1/documents/nonexistent")
                
                assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDocumentRoutesEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_upload_without_authentication(self, client):
        """Test document upload without authentication."""
        files = {"file": ("test.pdf", BytesIO(b"fake content"), "application/pdf")}
        response = client.post("/api/v1/documents/upload", files=files)
        
        # Should get authentication error
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
    
    def test_verify_without_authentication(self, client):
        """Test verification submission without authentication."""
        request_data = {"document_id": "doc_123", "chain_ids": ["test"]}
        response = client.post("/api/v1/documents/verify", json=request_data)
        
        # Should get authentication error
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
    
    def test_upload_empty_file(self, client):
        """Test upload with empty file."""
        empty_file = BytesIO(b"")
        files = {"file": ("empty.pdf", empty_file, "application/pdf")}
        
        with patch('src.api.routes.document_routes.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="user_123")
            response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in response.json()["detail"].lower() 