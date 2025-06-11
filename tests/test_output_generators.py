"""
Unit tests for PDF and DOCX output generators.

This module tests the document generation functionality including PDF and DOCX
creation with annotations, styling, content organization, and error handling.
"""

import pytest
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.models.output_generators import (
    PdfGenerator, DocxGenerator, BaseDocumentGenerator,
    DocumentGeneratorError, create_document_generator,
    generate_pdf_report, generate_docx_report
)
from src.models.output import (
    OutputVerificationResult, DocumentAnnotation, AnnotationLayer,
    AnnotationType, HighlightStyle, ColorScheme, OutputGenerationConfig,
    OutputFormat, DebateViewOutput
)
from src.models.issues import UnifiedIssue, IssueSeverity, IssueType, IssueRegistry
from src.models.document import ParsedDocument, DocumentSection
from src.models.verification import VerificationStatus, VerificationResult


class TestBaseDocumentGenerator:
    """Test the base document generator functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        generator = BaseDocumentGenerator()
        assert generator.config is not None
        assert generator.color_scheme is not None
        assert generator.temp_files == []
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = OutputGenerationConfig(
            output_format=OutputFormat.PDF,
            include_summary=False
        )
        generator = BaseDocumentGenerator(config)
        assert generator.config == config
        assert generator.config.include_summary is False
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with BaseDocumentGenerator() as generator:
            assert generator is not None
        # Should complete without error
    
    def test_severity_color_mapping(self):
        """Test severity color mapping."""
        generator = BaseDocumentGenerator()
        
        for severity in IssueSeverity:
            color = generator._get_severity_color(severity)
            assert isinstance(color, str)
            assert color.startswith('#')
            assert len(color) == 7
    
    def test_issue_filtering(self):
        """Test issue filtering based on configuration."""
        config = OutputGenerationConfig(
            output_format=OutputFormat.PDF,
            minimum_issue_severity=IssueSeverity.MEDIUM
        )
        generator = BaseDocumentGenerator(config)
        
        # Create mock issues
        high_issue = Mock()
        high_issue.severity = IssueSeverity.HIGH
        
        low_issue = Mock()
        low_issue.severity = IssueSeverity.LOW
        
        assert generator._should_include_issue(high_issue) is True
        assert generator._should_include_issue(low_issue) is False
    
    def test_text_truncation(self):
        """Test text truncation functionality."""
        generator = BaseDocumentGenerator()
        
        short_text = "Short text"
        assert generator._truncate_text(short_text) == short_text
        
        long_text = "x" * 200
        truncated = generator._truncate_text(long_text, 50)
        assert len(truncated) == 50
        assert truncated.endswith("...")
    
    def test_timestamp_formatting(self):
        """Test timestamp formatting."""
        generator = BaseDocumentGenerator()
        
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        formatted = generator._format_timestamp(timestamp)
        
        assert "2024-01-15 10:30:45 UTC" in formatted


class TestPdfGenerator:
    """Test the PDF generator functionality."""
    
    @pytest.fixture
    def sample_output_result(self):
        """Create a sample output verification result."""
        # Create mock document
        document = Mock(spec=ParsedDocument)
        document.content = "This is a sample document with some content for testing PDF generation."
        document.document_id = "test-doc-123"
        
        # Create mock issues
        issue1 = Mock(spec=UnifiedIssue)
        issue1.issue_id = "issue-1"
        issue1.issue_type = IssueType.FACTUAL_ERROR
        issue1.severity = IssueSeverity.HIGH
        issue1.description = "Sample factual accuracy issue"
        issue1.confidence = 0.85
        issue1.location = Mock()
        issue1.location.start_position = 10
        issue1.location.end_position = 25
        issue1.remediation_suggestion = "Check the facts"
        
        issue_registry = Mock(spec=IssueRegistry)
        issue_registry.issues = [issue1]
        
        # Create mock annotations
        annotation1 = Mock(spec=DocumentAnnotation)
        annotation1.annotation_id = "ann-1"
        annotation1.start_position = 10
        annotation1.end_position = 25
        annotation1.content = "This text needs verification"
        annotation1.style = HighlightStyle.HIGH
        annotation1.related_issue_id = "issue-1"
        
        layer1 = Mock(spec=AnnotationLayer)
        layer1.layer_id = "layer-1"
        layer1.annotations = [annotation1]
        
        # Create output result
        result = Mock(spec=OutputVerificationResult)
        result.document_id = "test-doc-123"
        result.document = document
        result.verification_status = "verified_with_issues"
        result.total_issues = 1
        result.overall_confidence = 0.75
        result.generated_at = datetime.now(timezone.utc)
        result.generator_version = "1.0.0"
        result.issue_registry = issue_registry
        result.annotation_layers = [layer1]
        result.critical_issues = []
        result.debate_views = []
        result.generate_summary.return_value = {
            'total_issues': 1,
            'verification_status': 'verified_with_issues',
            'overall_confidence': 0.75,
            'processing_time_seconds': 1.5,
            'issues_by_severity': {IssueSeverity.HIGH: 1}
        }
        
        return result
    
    def test_initialization(self):
        """Test PDF generator initialization."""
        generator = PdfGenerator()
        assert generator.styles is not None
        assert generator.page_width > 0
        assert generator.page_height > 0
        assert generator.margin > 0
    
    def test_custom_config_initialization(self):
        """Test PDF generator with custom config."""
        config = OutputGenerationConfig(
            output_format=OutputFormat.PDF,
            include_table_of_contents=False
        )
        generator = PdfGenerator(config)
        assert generator.config.include_table_of_contents is False
    
    @patch('src.models.output_generators.SimpleDocTemplate')
    def test_generate_basic_pdf(self, mock_doc_template, sample_output_result):
        """Test basic PDF generation."""
        # Setup mock
        mock_doc = Mock()
        mock_doc_template.return_value = mock_doc
        
        generator = PdfGenerator()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result_path = generator.generate(sample_output_result, output_path)
            
            assert result_path == output_path
            mock_doc.build.assert_called_once()
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_create_title_page(self, sample_output_result):
        """Test title page creation."""
        generator = PdfGenerator()
        story = generator._create_title_page(sample_output_result)
        
        assert len(story) > 0
        # Should contain title and document info
    
    def test_create_table_of_contents(self, sample_output_result):
        """Test table of contents creation."""
        generator = PdfGenerator()
        story = generator._create_table_of_contents(sample_output_result)
        
        assert len(story) > 0
    
    def test_create_executive_summary(self, sample_output_result):
        """Test executive summary creation."""
        generator = PdfGenerator()
        story = generator._create_executive_summary(sample_output_result)
        
        assert len(story) > 0
    
    def test_create_annotated_content(self, sample_output_result):
        """Test annotated content creation."""
        generator = PdfGenerator()
        story = generator._create_annotated_content(sample_output_result)
        
        assert len(story) > 0
    
    def test_annotation_filtering(self, sample_output_result):
        """Test annotation filtering."""
        generator = PdfGenerator()
        
        # Create test annotation
        annotation = Mock()
        annotation.start_position = 0
        annotation.end_position = 10
        
        # Should include by default
        assert generator._should_include_annotation(annotation) is True
    
    def test_annotation_display_creation(self, sample_output_result):
        """Test annotation display creation."""
        generator = PdfGenerator()
        
        annotation = Mock()
        annotation.style = HighlightStyle.HIGH
        annotation.content = "Test comment"
        annotation.related_issue_id = "test-issue"
        
        story = generator._create_annotation_display(annotation, "Test text")
        assert len(story) > 0
    
    @patch('src.models.output_generators.SimpleDocTemplate')
    def test_error_handling(self, mock_doc_template, sample_output_result):
        """Test error handling during generation."""
        mock_doc_template.side_effect = Exception("PDF generation failed")
        
        generator = PdfGenerator()
        
        with pytest.raises(DocumentGeneratorError):
            generator.generate(sample_output_result, "/invalid/path/test.pdf")


class TestDocxGenerator:
    """Test the DOCX generator functionality."""
    
    @pytest.fixture
    def sample_output_result(self):
        """Create a sample output verification result."""
        # Create mock document
        document = Mock(spec=ParsedDocument)
        document.content = "This is a sample document with some content for testing DOCX generation."
        document.document_id = "test-doc-456"
        
        # Create mock issues
        issue1 = Mock(spec=UnifiedIssue)
        issue1.issue_id = "issue-1"
        issue1.issue_type = IssueType.CONSISTENCY_ISSUE
        issue1.severity = IssueSeverity.MEDIUM
        issue1.description = "Sample logical consistency issue"
        issue1.confidence = 0.75
        issue1.location = Mock()
        issue1.location.start_position = 15
        issue1.location.end_position = 30
        issue1.remediation_suggestion = "Review logic flow"
        
        issue_registry = Mock(spec=IssueRegistry)
        issue_registry.issues = [issue1]
        
        # Create mock annotations
        annotation1 = Mock(spec=DocumentAnnotation)
        annotation1.annotation_id = "ann-1"
        annotation1.start_position = 15
        annotation1.end_position = 30
        annotation1.content = "Logic needs clarification"
        annotation1.style = HighlightStyle.MEDIUM
        annotation1.related_issue_id = "issue-1"
        
        layer1 = Mock(spec=AnnotationLayer)
        layer1.layer_id = "layer-1"
        layer1.annotations = [annotation1]
        
        # Create output result
        result = Mock(spec=OutputVerificationResult)
        result.document_id = "test-doc-456"
        result.document = document
        result.verification_status = "verified"
        result.total_issues = 1
        result.overall_confidence = 0.85
        result.generated_at = datetime.now(timezone.utc)
        result.generator_version = "1.0.0"
        result.issue_registry = issue_registry
        result.annotation_layers = [layer1]
        result.critical_issues = []
        result.debate_views = []
        result.generate_summary.return_value = {
            'total_issues': 1,
            'verification_status': 'verified',
            'overall_confidence': 0.85,
            'processing_time_seconds': 2.1,
            'issues_by_severity': {IssueSeverity.MEDIUM: 1}
        }
        
        return result
    
    def test_initialization(self):
        """Test DOCX generator initialization."""
        generator = DocxGenerator()
        assert generator.config is not None
    
    def test_custom_config_initialization(self):
        """Test DOCX generator with custom config."""
        config = OutputGenerationConfig(
            output_format=OutputFormat.DOCX,
            include_appendices=False
        )
        generator = DocxGenerator(config)
        assert generator.config.include_appendices is False
    
    @patch('src.models.output_generators.Document')
    def test_generate_basic_docx(self, mock_document, sample_output_result):
        """Test basic DOCX generation."""
        # Use minimal config to avoid complex table operations
        config = OutputGenerationConfig(
            output_format=OutputFormat.DOCX,
            include_summary=False,
            include_table_of_contents=False,
            include_appendices=False
        )
        
        # Setup mock
        mock_doc = Mock()
        mock_document.return_value = mock_doc
        mock_doc.core_properties = Mock()
        
        # Create a proper mock for styles that supports 'in' operator
        mock_styles = Mock()
        mock_styles.__contains__ = Mock(return_value=False)  # Style doesn't exist initially
        mock_styles.add_style = Mock()
        mock_doc.styles = mock_styles
        
        mock_heading = Mock()
        mock_heading.alignment = None
        mock_doc.add_heading.return_value = mock_heading
        
        mock_paragraph = Mock()
        mock_paragraph.add_run.return_value = Mock()
        mock_doc.add_paragraph.return_value = mock_paragraph
        
        # Create a comprehensive mock for table with rows
        mock_table = Mock()
        mock_rows = []
        for i in range(10):  # Create enough rows for any table
            mock_row = Mock()
            mock_cell = Mock()
            mock_cell.text = ""
            
            # Create mock for paragraphs and runs
            mock_run = Mock()
            mock_run.bold = None
            mock_paragraph = Mock()
            mock_paragraph.runs = [mock_run]
            mock_cell.paragraphs = [mock_paragraph]
            
            mock_row.cells = [mock_cell, mock_cell]  # Two cells per row
            mock_rows.append(mock_row)
        
        mock_table.rows = mock_rows
        mock_table.style = None
        mock_table.add_row.return_value = mock_rows[0] if mock_rows else Mock()  # Return existing row mock
        mock_doc.add_table.return_value = mock_table
        
        mock_doc.add_page_break = Mock()
        mock_doc.save = Mock()
        
        generator = DocxGenerator(config)
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            result_path = generator.generate(sample_output_result, output_path)
            
            assert result_path == output_path
            mock_doc.save.assert_called_once_with(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_highlight_color_mapping(self):
        """Test highlight color mapping for severities."""
        generator = DocxGenerator()
        
        for severity in IssueSeverity:
            color_index = generator._get_highlight_color_index(severity)
            assert color_index is not None
    
    @patch('src.models.output_generators.Document')
    def test_document_properties_setup(self, mock_document, sample_output_result):
        """Test document properties setup."""
        mock_doc = Mock()
        mock_doc.core_properties = Mock()
        
        generator = DocxGenerator()
        generator._setup_document_properties(mock_doc, sample_output_result)
        
        # Verify properties were set
        assert mock_doc.core_properties.title is not None
        assert mock_doc.core_properties.author is not None
    
    def test_annotation_filtering(self, sample_output_result):
        """Test annotation filtering."""
        generator = DocxGenerator()
        
        # Create test annotation
        annotation = Mock()
        annotation.start_position = 0
        annotation.end_position = 10
        
        # Should include by default
        assert generator._should_include_annotation(annotation) is True
    
    def test_find_issue_by_id(self, sample_output_result):
        """Test finding issues by ID."""
        generator = DocxGenerator()
        
        # Should find existing issue
        issue = generator._find_issue_by_id(sample_output_result, "issue-1")
        assert issue is not None
        
        # Should not find non-existent issue
        no_issue = generator._find_issue_by_id(sample_output_result, "non-existent")
        assert no_issue is None
    
    @patch('src.models.output_generators.Document')
    def test_error_handling(self, mock_document, sample_output_result):
        """Test error handling during generation."""
        mock_document.side_effect = Exception("DOCX generation failed")
        
        generator = DocxGenerator()
        
        with pytest.raises(DocumentGeneratorError):
            generator.generate(sample_output_result, "/invalid/path/test.docx")


class TestFactoryFunctions:
    """Test factory functions and convenience methods."""
    
    def test_create_pdf_generator(self):
        """Test creating PDF generator via factory."""
        generator = create_document_generator(OutputFormat.PDF)
        assert isinstance(generator, PdfGenerator)
    
    def test_create_docx_generator(self):
        """Test creating DOCX generator via factory."""
        generator = create_document_generator(OutputFormat.DOCX)
        assert isinstance(generator, DocxGenerator)
    
    def test_create_generator_with_config(self):
        """Test creating generator with custom config."""
        config = OutputGenerationConfig(
            output_format=OutputFormat.PDF,
            include_summary=False
        )
        generator = create_document_generator(OutputFormat.PDF, config)
        assert generator.config.include_summary is False
    
    def test_unsupported_format(self):
        """Test error for unsupported format."""
        with pytest.raises(DocumentGeneratorError):
            create_document_generator("unsupported_format")
    
    @patch('src.models.output_generators.PdfGenerator')
    def test_generate_pdf_report_convenience(self, mock_generator_class):
        """Test PDF report convenience function."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "/path/to/output.pdf"
        mock_generator_class.return_value.__enter__.return_value = mock_generator
        
        output_result = Mock()
        result_path = generate_pdf_report(output_result, "/path/to/output.pdf")
        
        assert result_path == "/path/to/output.pdf"
        mock_generator.generate.assert_called_once_with(output_result, "/path/to/output.pdf")
    
    @patch('src.models.output_generators.DocxGenerator')
    def test_generate_docx_report_convenience(self, mock_generator_class):
        """Test DOCX report convenience function."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "/path/to/output.docx"
        mock_generator_class.return_value.__enter__.return_value = mock_generator
        
        output_result = Mock()
        result_path = generate_docx_report(output_result, "/path/to/output.docx")
        
        assert result_path == "/path/to/output.docx"
        mock_generator.generate.assert_called_once_with(output_result, "/path/to/output.docx")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_content_handling(self):
        """Test handling of empty document content."""
        # Create minimal output result
        document = Mock()
        document.content = ""
        document.document_id = "empty-doc"
        
        result = Mock()
        result.document = document
        result.annotation_layers = []
        result.issue_registry = Mock()
        result.issue_registry.issues = []
        result.critical_issues = []
        result.debate_views = []
        result.verification_status = "verified"
        result.total_issues = 0
        result.overall_confidence = 1.0
        result.generated_at = datetime.now(timezone.utc)
        result.generator_version = "1.0.0"
        result.generate_summary.return_value = {
            'total_issues': 0,
            'verification_status': 'verified',
            'overall_confidence': 1.0,
            'processing_time_seconds': 0.5,
            'issues_by_severity': {}
        }
        
        # Should handle empty content gracefully
        pdf_generator = PdfGenerator()
        docx_generator = DocxGenerator()
        
        pdf_story = pdf_generator._create_annotated_content(result)
        assert isinstance(pdf_story, list)
    
    def test_no_annotations_handling(self):
        """Test handling when no annotations are present."""
        # Create result with content but no annotations
        document = Mock()
        document.content = "Some content without annotations"
        
        result = Mock()
        result.document = document
        result.annotation_layers = []  # No annotation layers
        
        pdf_generator = PdfGenerator()
        story = pdf_generator._create_annotated_content(result)
        
        # Should still create content
        assert len(story) > 0
    
    def test_overlapping_annotations(self):
        """Test handling of overlapping annotations."""
        # Create document with overlapping annotations
        document = Mock()
        document.content = "This is test content for overlapping annotations"
        
        # Create overlapping annotations
        ann1 = Mock()
        ann1.start_position = 5
        ann1.end_position = 15
        ann1.content = "First annotation"
        ann1.style = HighlightStyle.LOW
        ann1.related_issue_id = "issue-1"
        
        ann2 = Mock()
        ann2.start_position = 10
        ann2.end_position = 20
        ann2.content = "Overlapping annotation"
        ann2.style = HighlightStyle.MEDIUM
        ann2.related_issue_id = "issue-2"
        
        layer = Mock()
        layer.annotations = [ann1, ann2]
        
        result = Mock()
        result.document = document
        result.annotation_layers = [layer]
        
        pdf_generator = PdfGenerator()
        
        # Should handle overlapping annotations without crashing
        story = pdf_generator._create_annotated_content(result)
        assert isinstance(story, list)
    
    def test_missing_issue_reference(self):
        """Test handling when annotation references missing issue."""
        output_result = Mock()
        output_result.issue_registry = Mock()
        output_result.issue_registry.issues = []  # No issues
        
        docx_generator = DocxGenerator()
        
        # Should return None for missing issue
        issue = docx_generator._find_issue_by_id(output_result, "non-existent-issue")
        assert issue is None
    
    def test_cleanup_functionality(self):
        """Test cleanup of temporary files."""
        generator = BaseDocumentGenerator()
        
        # Add some fake temp files
        generator.temp_files = ["/fake/file1.tmp", "/fake/file2.tmp"]
        
        # Cleanup should not crash even if files don't exist
        generator.cleanup()
        
        # Should clear the list
        assert generator.temp_files == [] 