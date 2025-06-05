#!/usr/bin/env python3
"""
Example script demonstrating the Document Ingestion Service.

This script shows how to use the VeritasLogos document ingestion system
to parse various document formats and extract structured content.
"""

import sys
import os
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_ingestion import document_service
from models.document import DocumentFormat


def demonstrate_parsing():
    """Demonstrate basic document parsing functionality."""
    
    print("🔍 VeritasLogos Document Ingestion Service Demo")
    print("=" * 50)
    
    # Show supported formats
    print(f"\n📄 Supported formats: {', '.join(document_service.get_supported_formats())}")
    
    # Get parser statistics
    stats = document_service.get_parser_statistics()
    print(f"📊 Available parsers: {stats['total_parsers']}")
    print(f"🔧 Parser types: {', '.join(stats['parser_types'])}")
    
    # Create test documents
    test_documents = create_test_documents()
    
    print(f"\n🧪 Testing with {len(test_documents)} sample documents...")
    
    for doc_path in test_documents:
        print(f"\n" + "─" * 40)
        print(f"📂 Processing: {doc_path}")
        
        # Validate document first
        validation = document_service.validate_document(doc_path)
        print(f"✅ Valid: {validation['is_valid']}")
        
        if not validation['is_valid']:
            print(f"❌ Error: {validation['error']}")
            continue
        
        # Get document info
        info = document_service.get_document_info(doc_path)
        print(f"📋 Info: {info['file_size_bytes']} bytes, {info['document_format']} format")
        
        # Parse the document
        try:
            result = document_service.parse_document(doc_path)
            
            if result.is_valid:
                print(f"✅ Parsed successfully!")
                print(f"📝 Content length: {len(result.content)} characters")
                print(f"📄 Sections: {len(result.sections)}")
                print(f"🔢 Token count: {result.metadata.token_count}")
                print(f"⏱️  Processing time: {result.metadata.processing_time_seconds:.3f}s")
                
                # Show first few sections
                if result.sections:
                    print("\n📑 Sample sections:")
                    for i, section in enumerate(result.sections[:3]):
                        content_preview = section.content[:100] + "..." if len(section.content) > 100 else section.content
                        print(f"  {i+1}. [{section.section_type}] {content_preview}")
                
                # Show content preview
                content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
                print(f"\n📃 Content preview:\n{content_preview}")
                
            else:
                print(f"❌ Parsing failed: {result.errors}")
                
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")
    
    # Clean up test files
    cleanup_test_documents(test_documents)
    
    print(f"\n" + "=" * 50)
    print("✨ Demo completed successfully!")


def create_test_documents():
    """Create sample documents for testing."""
    test_docs = []
    
    # Create a simple text file
    txt_content = """Document Ingestion Test

This is a test document for the VeritasLogos document ingestion system.

It contains multiple paragraphs to test section detection.

Features:
- Multi-format support
- Token counting
- Metadata extraction
- Error handling

End of document."""
    
    txt_path = "test_document.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    test_docs.append(txt_path)
    
    # Create a Markdown file
    md_content = """# Document Ingestion Test

This is a **test document** for the VeritasLogos document ingestion system.

## Features

The system supports:

1. **Multi-format parsing**
   - PDF files (with OCR fallback)
   - DOCX documents  
   - Markdown files
   - Plain text files

2. **Advanced processing**
   - Token counting with tiktoken
   - File size validation (up to 150MB)
   - Section detection and classification
   - Metadata extraction

## Code Example

```python
from document_ingestion import document_service

# Parse a document
result = document_service.parse_document("sample.pdf")
if result.is_valid:
    print(f"Extracted {len(result.content)} characters")
```

## Conclusion

The document ingestion system provides a unified interface for processing various document formats with comprehensive validation and error handling.
"""
    
    md_path = "test_document.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    test_docs.append(md_path)
    
    return test_docs


def cleanup_test_documents(test_docs):
    """Clean up test documents."""
    for doc_path in test_docs:
        try:
            os.unlink(doc_path)
        except OSError:
            pass


def main():
    """Main function."""
    try:
        demonstrate_parsing()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()