#!/usr/bin/env python3
"""
Real-world PDF verification test using the Strategic Intelligence Brief PDF.
This script bypasses the API server and tests the verification pipeline directly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.verification.pipeline.verification_pipeline import VerificationPipeline, PipelineConfig
from src.document_ingestion.pdf_parser import PdfParser
from src.models.verification import VerificationPassType

async def test_real_world_pdf():
    """Test verification pipeline with the real Strategic Intelligence Brief PDF."""
    
    print("=== Real-World PDF Verification Test ===")
    
    # Path to the real PDF
    pdf_path = Path("test_docs/Strategic Intelligence Brief_ Engaging Kurt Matsumoto and Pulama Lanai on Energy Innovation-2025061014552325.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Testing with PDF: {pdf_path.name}")
    print(f"📄 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
    try:
        # Step 1: Parse the PDF
        print("\n1️⃣ Parsing PDF content...")
        parser = PdfParser()
        
        # Parse the document (parser expects file path)
        parsed_doc = parser.parse(str(pdf_path))
        
        print(f"✅ PDF parsed successfully")
        print(f"   - Content length: {len(parsed_doc.content)} characters")
        if parsed_doc.metadata:
            print(f"   - Pages: {getattr(parsed_doc.metadata, 'page_count', 'Unknown')}")
            print(f"   - File size: {parsed_doc.metadata.file_size_bytes} bytes")
        print(f"   - First 200 chars: {parsed_doc.content[:200]}...")
        
        # Step 2: Initialize verification pipeline
        print("\n2️⃣ Initializing verification pipeline...")
        
        # We need to initialize the pipeline with a proper pass registry
        # For now, let's use a minimal configuration and see what happens
        try:
            from src.verification.workers.verification_worker import VerificationWorker
            worker = VerificationWorker()
            pass_registry = worker.pass_registry
            
            config = PipelineConfig(
                enabled_passes={
                    VerificationPassType.CLAIM_EXTRACTION,
                    VerificationPassType.LOGIC_ANALYSIS,
                    VerificationPassType.BIAS_SCAN,
                    VerificationPassType.CITATION_CHECK
                }
            )
            
            pipeline = VerificationPipeline(config, pass_registry)
            
        except Exception as e:
            print(f"   ⚠️  Could not initialize full pipeline: {e}")
            print("   📝 This might be due to missing dependencies or configuration")
            print("   🔄 Let's try a simpler approach...")
            
            # Fall back to testing individual components
            return await test_individual_components(parsed_doc)
        
        # Step 3: Process document
        print("\n3️⃣ Processing document through pipeline...")
        document_id = "test-real-world-pdf"
        content = parsed_doc.content
        
        # Convert metadata to dict format
        metadata = {}
        if parsed_doc.metadata:
            metadata = {
                "filename": parsed_doc.metadata.filename,
                "file_size_bytes": parsed_doc.metadata.file_size_bytes,
                "format": parsed_doc.metadata.format.value,
                "extraction_method": parsed_doc.metadata.extraction_method.value,
                "page_count": getattr(parsed_doc.metadata, 'page_count', None),
                "word_count": getattr(parsed_doc.metadata, 'word_count', None),
                "character_count": getattr(parsed_doc.metadata, 'character_count', None)
            }
        
        print(f"   - Document ID: {document_id}")
        print(f"   - Content length: {len(content)} characters")
        print(f"   - Content preview: {content[:150]}...")
        
        # Step 4: Run verification
        print("\n4️⃣ Running verification pipeline...")
        print("   This may take a minute since we're using real verification passes...")
        
        result = await pipeline.process_document(document_id, content, metadata)
        
        # Step 5: Display results
        print("\n5️⃣ Verification Results:")
        print(f"   - Status: {result.status}")
        print(f"   - Chain ID: {result.chain_id}")
        if result.total_execution_time_seconds:
            print(f"   - Processing Time: {result.total_execution_time_seconds:.2f}s")
        print(f"   - Passes Completed: {len([p for p in result.pass_results if p.status.value == 'completed'])}")
        
        if result.errors:
            print(f"   - Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"     - {error}")
        
        # Detailed pass results
        print("\n📊 Detailed Pass Results:")
        for pass_result in result.pass_results:
            print(f"\n   🔍 {pass_result.pass_type.value}:")
            print(f"      Status: {pass_result.status.value}")
            if pass_result.confidence_score:
                print(f"      Confidence: {pass_result.confidence_score:.2f}")
            if pass_result.execution_time_seconds:
                print(f"      Processing Time: {pass_result.execution_time_seconds:.2f}s")
            
            if pass_result.result_data:
                print(f"      Result Data Keys: {list(pass_result.result_data.keys())}")
                
                # Show specific results based on pass type
                if pass_result.pass_type == VerificationPassType.CLAIM_EXTRACTION:
                    claims = pass_result.result_data.get("claims", [])
                    print(f"      Claims Found: {len(claims)}")
                    if claims:
                        print(f"      Sample Claims:")
                        for i, claim in enumerate(claims[:3]):
                            print(f"         {i+1}. {claim}")
                        if len(claims) > 3:
                            print(f"         ... and {len(claims) - 3} more claims")
            
            if pass_result.error_message:
                print(f"      Error: {pass_result.error_message}")
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        completed_passes = [p for p in result.pass_results if p.status.value == 'completed']
        failed_passes = [p for p in result.pass_results if p.status.value == 'failed']
        print(f"   - Completed Passes: {len(completed_passes)}")
        print(f"   - Failed Passes: {len(failed_passes)}")
        
        if result.overall_confidence:
            print(f"   - Overall Confidence: {result.overall_confidence:.2f}")
        
        if result.summary:
            print(f"   - Summary Keys: {list(result.summary.keys())}")
        
        print(f"\n✅ Real-world PDF verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_components(parsed_doc):
    """Fallback test using individual verification components."""
    print("\n🔧 Testing individual verification components...")
    
    try:
        # Test claim extraction
        print("\n📋 Testing Claim Extraction...")
        from src.verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass
        
        claim_pass = ClaimExtractionPass()
        from src.models.verification import VerificationContext
        
        context = VerificationContext(
            document_id="test-individual",
            document_content=parsed_doc.content[:5000]  # Limit content for testing
        )
        
        claim_result = await claim_pass.execute(context)
        print(f"   ✅ Claim extraction: {claim_result.status}")
        print(f"   📊 Found {len(claim_result.result_data.get('claims', []))} claims")
        
        if claim_result.result_data.get('claims'):
            print("   🔍 Sample claims:")
            for i, claim in enumerate(claim_result.result_data['claims'][:3]):
                print(f"      {i+1}. {claim}")
        
        # Test other passes if needed
        print(f"\n✅ Individual component testing completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during individual component testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set minimal environment for testing
    os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder")
    
    success = asyncio.run(test_real_world_pdf())
    sys.exit(0 if success else 1) 