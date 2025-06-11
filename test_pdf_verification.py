#!/usr/bin/env python3
"""
Test script to verify a real PDF document using the verification pipeline
"""

import sys
import os
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pdf_verification():
    """Test the verification pipeline with a real PDF document."""
    print("🚀 Starting PDF verification test")
    
    try:
        # Import required modules
        from src.verification.config.chain_loader import ChainConfigLoader
        from src.verification.workers.verification_worker import VerificationWorker
        from src.models.verification import VerificationTask
        from src.document_ingestion import DocumentIngestionService
        from src.models.document import DocumentFormat
        
        print("✅ Successfully imported verification modules")
        
        # Find PDF file in test_docs
        test_docs_dir = project_root / "test_docs"
        pdf_files = list(test_docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in test_docs directory")
            return False
            
        pdf_file = pdf_files[0]
        print(f"📄 Found PDF file: {pdf_file.name}")
        print(f"   Size: {pdf_file.stat().st_size / 1024:.1f} KB")
        
        # Initialize document ingestion service
        print("\n🔧 Initializing document ingestion service...")
        doc_service = DocumentIngestionService()
        
        # Process the PDF document
        print(f"📖 Processing PDF: {pdf_file}")
        
        # Parse the PDF document using the document ingestion service
        parsed_doc = doc_service.parse_document(str(pdf_file))
        
        print(f"✅ Successfully processed PDF")
        print(f"   Content length: {len(parsed_doc.content)} characters")
        print(f"   Word count: {parsed_doc.metadata.word_count}")
        print(f"   Language: {parsed_doc.metadata.language}")
        
        # Load chain configuration
        print("\n📂 Loading chain configurations...")
        chain_loader = ChainConfigLoader()
        all_chains = chain_loader.load_all_chains()
        
        # Use comprehensive chain
        comprehensive_chain = all_chains.get('comprehensive')
        if not comprehensive_chain:
            print("❌ Comprehensive chain not found")
            return False
            
        print(f"🎯 Using chain: {comprehensive_chain.name}")
        print(f"   Passes: {len(comprehensive_chain.passes)}")
        
        # Initialize verification worker
        print("\n🔧 Initializing verification worker...")
        worker = VerificationWorker()
        
        print(f"✅ Worker initialized with {len(worker.pass_registry)} pass implementations")
        
        # Create verification task with document content
        print("\n🚀 Creating verification task...")
        task = VerificationTask(
            document_id=f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            document_content=parsed_doc.content,  # Pass content directly
            chain_config=comprehensive_chain
        )
        
        print(f"✅ Created verification task: {task.task_id}")
        
        # Run verification pipeline
        print("\n🔄 Running verification pipeline...")
        print("   This may take a few minutes with a real PDF document...")
        
        async def run_verification():
            return await worker.execute_verification_chain(task)
        
        # Execute the verification
        chain_result = asyncio.run(run_verification())
        
        print(f"\n✅ Verification completed!")
        print(f"   Status: {chain_result.status}")
        print(f"   Execution time: {chain_result.total_execution_time_seconds:.2f}s")
        print(f"   Overall confidence: {chain_result.overall_confidence:.3f}")
        
        # Display results summary
        print(f"\n📊 Results Summary:")
        for pass_result in chain_result.pass_results:
            status_emoji = "✅" if pass_result.status.value == "completed" else "❌"
            confidence_str = f"confidence: {pass_result.confidence_score:.3f}" if pass_result.confidence_score else "confidence: None"
            print(f"   {status_emoji} {pass_result.pass_id}: {pass_result.status.value} ({confidence_str})")
        
        # Show some detailed results if available
        print(f"\n🔍 Detailed Analysis:")
        for pass_result in chain_result.pass_results:
            if pass_result.result_data and pass_result.pass_type.value == "claim_extraction":
                claims = pass_result.result_data.get('claims', [])
                if claims:
                    print(f"   📋 Found {len(claims)} claims:")
                    for i, claim in enumerate(claims[:3], 1):  # Show first 3 claims
                        claim_text = claim.get('text', 'No text')[:100]
                        confidence = claim.get('confidence', 0)
                        print(f"      {i}. {claim_text}... (confidence: {confidence:.2f})")
                    if len(claims) > 3:
                        print(f"      ... and {len(claims) - 3} more claims")
        
        print(f"\n🎉 PDF verification test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during PDF verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_verification()
    sys.exit(0 if success else 1) 