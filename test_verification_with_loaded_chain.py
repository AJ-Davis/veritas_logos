#!/usr/bin/env python3
"""
Test script to verify the complete verification pipeline with loaded chain configurations
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

def create_test_document():
    """Create a test document for verification."""
    test_content = """
# Climate Change Analysis

Climate change is a significant global challenge. According to NASA, global temperatures have risen by approximately 1.1°C since the late 19th century.

## Key Facts:
- CO2 levels have increased by 40% since pre-industrial times [1]
- Arctic sea ice is declining at a rate of 13% per decade [2]
- Sea levels are rising at 3.3 mm per year [3]

## Scientific Consensus:
Multiple studies show that 97% of climate scientists agree that human activities are the primary cause of recent climate change.

## References:
[1] NOAA Climate Data
[2] NASA Arctic Sea Ice Data
[3] IPCC Reports
"""
    return test_content

async def test_verification_pipeline():
    """Test the complete verification pipeline."""
    print("🔧 Testing verification pipeline with loaded chain configuration...")
    
    try:
        # Import verification components
        from src.verification.config.chain_loader import ChainConfigLoader
        from src.verification.workers.verification_worker import VerificationWorker
        from src.models.verification import VerificationTask, VerificationContext
        from src.models.document import ParsedDocument, DocumentFormat
        import uuid
        
        print("✅ Successfully imported verification modules")
        
        # Load chain configurations
        print("\n📂 Loading chain configurations...")
        config_loader = ChainConfigLoader()
        chains = config_loader.load_all_chains()
        print(f"✅ Loaded {len(chains)} chain configurations")
        
        # Get the comprehensive chain
        if 'comprehensive' not in chains:
            print("❌ Comprehensive chain not found")
            return False
            
        comprehensive_chain = chains['comprehensive']
        print(f"🎯 Using chain: {comprehensive_chain.name}")
        
        # Initialize verification worker
        print("\n🔧 Initializing verification worker...")
        worker = VerificationWorker()
        print(f"✅ Worker initialized with {len(worker.pass_registry)} pass implementations")
        
        # Create test document
        print("\n📄 Creating test document...")
        test_content = create_test_document()
        document_id = str(uuid.uuid4())
        
        # Create document metadata
        from src.models.document import DocumentMetadata, ExtractionMethod
        
        doc_metadata = DocumentMetadata(
            filename="test_climate_change.md",
            file_size_bytes=len(test_content.encode('utf-8')),
            format=DocumentFormat.MARKDOWN,
            extraction_method=ExtractionMethod.DIRECT,
            processed_at=datetime.now(),
            processing_time_seconds=0.1,
            word_count=len(test_content.split()),
            character_count=len(test_content),
            language="en",
            encoding="utf-8"
        )
        
        # Create parsed document
        parsed_doc = ParsedDocument(
            content=test_content,
            sections=[],
            metadata=doc_metadata,
            raw_data={"document_id": document_id},
            errors=[],
            warnings=[]
        )
        
        print(f"✅ Created test document: {document_id}")
        print(f"Content length: {len(test_content)} characters")
        
        # Create verification context
        print("\n🔄 Creating verification context...")
        
        # Convert metadata to dict for context
        metadata_dict = {
            "filename": parsed_doc.metadata.filename,
            "format": parsed_doc.metadata.format.value,
            "word_count": parsed_doc.metadata.word_count,
            "character_count": parsed_doc.metadata.character_count,
            "language": parsed_doc.metadata.language,
            "document_id": document_id
        }
        
        context = VerificationContext(
            document_id=document_id,
            document_content=test_content,
            document_metadata=metadata_dict,
            chain_config=comprehensive_chain,
            previous_results=[],
            global_context={}
        )
        
        print(f"✅ Created verification context")
        
        # Test individual passes
        print(f"\n🧪 Testing individual verification passes...")
        execution_order = comprehensive_chain.get_execution_order()
        print(f"📋 Execution order: {len(execution_order)} passes")
        
        results = []
        for i, pass_config in enumerate(execution_order, 1):
            print(f"\n🔧 Testing pass {i}/{len(execution_order)}: {pass_config.name}")
            print(f"   Type: {pass_config.pass_type}")
            print(f"   Dependencies: {[dep.value for dep in pass_config.depends_on]}")
            
            # Check if dependencies are satisfied
            if context.are_dependencies_satisfied(pass_config):
                print(f"   ✅ Dependencies satisfied")
                
                # Get pass implementation
                if pass_config.pass_type in worker.pass_registry:
                    pass_impl = worker.pass_registry[pass_config.pass_type]
                    print(f"   ✅ Pass implementation found: {type(pass_impl).__name__}")
                    
                    try:
                        # Execute the pass
                        print(f"   🔄 Executing pass...")
                        result = await pass_impl.execute(context, pass_config)
                        
                        print(f"   ✅ Pass completed: {result.status}")
                        print(f"   Confidence: {result.confidence_score}")
                        if result.error_message:
                            print(f"   ⚠️  Error: {result.error_message}")
                        
                        # Add result to context for next passes
                        context.previous_results.append(result)
                        results.append(result)
                        
                    except Exception as e:
                        print(f"   ❌ Pass execution failed: {str(e)}")
                        print(f"   Error type: {type(e).__name__}")
                        # Continue with next pass
                        
                else:
                    print(f"   ⚠️  Pass implementation not found")
                    
            else:
                print(f"   ⏸️  Dependencies not satisfied, skipping")
        
        # Test complete pipeline execution
        print(f"\n🚀 Testing complete pipeline execution...")
        try:
            task = VerificationTask(
                document_id=document_id,
                document_content=test_content,  # Pass content directly
                chain_config=comprehensive_chain
            )
            
            print(f"✅ Created verification task: {task.task_id}")
            
            # Run the complete pipeline
            print(f"🔄 Running complete verification pipeline...")
            chain_result = await worker.execute_verification_chain(task)
            
            print(f"✅ Pipeline completed!")
            print(f"   Status: {chain_result.status}")
            print(f"   Execution time: {chain_result.total_execution_time_seconds:.2f}s")
            print(f"   Pass results: {len(chain_result.pass_results)}")
            print(f"   Overall confidence: {chain_result.overall_confidence}")
            
            # Show results summary
            if chain_result.pass_results:
                print(f"\n📊 Results Summary:")
                for result in chain_result.pass_results:
                    status_icon = "✅" if result.status.value == "completed" else "❌"
                    print(f"   {status_icon} {result.pass_id}: {result.status} (confidence: {result.confidence_score})")
            
            if chain_result.errors:
                print(f"\n❌ Errors encountered:")
                for error in chain_result.errors:
                    print(f"   - {error}")
                    
            if chain_result.warnings:
                print(f"\n⚠️  Warnings:")
                for warning in chain_result.warnings:
                    print(f"   - {warning}")
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ Verification pipeline test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during verification test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Starting verification pipeline test with loaded chain configuration\n")
    success = await test_verification_pipeline()
    
    if success:
        print("\n🎉 All tests completed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 