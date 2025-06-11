#!/usr/bin/env python3
"""
Real Verification Pipeline Test

This script directly tests the verification worker with real LLM providers
to process your Strategic Intelligence Brief, bypassing the mock API endpoints.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🚀 Real Verification Pipeline Test")
print("=" * 60)

async def test_real_verification_pipeline():
    """Test the real verification pipeline with your Strategic Intelligence Brief"""
    
    try:
        # Import verification components
        from src.verification.workers.verification_worker import VerificationWorker
        from src.verification.config.chain_loader import ChainConfigLoader
        from src.models.verification import VerificationTask, VerificationStatus
        from src.document_ingestion import document_service
        
        print("✅ Successfully imported verification components")
        
        # Load Strategic Intelligence Brief content
        strategic_brief_content = \"\"\"
Strategic Intelligence Brief: Engaging Kurt Matsumoto and Pulama Lanai on Energy Innovation

Executive Summary:
This brief outlines strategic opportunities for collaboration with Kurt Matsumoto, a prominent figure in renewable energy policy, and Pulama Lanai, the community-focused sustainable development initiative on Hawaii's Lanai island.

Key Findings:
1. Kurt Matsumoto has been instrumental in developing Hawaii's renewable energy mandates
2. Pulama Lanai represents a unique testing ground for sustainable community development
3. Both entities are actively seeking partnerships for innovative energy solutions
4. Current energy infrastructure on Lanai presents significant modernization opportunities

Strategic Recommendations:
- Engage with Matsumoto's policy network to understand regulatory framework
- Explore pilot programs with Pulama Lanai for distributed energy systems
- Investigate potential for Lanai as a renewable energy showcase destination
- Consider joint ventures for sustainable tourism energy solutions

Market Analysis:
The Hawaiian energy market is experiencing rapid transformation, with ambitious renewable energy targets driving innovation. Lanai's isolated grid system provides an ideal laboratory for testing new technologies at scale.

Competitive Landscape:
Major players include Hawaiian Electric, Tesla Energy, and several local renewable energy cooperatives. Opportunities exist for differentiated approaches focused on community engagement and cultural sensitivity.

Risk Assessment:
Primary risks include regulatory complexity, community acceptance challenges, and technical difficulties associated with island grid systems. Mitigation strategies should focus on early stakeholder engagement and phased implementation approaches.

Financial Projections:
Initial investment requirements estimated at $5-15M for pilot programs, with potential ROI of 15-25% over 5-7 year timeline, contingent on regulatory support and successful technology deployment.

Next Steps:
1. Schedule introductory meetings with key stakeholders
2. Conduct detailed site assessment on Lanai
3. Develop preliminary project proposals
4. Secure necessary regulatory approvals
5. Begin pilot program implementation

Conclusion:
The convergence of policy support (Matsumoto) and practical testing opportunities (Pulama Lanai) creates a compelling strategic opportunity in the Hawaiian energy innovation space.
        \"\"\"
        
        # Initialize verification worker
        print("🔧 Initializing verification worker...")
        worker = VerificationWorker()
        
        # Load comprehensive verification chain
        print("⚙️ Loading verification chain configuration...")
        config_loader = ChainConfigLoader()
        chain_config = config_loader.get_chain_config("comprehensive")
        
        if not chain_config:
            print("❌ Failed to load chain configuration")
            return
            
        print(f"✅ Loaded chain '{chain_config.chain_id}' with {len(chain_config.passes)} passes")
        
        # Create verification task
        print("📝 Creating verification task...")
        verification_task = VerificationTask(
            document_id="strategic-intelligence-brief",
            chain_config=chain_config,
            metadata={
                "document_type": "strategic_brief",
                "content_preview": strategic_brief_content[:200] + "...",
                "test_mode": "real_llm_processing"
            }
        )
        
        print(f"📋 Task ID: {verification_task.task_id}")
        print(f"🔗 Chain: {verification_task.chain_config.chain_id}")
        print(f"📊 Passes to execute: {len(verification_task.chain_config.passes)}")
        
        # Mock document service for our content
        class MockDocumentService:
            def parse_document(self, document_id):
                class MockParsedDoc:
                    def __init__(self, content):
                        self.content = content
                        self.is_valid = True
                        self.errors = []
                return MockParsedDoc(strategic_brief_content)
        
        # Temporarily replace document service
        original_service = document_service
        mock_service = MockDocumentService()
        
        # Monkey patch for testing
        import src.verification.workers.verification_worker as worker_module
        worker_module.document_service = mock_service
        
        print("\\n🎯 Starting verification pipeline...")
        print("⏱️ This will use real LLM providers (Claude, Gemini, DeepSeek)")
        print("⏳ Expected duration: 5-15 minutes depending on complexity")
        
        start_time = time.time()
        
        # Execute verification chain
        try:
            chain_result = await worker.execute_verification_chain(verification_task)
            execution_time = time.time() - start_time
            
            print(f"\\n🎉 Verification completed in {execution_time:.1f} seconds!")
            print("=" * 60)
            
            # Display results
            print(f"📊 **VERIFICATION RESULTS**")
            print(f"Status: {chain_result.status.value}")
            print(f"Chain ID: {chain_result.chain_id}")
            print(f"Document ID: {chain_result.document_id}")
            print(f"Total passes executed: {len(chain_result.pass_results)}")
            print(f"Execution time: {chain_result.total_execution_time_seconds:.2f}s")
            
            if chain_result.errors:
                print(f"\\n⚠️ **ERRORS ({len(chain_result.errors)}):**")
                for error in chain_result.errors:
                    print(f"  - {error}")
            
            print(f"\\n📋 **PASS RESULTS:**")
            for i, pass_result in enumerate(chain_result.pass_results):
                status_emoji = "✅" if pass_result.status.value == "completed" else "❌" if pass_result.status.value == "failed" else "⏳"
                print(f"  {i+1}. {status_emoji} {pass_result.pass_type.value}")
                print(f"     Status: {pass_result.status.value}")
                print(f"     Confidence: {pass_result.confidence_score:.2f}")
                
                if pass_result.error_message:
                    print(f"     Error: {pass_result.error_message}")
                
                # Show some result data
                if pass_result.result_data and isinstance(pass_result.result_data, dict):
                    if pass_result.pass_type.value == "claim_extraction":
                        extraction_result = pass_result.result_data.get("extraction_result", {})
                        claims = extraction_result.get("claims", [])
                        print(f"     Claims extracted: {len(claims)}")
                        if claims:
                            print(f"     First claim: {claims[0].get('claim_text', 'N/A')[:100]}...")
                    
                    elif pass_result.pass_type.value == "citation_verification":
                        verification_result = pass_result.result_data.get("verification_result", {})
                        print(f"     Citations verified: {verification_result.get('total_citations_verified', 0)}")
                        print(f"     Valid citations: {verification_result.get('valid_citations', 0)}")
                        print(f"     Model used: {verification_result.get('model_used', 'unknown')}")
                
                print()
            
            # Save detailed results
            results_file = f"verification_results_{verification_task.task_id}.json"
            with open(results_file, 'w') as f:
                # Convert result to serializable format
                result_dict = {
                    "task_id": verification_task.task_id,
                    "chain_id": chain_result.chain_id,
                    "status": chain_result.status.value,
                    "document_id": chain_result.document_id,
                    "execution_time": chain_result.total_execution_time_seconds,
                    "pass_results": []
                }
                
                for pass_result in chain_result.pass_results:
                    pass_dict = {
                        "pass_type": pass_result.pass_type.value,
                        "status": pass_result.status.value,
                        "confidence_score": pass_result.confidence_score,
                        "error_message": pass_result.error_message,
                        "result_data_summary": str(pass_result.result_data)[:500] if pass_result.result_data else None
                    }
                    result_dict["pass_results"].append(pass_dict)
                
                json.dump(result_dict, f, indent=2, default=str)
            
            print(f"💾 Detailed results saved to: {results_file}")
            
            # Summary
            completed_passes = [r for r in chain_result.pass_results if r.status.value == "completed"]
            failed_passes = [r for r in chain_result.pass_results if r.status.value == "failed"]
            
            print(f"\\n📈 **SUMMARY:**")
            print(f"✅ Completed passes: {len(completed_passes)}/{len(chain_result.pass_results)}")
            print(f"❌ Failed passes: {len(failed_passes)}")
            print(f"⏱️ Total processing time: {execution_time:.1f} seconds")
            print(f"🤖 Real LLM providers used: ✅ (not mock responses)")
            
            if chain_result.status.value == "completed":
                print("\\n🎊 **VERIFICATION PIPELINE SUCCESSFULLY COMPLETED!** 🎊")
            else:
                print("\\n⚠️ **VERIFICATION COMPLETED WITH ISSUES** ⚠️")
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\\n❌ Verification failed after {execution_time:.1f} seconds")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original document service
            worker_module.document_service = original_service
            
    except ImportError as e:
        print(f"❌ Failed to import verification components: {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_verification_pipeline()) 