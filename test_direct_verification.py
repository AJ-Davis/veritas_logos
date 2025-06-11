#!/usr/bin/env python3
"""
Direct Verification Pipeline Test

This script directly tests the verification worker with real LLM providers
to process your Strategic Intelligence Brief, bypassing the mock API endpoints.
"""

import asyncio
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🚀 Direct Verification Pipeline Test")
print("=" * 60)

async def test_direct_verification():
    """Test the verification pipeline directly with Strategic Intelligence Brief"""
    
    try:
        # Import verification components
        from src.verification.workers.verification_worker import VerificationWorker
        from src.verification.config.chain_loader import ChainConfigLoader
        from src.models.verification import VerificationTask
        
        print("✅ Successfully imported verification components")
        
        # Strategic Intelligence Brief content
        brief_content = """Strategic Intelligence Brief: Engaging Kurt Matsumoto and Pulama Lanai on Energy Innovation

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

Financial Projections:
Initial investment requirements estimated at $5-15M for pilot programs, with potential ROI of 15-25% over 5-7 year timeline, contingent on regulatory support and successful technology deployment.

Conclusion:
The convergence of policy support (Matsumoto) and practical testing opportunities (Pulama Lanai) creates a compelling strategic opportunity in the Hawaiian energy innovation space."""
        
        # Initialize worker
        print("🔧 Initializing verification worker...")
        worker = VerificationWorker()
        
        # Load chain config
        print("⚙️ Loading verification chain...")
        config_loader = ChainConfigLoader("src/verification/config/chains")
        
        # Load all chains first
        all_chains = config_loader.load_all_chains()
        print(f"📋 Available chains: {list(all_chains.keys())}")
        
        # Get comprehensive chain
        chain_config = config_loader.get_chain_config("comprehensive")
        
        if not chain_config:
            print("❌ Failed to load comprehensive chain configuration")
            print("Available chains:", list(all_chains.keys()))
            return
            
        print(f"✅ Loaded chain with {len(chain_config.passes)} passes")
        
        # Create task
        verification_task = VerificationTask(
            document_id="strategic-brief",
            chain_config=chain_config,
            metadata={"test_mode": "direct"}
        )
        
        print(f"📋 Task ID: {verification_task.task_id}")
        
        # Mock document service
        class MockDocService:
            def parse_document(self, doc_id):
                class MockDoc:
                    def __init__(self):
                        self.content = brief_content
                        self.is_valid = True
                        self.errors = []
                return MockDoc()
        
        # Replace document service temporarily
        import src.verification.workers.verification_worker as worker_mod
        original_service = worker_mod.document_service
        worker_mod.document_service = MockDocService()
        
        print("\n🎯 Starting verification with REAL LLM providers...")
        print("⏱️ This may take 5-15 minutes...")
        
        start_time = time.time()
        
        try:
            # Execute verification
            result = await worker.execute_verification_chain(verification_task)
            duration = time.time() - start_time
            
            print("\n🎉 Verification completed in {duration:.1f} seconds!")
            print("=" * 60)
            
            # Show results
            print(f"📊 **RESULTS**")
            print(f"Status: {result.status.value}")
            print(f"Passes executed: {len(result.pass_results)}")
            print(f"Execution time: {result.total_execution_time_seconds:.2f}s")
            
            if result.errors:
                print("\n⚠️ **ERRORS:**")
                for error in result.errors:
                    print(f"  - {error}")
            
            print("\n📋 **PASS RESULTS:**")
            for i, pass_result in enumerate(result.pass_results):
                emoji = "✅" if pass_result.status.value == "completed" else "❌"
                print(f"  {i+1}. {emoji} {pass_result.pass_type.value}")
                print(f"     Status: {pass_result.status.value}")
                print(f"     Confidence: {pass_result.confidence_score:.2f}")
                
                if pass_result.error_message:
                    print(f"     Error: {pass_result.error_message}")
                print()
            
            # Save results
            results_file = f"verification_results_{verification_task.task_id[:8]}.json"
            result_data = {
                "task_id": verification_task.task_id,
                "status": result.status.value,
                "execution_time": result.total_execution_time_seconds,
                "passes": [
                    {
                        "type": p.pass_type.value,
                        "status": p.status.value,
                        "confidence": p.confidence_score
                    }
                    for p in result.pass_results
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"💾 Results saved to: {results_file}")
            
            completed = len([r for r in result.pass_results if r.status.value == "completed"])
            failed = len([r for r in result.pass_results if r.status.value == "failed"])
            
            print("\n📈 **SUMMARY:**")
            print(f"✅ Completed: {completed}/{len(result.pass_results)}")
            print(f"❌ Failed: {failed}")
            print(f"⏱️ Duration: {duration:.1f}s")
            print(f"🤖 Real LLMs: ✅")
            
            if result.status.value == "completed":
                print("\n🎊 **SUCCESS!** Real LLM verification completed!")
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ Verification failed after {duration:.1f}s: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original service
            worker_mod.document_service = original_service
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_verification()) 