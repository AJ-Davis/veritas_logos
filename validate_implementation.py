#!/usr/bin/env python3
"""
Simple validation script to check if the claim extraction implementation is working.
This script performs basic validation without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path to allow src.* imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Test claim models
        from src.models.claims import (
            ExtractedClaim, 
            ClaimExtractionResult, 
            ClaimLocation, 
            ClaimType, 
            ClaimCategory
        )
        print("✅ Claim models imported successfully")
        
        # Test LLM client
        from src.llm.llm_client import LLMClient, LLMConfig, LLMProvider, LLMResponse
        print("✅ LLM client imported successfully")
        
        # Test prompts
        from src.llm.prompts import PromptType, prompt_manager
        print("✅ Prompt templates imported successfully")
        
        # Test claim extraction pass
        from src.verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass
        print("✅ Claim extraction pass imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_model_creation():
    """Test that models can be created successfully."""
    print("\nTesting model creation...")
    
    try:
        from src.models.claims import (
            ClaimLocation,
            ExtractedClaim,
            ClaimExtractionResult,
            ClaimType,
            ClaimCategory,
        )
        # Test ClaimLocation
        location = ClaimLocation(
            start_position=100,
            end_position=200,
            context_before="Before text",
            context_after="After text"
        )
        print("✅ ClaimLocation created successfully")
        
        # Test ExtractedClaim
        claim = ExtractedClaim(
            claim_text="Test claim",
            claim_type=ClaimType.FACTUAL,
            category=ClaimCategory.GENERAL,
            location=location,
            document_id="test_doc",
            extraction_confidence=0.9,
            extracted_by="test"
        )
        print("✅ ExtractedClaim created successfully")
        
        # Test ClaimExtractionResult
        result = ClaimExtractionResult(
            document_id="test",
            claims=[claim],
            total_claims_found=1,
            model_used="test",
            prompt_version="v1",
            document_length=1000
        )
        print("✅ ClaimExtractionResult created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_prompt_templates():
    """Test that prompt templates work correctly."""
    print("\nTesting prompt templates...")
    
    try:
        from src.llm.prompts import PromptType, prompt_manager
        
        # Get claim extraction template
        template = prompt_manager.get_template(PromptType.CLAIM_EXTRACTION, "v1")
        print("✅ Retrieved claim extraction template")
        
        # Create messages
        messages = prompt_manager.create_messages(
            template,
            document_text="Sample document text",
            max_claims=10
        )
        print("✅ Created prompt messages")
        
        # Validate messages structure
        assert len(messages) >= 2, "Should have at least system and user messages"
        assert messages[0]["role"] == "system", "First message should be system"
        assert messages[-1]["role"] == "user", "Last message should be user"
        assert "Sample document text" in messages[-1]["content"], "User message should contain document text"
        print("✅ Messages structure validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt template error: {e}")
        return False


def test_verification_pass_init():
    """Test that verification pass can be initialized."""
    print("\nTesting verification pass initialization...")
    
    try:
        from src.verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass
        from src.models.verification import VerificationPassType
        
        # Create pass with mock LLM client
        pass_instance = ClaimExtractionPass()
        print("✅ ClaimExtractionPass created successfully")
        
        # Verify properties
        assert pass_instance.pass_type == VerificationPassType.CLAIM_EXTRACTION
        print("✅ Pass type verified")
        
        assert len(pass_instance.get_required_dependencies()) == 0
        print("✅ Dependencies verified (should be empty)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification pass error: {e}")
        return False


def test_integration_with_worker():
    """Test integration with verification worker."""
    print("\nTesting integration with verification worker...")
    
    try:
        from src.verification.workers.verification_worker import VerificationWorker
        from src.verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass
        from src.models.verification import VerificationPassType
        
        # Create worker
        worker = VerificationWorker()
        print("✅ VerificationWorker created successfully")
        
        # Check that claim extraction pass is registered
        assert VerificationPassType.CLAIM_EXTRACTION in worker.pass_registry
        print("✅ Claim extraction pass is registered")
        
        # Verify it's the right type
        claim_pass = worker.pass_registry[VerificationPassType.CLAIM_EXTRACTION]
        assert isinstance(claim_pass, ClaimExtractionPass)
        print("✅ Registered pass is ClaimExtractionPass instance")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker integration error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print(" CLAIM EXTRACTION IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_prompt_templates,
        test_verification_pass_init,
        test_integration_with_worker
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Implementation looks good.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())