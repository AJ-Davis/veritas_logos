#!/usr/bin/env python3
"""
Test script for the Enhanced LLM Service with Multi-Provider Support

This script demonstrates:
1. Multi-provider LLM integration (Anthropic, Gemini, DeepSeek)
2. Automatic fallback between providers
3. Mock mode for testing without API keys
4. Real document verification with the Strategic Intelligence Brief
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🧠 Enhanced LLM Service Test")
print("=" * 50)

async def test_enhanced_llm_service():
    """Test the enhanced LLM service with multiple providers"""
    
    try:
        from src.llm.enhanced_llm_service import EnhancedLLMService, LLMProvider, LLMConfig
        print("✅ Successfully imported Enhanced LLM Service")
    except ImportError as e:
        print(f"❌ Failed to import Enhanced LLM Service: {e}")
        return False
    
    # Initialize the service (will auto-detect available providers)
    print("\n🔧 Initializing Enhanced LLM Service...")
    service = EnhancedLLMService()
    
    # Check available providers
    providers = service.get_available_providers()
    provider_info = service.get_provider_info()
    
    print(f"\n📊 Available Providers: {len(providers)}")
    for provider in providers:
        info = provider_info[provider.value]
        status = "✅ Enabled" if info["enabled"] else "❌ Disabled"
        print(f"  - {provider.value}: {info['model']} (Priority: {info['priority']}) {status}")
    
    # Test basic text generation
    print("\n🧪 Testing Basic Text Generation...")
    test_prompt = "Explain the concept of renewable energy in 2-3 sentences."
    
    try:
        response = await service.generate_text(
            prompt=test_prompt,
            max_tokens=150,
            temperature=0.7
        )
        
        print(f"✅ Text Generation Successful!")
        print(f"Provider Used: {response.provider.value}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Response: {response.content[:200]}...")
        
    except Exception as e:
        print(f"❌ Text Generation Failed: {e}")
        return False
    
    # Test with specific provider preference
    print("\n🎯 Testing Provider Preference...")
    for provider in providers:
        try:
            response = await service.generate_text(
                prompt="What is artificial intelligence?",
                max_tokens=100,
                preferred_provider=provider
            )
            print(f"✅ {provider.value}: {response.content[:100]}...")
        except Exception as e:
            print(f"❌ {provider.value}: Failed - {e}")
    
    # Test claim extraction with the PDF content
    print("\n📄 Testing Document Verification...")
    
    # Read the PDF content (simulated - in real usage this would be extracted from PDF)
    test_document = """
    Strategic Intelligence Brief: Energy Innovation
    
    Key findings from our analysis:
    
    1. Solar energy costs have decreased by 85% over the past decade
    2. Wind power capacity has increased by 260% globally since 2010  
    3. Energy storage technology costs have fallen by 70% in the last 5 years
    4. Renewable energy now accounts for 30% of global electricity generation
    5. Investment in clean energy reached $1.8 trillion in 2023
    
    These developments indicate a fundamental shift in the global energy landscape.
    Expert analysis suggests that renewable energy will dominate new capacity additions through 2030.
    """
    
    try:
        # Test with claim extraction prompt
        claim_prompt = """Analyze this document and extract the key factual claims that can be verified:

{document}

For each claim, identify:
1. The specific factual assertion
2. Whether it includes quantitative data
3. The level of verification needed
""".format(document=test_document)
        
        response = await service.generate_text(
            prompt=claim_prompt,
            system_prompt="You are an expert fact-checker analyzing documents for verifiable claims.",
            max_tokens=800,
            temperature=0.3
        )
        
        print(f"✅ Document Analysis Successful!")
        print(f"Provider Used: {response.provider.value}")
        print(f"Analysis Preview: {response.content[:300]}...")
        
    except Exception as e:
        print(f"❌ Document Analysis Failed: {e}")
    
    # Test with different temperature settings
    print("\n🌡️ Testing Temperature Variations...")
    creative_prompt = "Write a creative title for a renewable energy report."
    
    for temp in [0.1, 0.5, 0.9]:
        try:
            response = await service.generate_text(
                prompt=creative_prompt,
                max_tokens=50,
                temperature=temp
            )
            print(f"Temperature {temp}: {response.content.strip()}")
        except Exception as e:
            print(f"Temperature {temp}: Failed - {e}")
    
    print(f"\n🎉 Enhanced LLM Service Test Complete!")
    return True


async def test_verification_integration():
    """Test the enhanced LLM service with the actual verification system"""
    print(f"\n🔗 Testing Integration with Verification System...")
    
    try:
        from src.verification.passes.implementations.claim_extraction import ClaimExtractionPass
        print("✅ Successfully imported ClaimExtractionPass")
    except ImportError as e:
        print(f"❌ Failed to import ClaimExtractionPass: {e}")
        return False
    
    # Initialize the claim extraction pass (should use enhanced LLM)
    claim_extractor = ClaimExtractionPass()
    
    # Test document
    test_doc = """
    Climate Change Impact Report 2024
    
    Global temperatures have risen by 1.2°C since pre-industrial times according to NASA data.
    Arctic sea ice is declining at a rate of 13% per decade.
    Ocean levels have risen 21-24 centimeters since 1880.
    Carbon dioxide levels reached 421 parts per million in 2023, the highest in human history.
    
    The Intergovernmental Panel on Climate Change states that limiting warming to 1.5°C 
    requires reducing global greenhouse gas emissions by 45% by 2030.
    """
    
    try:
        result = await claim_extractor.execute(
            document_text=test_doc,
            context={"source": "test_document"}
        )
        
        print(f"✅ Verification Integration Successful!")
        print(f"Status: {result.status.value}")
        print(f"Claims Extracted: {result.details.get('claims_extracted', 0)}")
        print(f"Provider Used: {result.details.get('provider_used', 'unknown')}")
        
        # Show extracted claims
        claims = result.details.get('claims', [])
        if claims:
            print("\n📋 Extracted Claims:")
            for i, claim in enumerate(claims[:3], 1):  # Show first 3 claims
                print(f"  {i}. {claim.get('text', '')[:100]}...")
                print(f"     Category: {claim.get('category', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Verification Integration Failed: {e}")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Enhanced LLM Service Tests")
    print("=" * 60)
    
    # Check environment
    print("🔍 Environment Check:")
    api_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "not-set"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "not-set"), 
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "not-set"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", "not-set")
    }
    
    for key, value in api_keys.items():
        status = "✅ Set" if value != "not-set" and value != "your-anthropic-api-key-here" else "❌ Not Set"
        print(f"  {key}: {status}")
    
    print("\nNote: Tests will use mock responses for providers without API keys.")
    
    # Run tests
    success = True
    
    try:
        success &= await test_enhanced_llm_service()
        success &= await test_verification_integration()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nYour Enhanced LLM Service is ready for multi-provider document verification!")
            print("\nTo use with real API keys:")
            print("1. Set ANTHROPIC_API_KEY for Claude")
            print("2. Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini")  
            print("3. Set DEEPSEEK_API_KEY for DeepSeek")
            print("4. Restart the server to pick up new environment variables")
        else:
            print("\n❌ Some tests failed - check the logs above")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        success = False
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 