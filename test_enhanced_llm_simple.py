#!/usr/bin/env python3
"""
Simple test for Enhanced LLM Service - no verification integration

This focuses only on testing the enhanced LLM service itself.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🧠 Enhanced LLM Service Simple Test")
print("=" * 50)

async def test_enhanced_llm_basic():
    """Test the enhanced LLM service basic functionality"""
    
    print("🔧 Testing Enhanced LLM Service Import...")
    try:
        from src.llm.enhanced_llm_service import EnhancedLLMService, LLMProvider, LLMConfig
        print("✅ Successfully imported Enhanced LLM Service")
    except ImportError as e:
        print(f"❌ Failed to import Enhanced LLM Service: {e}")
        return False
    
    print("\n🔧 Initializing Service...")
    try:
        service = EnhancedLLMService()
        print("✅ Service initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    print("\n📊 Checking Available Providers...")
    try:
        providers = service.get_available_providers()
        provider_info = service.get_provider_info()
        
        print(f"Available Providers: {len(providers)}")
        for provider in providers:
            info = provider_info[provider.value]
            status = "✅ Enabled" if info["enabled"] else "❌ Disabled"
            print(f"  - {provider.value}: {info['model']} (Priority: {info['priority']}) {status}")
    except Exception as e:
        print(f"❌ Provider check failed: {e}")
        return False
    
    print("\n🧪 Testing Text Generation...")
    try:
        response = await service.generate_text(
            prompt="What is renewable energy?",
            max_tokens=100,
            temperature=0.7
        )
        
        print("✅ Text Generation Successful!")
        print(f"Provider Used: {response.provider.value}")
        print(f"Model: {response.model}")
        print(f"Response: {response.content[:150]}...")
        
    except Exception as e:
        print(f"❌ Text Generation Failed: {e}")
        return False
    
    print("\n🔄 Testing Provider Fallback...")
    try:
        # Test with different providers
        for provider in providers:
            response = await service.generate_text(
                prompt="Explain climate change in one sentence.",
                max_tokens=50,
                preferred_provider=provider
            )
            print(f"✅ {provider.value}: Success")
    except Exception as e:
        print(f"❌ Provider fallback test failed: {e}")
        return False
    
    print("\n🎉 Enhanced LLM Service Basic Test Complete!")
    return True


async def test_enhanced_llm_with_pdf_content():
    """Test enhanced LLM with PDF-like content"""
    
    print("\n📄 Testing with PDF-like Content...")
    
    try:
        from src.llm.enhanced_llm_service import get_enhanced_llm_service
        service = get_enhanced_llm_service()
        
        # Simulate Strategic Intelligence Brief content
        pdf_content = """
        Strategic Intelligence Brief: Energy Innovation
        
        Executive Summary:
        The global energy landscape is undergoing rapid transformation. Key developments include:
        
        1. Solar photovoltaic costs decreased by 85% between 2010-2020
        2. Wind energy capacity increased 260% globally since 2010
        3. Energy storage costs fell 70% in the last 5 years
        4. Renewable energy accounts for 30% of global electricity generation
        5. Clean energy investment reached $1.8 trillion in 2023
        
        These trends indicate accelerating adoption of renewable technologies.
        Government policies and corporate commitments are driving this transition.
        """
        
        # Test claim extraction prompt
        response = await service.generate_text(
            prompt=f"""Analyze this energy sector document and extract 3-5 key factual claims that can be verified:

{pdf_content}

For each claim, provide:
1. The exact claim text
2. Whether it's a statistical claim or general factual claim
3. How verifiable it is (high/medium/low)""",
            system_prompt="You are an expert fact-checker analyzing documents for verifiable claims.",
            max_tokens=600,
            temperature=0.3
        )
        
        print("✅ PDF Content Analysis Successful!")
        print(f"Provider Used: {response.provider.value}")
        print(f"Analysis: {response.content[:400]}...")
        
    except Exception as e:
        print(f"❌ PDF Content Analysis Failed: {e}")
        return False
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Enhanced LLM Service Simple Tests")
    print("=" * 60)
    
    # Environment check
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
        success &= await test_enhanced_llm_basic()
        success &= await test_enhanced_llm_with_pdf_content()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nYour Enhanced LLM Service is working correctly!")
            print("\nTo use with real LLM providers:")
            print("1. Set ANTHROPIC_API_KEY for Claude")
            print("2. Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini")  
            print("3. Set DEEPSEEK_API_KEY for DeepSeek")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        success = False
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 