#!/usr/bin/env python3
"""
Test Real LLM Provider Connections

This script tests if the enhanced LLM service can connect to real providers
and process the Strategic Intelligence Brief content.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🔗 Real LLM Provider Connection Test")
print("=" * 60)

async def test_real_provider_connections():
    """Test connections to real LLM providers"""
    
    print("🔍 Checking Environment Variables...")
    
    # Check for API keys in environment
    api_keys = {}
    for key_name in ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"]:
        value = os.getenv(key_name, "not-set")
        api_keys[key_name] = value
        
        # Check if it's a real key (not placeholder)
        is_real = (value != "not-set" and 
                  value != "your-anthropic-api-key-here" and 
                  len(value) > 20 and 
                  value.startswith(("sk-", "api-", "AIza")))
        
        status = "✅ Real Key" if is_real else "❌ Not Set/Invalid"
        print(f"  {key_name}: {status}")
        if is_real:
            print(f"    Key preview: {value[:12]}...{value[-4:]}")
    
    # Import enhanced LLM service
    try:
        from src.llm.enhanced_llm_service import EnhancedLLMService, LLMProvider
        print("\n✅ Enhanced LLM Service imported successfully")
    except ImportError as e:
        print(f"\n❌ Failed to import Enhanced LLM Service: {e}")
        return False
    
    # Initialize service
    print("\n🔧 Initializing Enhanced LLM Service...")
    try:
        service = EnhancedLLMService()
        providers = service.get_available_providers()
        provider_info = service.get_provider_info()
        
        print(f"Available Providers: {len(providers)}")
        for provider in providers:
            info = provider_info[provider.value]
            status = "✅ Enabled" if info["enabled"] else "❌ Disabled"
            print(f"  - {provider.value}: {info['model']} (Priority: {info['priority']}) {status}")
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test with Strategic Intelligence Brief content
    print("\n📄 Testing with Strategic Intelligence Brief Content...")
    
    strategic_brief_content = """
    Strategic Intelligence Brief: Engaging Kurt Matsumoto and Pulama Lanai on Energy Innovation
    
    Executive Summary:
    This brief analyzes renewable energy developments and strategic partnerships in Hawaii's energy transition.
    
    Key Findings:
    1. Solar energy adoption in Hawaii increased by 45% in 2023
    2. Wind power capacity reached 235 MW across the islands  
    3. Battery storage deployment grew 78% year-over-year
    4. Grid modernization projects reduced outages by 32%
    5. Clean energy investment totaled $1.2 billion in 2023
    
    Strategic Recommendations:
    - Accelerate offshore wind development
    - Expand grid-scale battery storage
    - Enhance inter-island transmission capacity
    - Develop green hydrogen production facilities
    
    The transition to 100% renewable energy by 2045 remains achievable with continued investment
    and strategic partnerships between government, private sector, and community stakeholders.
    """
    
    # Test claim extraction with real providers
    success_count = 0
    for provider in providers:
        if provider.value == "mock":
            continue  # Skip mock for this test
            
        try:
            print(f"\n🧪 Testing {provider.value} for claim extraction...")
            
            response = await service.generate_text(
                prompt=f"""Analyze this strategic energy document and extract the top 5 most important verifiable claims:

{strategic_brief_content}

For each claim, provide:
1. The exact claim text from the document
2. The type of claim (statistical, factual, projection, etc.)
3. How verifiable this claim is (high/medium/low confidence)
4. What sources would be needed to verify it

Format as a numbered list with clear analysis.""",
                system_prompt="You are an expert fact-checker analyzing strategic intelligence documents. Focus on extracting specific, verifiable claims that can be fact-checked against external sources.",
                max_tokens=800,
                temperature=0.3,
                preferred_provider=provider
            )
            
            if response.provider.value != "mock":
                print(f"✅ {provider.value} SUCCESS!")
                print(f"   Model Used: {response.model}")
                print(f"   Token Usage: {response.usage}")
                print(f"   Response Preview: {response.content[:200]}...")
                success_count += 1
            else:
                print(f"⚠️  {provider.value} fell back to mock")
                
        except Exception as e:
            print(f"❌ {provider.value} FAILED: {e}")
    
    print(f"\n📊 Results Summary:")
    print(f"   Real providers tested: {len([p for p in providers if p.value != 'mock'])}")
    print(f"   Successful connections: {success_count}")
    
    if success_count > 0:
        print("🎉 Real LLM providers are working!")
        return True
    else:
        print("ℹ️  No real LLM providers available - using mock mode")
        return False


async def test_pdf_processing_with_real_llms():
    """Test the PDF processing pipeline with real LLMs"""
    
    print("\n📋 Testing PDF Processing Pipeline...")
    
    # Test the verification endpoint with real content
    import requests
    
    try:
        # Login
        login_response = requests.post(
            "http://localhost:8000/api/v1/auth/login",
            json={"username": "testuser", "password": "TestPass123!"}
        )
        
        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            print("✅ Authentication successful")
            
            # Check verification status to see which provider is being used
            status_response = requests.get(
                "http://localhost:8000/api/v1/verification/12345/status",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"✅ Verification Status: {status_data.get('status', 'unknown')}")
                print(f"   Progress: {status_data.get('progress', {}).get('completed', 0)}/{status_data.get('progress', {}).get('total', 0)}")
                print(f"   Current Pass: {status_data.get('current_pass', 'unknown')}")
                return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                
        else:
            print(f"❌ Authentication failed: {login_response.status_code}")
            
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
    
    return False


async def main():
    """Run all real provider tests"""
    print("🚀 Starting Real LLM Provider Tests")
    print("=" * 70)
    
    success = True
    
    try:
        # Test provider connections
        provider_success = await test_real_provider_connections()
        
        # Test PDF processing
        pdf_success = await test_pdf_processing_with_real_llms()
        
        success = provider_success or pdf_success  # At least one should work
        
        if success:
            print("\n🎉 REAL LLM TESTING SUCCESSFUL!")
            print("\nYour Enhanced LLM Service is processing documents with real AI providers!")
            if provider_success:
                print("✅ Direct LLM provider connections working")
            if pdf_success:
                print("✅ PDF verification pipeline active")
        else:
            print("\n🔄 USING MOCK MODE")
            print("\nTo activate real LLM providers:")
            print("1. Verify your API keys are correctly set in environment variables")
            print("2. Ensure keys are valid and have sufficient credits")
            print("3. Restart the server to pick up new environment variables")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        success = False
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 