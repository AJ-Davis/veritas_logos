#!/usr/bin/env python3
"""
Simple verification test script that bypasses file upload.
"""

import requests
import json

# Test document content
test_content = """
# Climate Change Analysis

## Key Claims

1. Global temperatures have risen by approximately 1.1°C since pre-industrial times.
2. Solar energy costs have fallen by over 80% in the last decade.
3. Wind power costs have decreased by approximately 70% in the same period.
4. Ocean pH levels have dropped by 0.1 units since the Industrial Revolution.

## Sources
- IPCC reports
- International Renewable Energy Agency
- NASA climate data
"""

def test_verification():
    """Test the verification system with sample content."""
    
    # API base URL
    base_url = "http://localhost:8000/api/v1"
    
    # Authentication
    auth_data = {
        "username": "testuser",
        "password": "TestPass123!"
    }
    
    print("🔐 Authenticating...")
    auth_response = requests.post(f"{base_url}/auth/login", json=auth_data)
    
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.text}")
        return
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful!")
    
    # Test verification endpoint with text content
    print("\n📄 Testing verification with sample content...")
    
    verification_data = {
        "document_text": test_content,
        "verification_type": "comprehensive",
        "include_citations": True,
        "include_bias_analysis": True
    }
    
    try:
        verify_response = requests.post(
            f"{base_url}/verify", 
            headers=headers,
            json=verification_data,
            timeout=30
        )
        
        print(f"Response status: {verify_response.status_code}")
        
        if verify_response.status_code == 200:
            result = verify_response.json()
            print("✅ Verification successful!")
            print(f"Task ID: {result.get('task_id', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            return result.get('task_id')
        else:
            print(f"❌ Verification failed: {verify_response.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out - verification may still be processing")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    print("🧪 Testing Veritas Logos Verification System")
    print("=" * 50)
    
    task_id = test_verification()
    
    if task_id:
        print(f"\n🎉 Test completed! Task ID: {task_id}")
        print("You can check the task status using the API or web interface.")
    else:
        print("\n❌ Test failed or incomplete.") 