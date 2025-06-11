#!/usr/bin/env python3
"""
Monitor Strategic Intelligence Brief Verification

This script monitors your document verification progress and automatically
downloads the report when processing is complete.
"""

import requests
import time
import json
from datetime import datetime

def monitor_verification():
    """Monitor verification progress and download report when complete"""
    
    # API configuration
    BASE_URL = "http://localhost:8000"
    USERNAME = "testuser"
    PASSWORD = "TestPass123!"
    TASK_ID = "12345"
    
    print("🔍 Strategic Intelligence Brief Verification Monitor")
    print("=" * 60)
    
    # Authenticate
    print("🔑 Authenticating...")
    auth_response = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"username": USERNAME, "password": PASSWORD}
    )
    
    if auth_response.status_code != 200:
        print("❌ Authentication failed!")
        return False
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful!")
    
    # Monitor progress
    print(f"\n📊 Monitoring verification progress for task {TASK_ID}...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Get current status
            status_response = requests.get(
                f"{BASE_URL}/api/v1/verification/{TASK_ID}/status",
                headers=headers
            )
            
            if status_response.status_code != 200:
                print("❌ Failed to get status")
                break
            
            status_data = status_response.json()
            status = status_data["status"]
            progress = status_data["progress"]
            current_pass = status_data.get("current_pass", "unknown")
            
            # Display progress
            completed = progress["completed"]
            total = progress["total"]
            percentage = (completed / total) * 100
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Status: {status} | Progress: {completed}/{total} ({percentage:.1f}%) | Current: {current_pass}")
            
            # Check if complete
            if status == "COMPLETED":
                print(f"\n🎉 Verification completed!")
                break
            elif status == "FAILED":
                print(f"\n❌ Verification failed!")
                error_msg = status_data.get("error_message", "Unknown error")
                print(f"Error: {error_msg}")
                break
            
            # Wait before next check
            time.sleep(10)
        
        # Download report if completed successfully
        if status == "COMPLETED":
            print(f"\n📥 Downloading verification report...")
            
            # Get detailed results
            results_response = requests.get(
                f"{BASE_URL}/api/v1/verification/{TASK_ID}/results",
                headers=headers
            )
            
            if results_response.status_code == 200:
                results_data = results_response.json()
                
                # Save JSON report
                with open("strategic_brief_verification_report.json", "w") as f:
                    json.dump(results_data, f, indent=2)
                print("✅ JSON report saved: strategic_brief_verification_report.json")
                
                # Display summary
                print(f"\n📋 Verification Summary:")
                summary = results_data.get("summary", {})
                print(f"   Overall Score: {summary.get('verification_score', 'N/A')}/10")
                print(f"   Total Issues: {summary.get('total_issues', 'N/A')}")
                print(f"   Critical Issues: {summary.get('critical_issues', 'N/A')}")
                print(f"   Confidence: {summary.get('average_confidence', 'N/A')}")
                
                # Get metadata
                metadata = results_data.get("metrics", {})
                print(f"\n⚡ Processing Metrics:")
                print(f"   Processing Time: {metadata.get('processing_time', 'N/A')} seconds")
                print(f"   Tokens Processed: {metadata.get('tokens_processed', 'N/A')}")
                print(f"   API Calls: {metadata.get('api_calls', 'N/A')}")
                
            else:
                print("❌ Failed to download detailed results")
            
            # Try to download PDF report
            try:
                pdf_response = requests.get(
                    f"{BASE_URL}/api/v1/verification/{TASK_ID}/report?format=pdf",
                    headers=headers
                )
                
                if pdf_response.status_code == 200:
                    with open("strategic_brief_verification_report.pdf", "wb") as f:
                        f.write(pdf_response.content)
                    print("✅ PDF report saved: strategic_brief_verification_report.pdf")
                else:
                    print("⚠️  PDF report not available (endpoint may not be implemented)")
            except:
                print("⚠️  PDF download not available")
            
            print(f"\n🚀 Your Strategic Intelligence Brief has been fully verified!")
            print(f"📊 Check the JSON report for detailed analysis of all claims.")
            
            return True
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped by user")
        print(f"You can resume monitoring by running this script again.")
        return False
    
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        return False


if __name__ == "__main__":
    monitor_verification() 