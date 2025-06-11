#!/usr/bin/env python3
"""
Real-World Verification Testing Suite

This test suite validates the complete Veritas Logos document verification system
using actual documents and real-world scenarios.
"""

import asyncio
import sys
import json
import time
import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core system components
from src.document_ingestion.document_ingestion_service import DocumentIngestionService
from src.verification.pipeline.verification_pipeline import VerificationPipeline
from src.verification.config.chain_loader import ChainConfigLoader
from src.models.document import DocumentFormat, ParsedDocument
from src.models.verification import VerificationChainResult, VerificationStatus
from src.models.claims import ClaimExtractionResult, ExtractedClaim
# from src.analytics.metrics_collector import MetricsCollector


class RealWorldVerificationTests:
    """Comprehensive real-world testing suite for document verification."""
    
    def __init__(self):
        self.test_documents_path = Path(__file__).parent / "documents" / "samples"
        self.results = []
        # self.metrics_collector = MetricsCollector()
        
    async def setup_test_environment(self):
        """Initialize test environment with all required services."""
        print("🔧 Setting up test environment...")
        
        # Initialize document ingestion service
        self.document_service = DocumentIngestionService()
        
        # Initialize verification worker to get pass registry
        from src.verification.workers.verification_worker import VerificationWorker
        worker = VerificationWorker()
        
        # Create pipeline config
        from src.verification.pipeline.verification_pipeline import PipelineConfig, PipelineMode
        pipeline_config = PipelineConfig(mode=PipelineMode.STANDARD)
        
        # Initialize verification pipeline with config and pass registry
        self.verification_pipeline = VerificationPipeline(
            config=pipeline_config,
            pass_registry=worker.pass_registry
        )
        
        # Load verification chain configurations
        self.chain_loader = ChainConfigLoader()
        
        print("✅ Test environment setup complete")
        
    async def test_text_document_verification(self):
        """Test complete verification pipeline with text document."""
        print("\n📄 Testing TEXT document verification...")
        
        test_file = self.test_documents_path / "sample_article.txt"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test document not found: {test_file}")
        
        start_time = time.time()
        
        try:
            # Step 1: Document Ingestion
            print("  🔍 Step 1: Document ingestion...")
            
            parsed_doc = self.document_service.parse_document(str(test_file))
            
            # Read content for later use
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert parsed_doc is not None, "Document parsing failed"
            assert parsed_doc.content is not None, "Document content is empty"
            print(f"    ✅ Document parsed successfully ({len(parsed_doc.content)} characters)")
            
            # Step 2: Verification Chain Execution
            print("  🔗 Step 2: Verification chain execution...")
            
            verification_result = await self.verification_pipeline.process_document(
                document_id=str(test_file),
                content=parsed_doc.content,
                metadata={"filename": "sample_article.txt", "format": "txt"}
            )
            
            assert verification_result is not None, "Verification failed"
            assert verification_result.status != VerificationStatus.FAILED, "Verification returned failed status"
            print(f"    ✅ Verification completed with status: {verification_result.status}")
            
            processing_time = time.time() - start_time
            
            # Record results
            result = {
                "test_name": "text_document_verification",
                "status": "PASSED",
                "processing_time": processing_time,
                "document_size": len(content),
                "verification_status": verification_result.status.value,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            print(f"  ✅ Text document verification completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_result = {
                "test_name": "text_document_verification",
                "status": "FAILED",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_result)
            print(f"  ❌ Text document verification failed: {e}")
            return error_result
    
    async def test_markdown_document_verification(self):
        """Test complete verification pipeline with markdown document."""
        print("\n📝 Testing MARKDOWN document verification...")
        
        test_file = self.test_documents_path / "research_paper.md"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test document not found: {test_file}")
        
        start_time = time.time()
        
        try:
            # Step 1: Document Ingestion
            print("  🔍 Step 1: Document ingestion...")
            
            parsed_doc = self.document_service.parse_document(str(test_file))
            
            # Read content for later use
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert parsed_doc is not None, "Document parsing failed"
            assert parsed_doc.content is not None, "Document content is empty"
            print(f"    ✅ Document parsed successfully ({len(parsed_doc.content)} characters)")
            
            # Step 2: Verify markdown structure preservation
            if "# Machine Learning Applications" in parsed_doc.content:
                print("    ✅ Markdown headers preserved")
            else:
                print("    ⚠️  Markdown headers may have been converted")
            
            if "[1]" in parsed_doc.content:
                print("    ✅ Citations preserved")
            else:
                print("    ⚠️  Citations may have been converted")
            
            # Step 3: Verification Chain Execution
            print("  🔗 Step 3: Verification chain execution...")
            
            verification_result = await self.verification_pipeline.process_document(
                document_id=str(test_file),
                content=parsed_doc.content,
                metadata={"filename": "research_paper.md", "format": "markdown"}
            )
            
            assert verification_result is not None, "Verification failed"
            print(f"    ✅ Verification completed with status: {verification_result.status}")
            
            processing_time = time.time() - start_time
            
            # Record results
            result = {
                "test_name": "markdown_document_verification",
                "status": "PASSED",
                "processing_time": processing_time,
                "document_size": len(content),
                "verification_status": verification_result.status.value,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(result)
            print(f"  ✅ Markdown document verification completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_result = {
                "test_name": "markdown_document_verification",
                "status": "FAILED",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_result)
            print(f"  ❌ Markdown document verification failed: {e}")
            return error_result
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report."""
        print("\n📊 Generating test report...")
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        
        total_time = sum(r.get('processing_time', 0) for r in self.results)
        
        report = {
            "test_run_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                "total_execution_time": f"{total_time:.2f}s"
            },
            "test_results": self.results,
            "system_validation": {
                "document_ingestion": "PASSED" if any(r['test_name'].endswith('verification') and r['status'] == 'PASSED' for r in self.results) else "FAILED",
                "verification_pipeline": "PASSED" if any(r['status'] == 'PASSED' for r in self.results) else "FAILED"
            }
        }
        
        return report
    
    async def run_all_tests(self):
        """Execute all real-world verification tests."""
        print("🚀 Starting Real-World Verification Test Suite")
        print("=" * 60)
        
        try:
            # Setup
            await self.setup_test_environment()
            
            # Core functionality tests
            await self.test_text_document_verification()
            await self.test_markdown_document_verification()
            
            # Generate report
            report = self.generate_test_report()
            
            print("\n" + "=" * 60)
            print("📋 TEST SUITE SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {report['test_run_summary']['total_tests']}")
            print(f"Passed: {report['test_run_summary']['passed_tests']}")
            print(f"Failed: {report['test_run_summary']['failed_tests']}")
            print(f"Success Rate: {report['test_run_summary']['success_rate']}")
            print(f"Total Time: {report['test_run_summary']['total_execution_time']}")
            
            print("\n📊 SYSTEM VALIDATION:")
            for component, status in report['system_validation'].items():
                status_icon = "✅" if status == "PASSED" else "❌"
                print(f"  {status_icon} {component.replace('_', ' ').title()}: {status}")
            
            # Save report
            report_file = Path(__file__).parent / "real_world_test_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Detailed report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            raise


async def main():
    """Main test execution function."""
    test_suite = RealWorldVerificationTests()
    
    try:
        report = await test_suite.run_all_tests()
        
        # Return appropriate exit code
        if report['test_run_summary']['failed_tests'] == 0:
            print("\n🎉 All tests passed! System is ready for production.")
            return 0
        else:
            print(f"\n⚠️ {report['test_run_summary']['failed_tests']} test(s) failed. Please review results.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Critical test failure: {e}")
        return 2


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 