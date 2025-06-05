"""
Example script demonstrating claim extraction functionality.

This script shows how to:
1. Set up the claim extraction pass
2. Process a document to extract claims
3. Analyze the extracted claims
"""

import asyncio
import os
import json
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.verification import (
    VerificationContext,
    VerificationPassConfig,
    VerificationPassType
)
from models.claims import ClaimExtractionResult
from verification.passes.implementations.claim_extraction_pass import ClaimExtractionPass
from llm.llm_client import LLMClient, LLMConfig, LLMProvider


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_claim_summary(extraction_result: ClaimExtractionResult):
    """Print a summary of extracted claims."""
    print(f"\n📊 EXTRACTION SUMMARY")
    print(f"   Document ID: {extraction_result.document_id}")
    print(f"   Total Claims: {extraction_result.total_claims_found}")
    print(f"   Average Confidence: {extraction_result.average_confidence:.2f}")
    print(f"   Processing Time: {extraction_result.processing_time_seconds:.2f}s")
    print(f"   Model Used: {extraction_result.model_used}")
    print(f"   Claims per 1000 chars: {extraction_result.claims_per_1000_chars:.1f}")


def print_claim_details(claims):
    """Print detailed information about extracted claims."""
    print(f"\n📝 EXTRACTED CLAIMS")
    
    for i, claim in enumerate(claims, 1):
        print(f"\n--- Claim {i} ---")
        print(f"Text: {claim.claim_text}")
        print(f"Type: {claim.claim_type.value}")
        print(f"Category: {claim.category.value}")
        print(f"Confidence: {claim.extraction_confidence:.2f}")
        
        if claim.importance_score:
            print(f"Importance: {claim.importance_score:.2f}")
        
        if claim.citations:
            print(f"Citations: {', '.join(claim.citations)}")
        
        if claim.location.context_before:
            context_before = claim.location.context_before[-50:] if len(claim.location.context_before) > 50 else claim.location.context_before
            print(f"Context Before: ...{context_before}")
        
        if claim.location.context_after:
            context_after = claim.location.context_after[:50] if len(claim.location.context_after) > 50 else claim.location.context_after
            print(f"Context After: {context_after}...")
        
        print(f"Position: {claim.location.start_position}-{claim.location.end_position}")


def print_statistics(extraction_result: ClaimExtractionResult):
    """Print statistics about extracted claims."""
    print(f"\n📈 STATISTICS")
    
    # Type distribution
    print(f"\n📊 Claim Types:")
    for claim_type, count in extraction_result.claim_type_distribution.items():
        percentage = (count / extraction_result.total_claims_found) * 100
        print(f"   {claim_type}: {count} ({percentage:.1f}%)")
    
    # Confidence distribution
    print(f"\n🎯 Confidence Distribution:")
    for level, count in extraction_result.confidence_distribution.items():
        percentage = (count / extraction_result.total_claims_found) * 100
        print(f"   {level}: {count} ({percentage:.1f}%)")


async def run_claim_extraction_example():
    """Run the claim extraction example."""
    
    print_header("CLAIM EXTRACTION EXAMPLE")
    
    # Sample document content - scientific paper about climate change
    sample_document = """
    Climate Change and Renewable Energy: A Comprehensive Analysis
    
    Abstract:
    This study examines the relationship between climate change and renewable energy adoption globally. 
    Our research indicates that solar energy capacity has grown by 260% over the past decade, making it 
    the fastest-growing energy source worldwide. Wind power now accounts for approximately 24% of renewable 
    electricity generation in OECD countries.
    
    Introduction:
    Climate change represents one of the most significant challenges of the 21st century. The 
    Intergovernmental Panel on Climate Change (IPCC) reports that global mean temperatures have increased 
    by 1.1°C since the pre-industrial period (1850-1900). Carbon dioxide levels in the atmosphere have 
    reached 415 parts per million, the highest level in over 3 million years.
    
    Methodology:
    We analyzed data from 195 countries over a 15-year period (2008-2023). Our dataset includes energy 
    production statistics, carbon emission measurements, and economic indicators. The study employed 
    machine learning algorithms to identify correlations between renewable energy investments and 
    emission reductions.
    
    Results:
    Our analysis reveals several key findings:
    
    1. Countries that increased their renewable energy share by more than 50% showed a corresponding 
       35% reduction in carbon intensity over the study period.
    
    2. Solar photovoltaic costs have decreased by 89% since 2010, making it cost-competitive with 
       fossil fuels in 140+ countries.
    
    3. Electric vehicle sales surged to 10.5 million units globally in 2022, representing a 55% 
       increase from the previous year.
    
    4. Offshore wind capacity expanded by 8.8 GW in 2022, bringing total global capacity to 57 GW.
    
    5. Green hydrogen production is projected to reach 24 million tons annually by 2030, up from 
       current levels of less than 1 million tons.
    
    Discussion:
    The data demonstrates a strong correlation between renewable energy deployment and carbon emission 
    reductions. Countries like Denmark generate 80% of their electricity from renewable sources, 
    primarily wind power. Norway produces 98% of its electricity from renewable sources, mainly 
    hydroelectric power.
    
    The economic benefits are equally compelling. The renewable energy sector employed 13.7 million 
    people worldwide in 2022, an increase of 1 million jobs from the previous year. Investment in 
    renewable energy reached $1.8 trillion globally in 2022.
    
    Conclusion:
    The transition to renewable energy is both environmentally necessary and economically viable. 
    To limit global warming to 1.5°C above pre-industrial levels, as outlined in the Paris Agreement, 
    renewable energy must comprise 90% of the global energy mix by 2050. Current trends suggest this 
    target is achievable with sustained policy support and continued technological innovation.
    """
    
    print("📄 Processing sample document...")
    print(f"Document length: {len(sample_document):,} characters")
    
    # Set up claim extraction pass
    print("\n🔧 Setting up claim extraction...")
    
    # For this example, we'll use mock responses since we might not have API keys
    # In a real scenario, you would set up actual LLM client with API keys
    
    # Check if we have real API keys
    has_openai_key = bool(os.getenv('OPENAI_API_KEY'))
    has_anthropic_key = bool(os.getenv('ANTHROPIC_API_KEY'))
    
    if has_openai_key or has_anthropic_key:
        print("✅ Found API keys, using real LLM client")
        claim_pass = ClaimExtractionPass()  # Will create default client from env vars
    else:
        print("⚠️  No API keys found, using mock client for demonstration")
        # We'll create a mock scenario for demonstration
        from unittest.mock import Mock, AsyncMock
        from llm.llm_client import LLMResponse
        
        # Create mock response with realistic claims
        mock_response_content = {
            "claims": [
                {
                    "claim_text": "Solar energy capacity has grown by 260% over the past decade",
                    "claim_type": "statistical",
                    "category": "environmental",
                    "extraction_confidence": 0.95,
                    "importance_score": 0.8,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 200,
                    "end_position": 260
                },
                {
                    "claim_text": "Wind power now accounts for approximately 24% of renewable electricity generation in OECD countries",
                    "claim_type": "statistical", 
                    "category": "environmental",
                    "extraction_confidence": 0.9,
                    "importance_score": 0.7,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 320,
                    "end_position": 420
                },
                {
                    "claim_text": "Global mean temperatures have increased by 1.1°C since the pre-industrial period",
                    "claim_type": "statistical",
                    "category": "environmental", 
                    "extraction_confidence": 0.95,
                    "importance_score": 0.9,
                    "citations": ["IPCC"],
                    "requires_fact_check": True,
                    "start_position": 600,
                    "end_position": 680
                },
                {
                    "claim_text": "Carbon dioxide levels in the atmosphere have reached 415 parts per million",
                    "claim_type": "statistical",
                    "category": "environmental",
                    "extraction_confidence": 0.92,
                    "importance_score": 0.85,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 700,
                    "end_position": 775
                },
                {
                    "claim_text": "Solar photovoltaic costs have decreased by 89% since 2010",
                    "claim_type": "statistical",
                    "category": "economic",
                    "extraction_confidence": 0.88,
                    "importance_score": 0.75,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 1200,
                    "end_position": 1260
                },
                {
                    "claim_text": "Electric vehicle sales surged to 10.5 million units globally in 2022",
                    "claim_type": "statistical",
                    "category": "environmental",
                    "extraction_confidence": 0.9,
                    "importance_score": 0.7,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 1400,
                    "end_position": 1470
                },
                {
                    "claim_text": "Denmark generates 80% of their electricity from renewable sources",
                    "claim_type": "statistical",
                    "category": "environmental",
                    "extraction_confidence": 0.85,
                    "importance_score": 0.65,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 1800,
                    "end_position": 1865
                },
                {
                    "claim_text": "The renewable energy sector employed 13.7 million people worldwide in 2022",
                    "claim_type": "statistical",
                    "category": "economic",
                    "extraction_confidence": 0.87,
                    "importance_score": 0.7,
                    "citations": [],
                    "requires_fact_check": True,
                    "start_position": 2100,
                    "end_position": 2175
                }
            ],
            "metadata": {
                "processing_notes": "Successfully extracted claims from climate change research paper",
                "extraction_warnings": ["Some claims may require additional context for verification"]
            }
        }
        
        mock_response = LLMResponse(
            content=json.dumps(mock_response_content),
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            usage={"tokens": 500},
            response_time_seconds=3.5,
            metadata={}
        )
        
        # Create mock LLM client
        mock_llm_client = Mock()
        mock_llm_client.generate_structured_response = AsyncMock(return_value=mock_response)
        mock_llm_client.get_available_providers.return_value = ["openai:gpt-4"]
        
        claim_pass = ClaimExtractionPass(llm_client=mock_llm_client)
    
    # Create verification context
    context = VerificationContext(
        document_id="climate_change_paper.pdf",
        document_content=sample_document
    )
    
    # Create pass configuration
    config = VerificationPassConfig(
        pass_type=VerificationPassType.CLAIM_EXTRACTION,
        name="extract_claims",
        parameters={
            "max_claims": 20,
            "model": "gpt-4",
            "prompt_version": "v1",
            "min_confidence": 0.6
        },
        enabled=True,
        timeout_seconds=60
    )
    
    # Execute claim extraction
    print("\n🚀 Executing claim extraction...")
    
    try:
        result = await claim_pass.execute(context, config)
        
        if result.status.value == "completed":
            print("✅ Claim extraction completed successfully!")
            
            # Parse the extraction result
            extraction_result = ClaimExtractionResult(**result.result_data["extraction_result"])
            
            # Display results
            print_claim_summary(extraction_result)
            print_claim_details(extraction_result.claims)
            print_statistics(extraction_result)
            
            # Save results to file
            output_file = Path("claim_extraction_results.json")
            with open(output_file, 'w') as f:
                json.dump(result.result_data, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_file}")
            
            # Additional analysis
            print_header("ADDITIONAL ANALYSIS")
            
            high_confidence_claims = [
                claim for claim in extraction_result.claims 
                if claim.extraction_confidence >= 0.9
            ]
            print(f"\n🎯 High-confidence claims (≥0.9): {len(high_confidence_claims)}")
            
            claims_with_citations = [
                claim for claim in extraction_result.claims
                if len(claim.citations) > 0
            ]
            print(f"📚 Claims with citations: {len(claims_with_citations)}")
            
            statistical_claims = [
                claim for claim in extraction_result.claims
                if claim.claim_type.value == "statistical"
            ]
            print(f"📊 Statistical claims: {len(statistical_claims)}")
            
            # Show most important claims
            important_claims = sorted(
                extraction_result.claims,
                key=lambda c: c.importance_score or 0,
                reverse=True
            )[:3]
            
            print(f"\n⭐ Top 3 most important claims:")
            for i, claim in enumerate(important_claims, 1):
                importance = claim.importance_score or 0
                print(f"   {i}. {claim.claim_text[:60]}... (importance: {importance:.2f})")
            
        else:
            print(f"❌ Claim extraction failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print_header("EXAMPLE COMPLETED")
    print("This example demonstrated:")
    print("• Setting up claim extraction with LLM integration")
    print("• Processing a scientific document")
    print("• Extracting and categorizing claims")
    print("• Analyzing extraction results and statistics")
    print("• Handling both real and mock LLM responses")


async def main():
    """Main entry point."""
    await run_claim_extraction_example()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())