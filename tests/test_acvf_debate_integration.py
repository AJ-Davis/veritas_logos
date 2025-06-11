"""
Comprehensive integration tests for the Adversarial Cross-Validation Framework (ACVF) debate system.

This test suite focuses on end-to-end debate workflows, multi-round scenarios,
integration with the verification pipeline, and real-world debate scenarios.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional

# Import ACVF components
from src.models.acvf import (
    ACVFRole, DebateStatus, JudgeVerdict, ConfidenceLevel,
    ModelAssignment, DebateArgument, JudgeScore, DebateRound,
    ACVFConfiguration, ACVFResult
)
from src.verification.acvf_controller import ACVFController
from src.models.verification import VerificationContext, VerificationResult, VerificationStatus, VerificationPassType
from src.llm.llm_client import LLMClient, LLMResponse, LLMProvider


@pytest.fixture
def mock_llm_responses():
    """Predefined LLM responses for consistent testing."""
    return {
        "challenger_responses": [
            "This claim lacks sufficient empirical evidence. The study cited has methodological flaws including small sample size and potential selection bias.",
            "The data interpretation is questionable. Alternative explanations for the observed phenomenon have not been adequately considered.",
            "The conclusion overreaches based on the evidence presented. Correlation does not imply causation in this context."
        ],
        "defender_responses": [
            "The methodology follows established scientific protocols. The sample size is appropriate for this type of study and meets statistical power requirements.",
            "Multiple independent studies have replicated these findings. The preponderance of evidence supports the conclusion.",
            "The causal mechanism is well-established through previous research. This study adds confirmatory evidence to the existing body of knowledge."
        ],
        "judge_responses": [
            {
                "verdict": "defender_wins",
                "confidence": 0.75,
                "reasoning": "The defender provided more robust evidence and addressed the challenger's concerns adequately.",
                "challenger_score": 0.3,
                "defender_score": 0.8
            },
            {
                "verdict": "challenger_wins", 
                "confidence": 0.65,
                "reasoning": "The challenger identified significant methodological concerns that were not sufficiently addressed.",
                "challenger_score": 0.7,
                "defender_score": 0.4
            },
            {
                "verdict": "tie",
                "confidence": 0.55,
                "reasoning": "Both sides presented valid arguments. More evidence needed for conclusive determination.",
                "challenger_score": 0.6,
                "defender_score": 0.6
            }
        ]
    }


@pytest.fixture
def debate_test_documents():
    """Test documents with varying levels of controversy and complexity."""
    return {
        "climate_research": {
            "content": """Climate Change Research Study: Global Temperature Trends

            A comprehensive analysis of global temperature data from 1880-2023 shows a consistent warming trend of 1.1°C above pre-industrial levels. The study analyzed temperature records from 15,000 weather stations worldwide and employed advanced statistical methods to account for urban heat island effects and data quality variations.

            Key findings:
            1. The warming trend has accelerated since 1980, with the last decade being the warmest on record
            2. Arctic regions show amplified warming at 2.5 times the global average
            3. Ocean temperature measurements confirm atmospheric warming trends
            4. Statistical analysis shows less than 0.01% probability that observed warming is due to natural variation

            Citations:
            - NOAA Global Temperature Anomalies Dataset (2023)
            - Hansen et al. (2010) Global Surface Temperature Change, Reviews of Geophysics
            - IPCC Sixth Assessment Report (2021)

            The evidence strongly supports anthropogenic climate change as the primary driver of observed warming.
            """,
            "controversy_level": "medium",
            "expected_issues": ["methodology questions", "data interpretation debates"]
        },
        "vaccine_efficacy": {
            "content": """COVID-19 Vaccine Efficacy Meta-Analysis

            A meta-analysis of 15 randomized controlled trials involving 150,000 participants demonstrates COVID-19 vaccine efficacy of 91% against symptomatic infection and 96% against severe disease. The analysis included data from mRNA vaccines (Pfizer-BioNTech, Moderna) and viral vector vaccines (Johnson & Johnson, AstraZeneca).

            Primary endpoints:
            - Symptomatic COVID-19 prevention: 91% efficacy (95% CI: 88-94%)
            - Severe disease prevention: 96% efficacy (95% CI: 93-98%)
            - Hospitalization prevention: 97% efficacy (95% CI: 95-99%)

            Safety profile showed mild to moderate side effects in 60% of recipients, with serious adverse events occurring at the same rate as placebo groups.

            The study concludes that COVID-19 vaccines provide substantial protection against infection and severe outcomes.
            """,
            "controversy_level": "high",
            "expected_issues": ["safety concerns", "efficacy disputes", "bias allegations"]
        },
        "economic_policy": {
            "content": """Impact of Minimum Wage Increases on Employment

            A longitudinal study of 50 metropolitan areas over 20 years examined the relationship between minimum wage increases and employment levels. The study used a difference-in-differences methodology to compare areas with varying minimum wage policies.

            Results show:
            - Small employment reductions (2-4%) in low-skilled sectors following wage increases above $15/hour
            - Increased earnings for workers who remain employed offset employment losses
            - No significant impact on overall metropolitan employment rates
            - Heterogeneous effects across industries, with food service showing larger employment responses

            Policy implications suggest moderate minimum wage increases have mixed but generally positive welfare effects.
            """,
            "controversy_level": "high", 
            "expected_issues": ["economic methodology", "policy interpretation", "ideological bias"]
        },
        "technical_spec": {
            "content": """Quantum Computing Error Correction Protocol

            Technical specification for a new quantum error correction code that achieves a logical error rate of 10^-15 using 1000 physical qubits. The protocol employs surface code topology with optimized syndrome extraction circuits.

            Performance characteristics:
            - Physical qubit error rate threshold: 10^-3
            - Code distance: 31
            - Syndrome extraction time: 1.2μs
            - Logical gate set: Clifford + T gates
            - Fault-tolerant threshold: 99.9%

            Implementation requires specialized control electronics and calibrated pulse sequences for multi-qubit gate operations.
            """,
            "controversy_level": "low",
            "expected_issues": ["technical accuracy", "feasibility questions"]
        }
    }


@pytest.fixture  
def acvf_integration_config():
    """ACVF configuration optimized for integration testing."""
    challenger = ModelAssignment(
        model_id="claude-3-opus",
        provider="anthropic",
        role=ACVFRole.CHALLENGER,
        temperature=0.8,
        max_tokens=1500
    )
    
    defender = ModelAssignment(
        model_id="gpt-4",
        provider="openai", 
        role=ACVFRole.DEFENDER,
        temperature=0.7,
        max_tokens=1500
    )
    
    judge1 = ModelAssignment(
        model_id="claude-3-opus",
        provider="anthropic",
        role=ACVFRole.JUDGE,
        temperature=0.3,
        max_tokens=2000
    )
    
    judge2 = ModelAssignment(
        model_id="gpt-4",
        provider="openai",
        role=ACVFRole.JUDGE,
        temperature=0.3,
        max_tokens=2000
    )
    
    return ACVFConfiguration(
        config_id="integration_test_config",
        name="Integration Test Configuration",
        description="Configuration for ACVF integration testing",
        challenger_models=[challenger],
        defender_models=[defender],
        judge_models=[judge1, judge2],
        max_rounds_per_debate=3,
        escalation_threshold=0.6,
        consensus_threshold=0.7,
        trigger_conditions={
            "min_confidence_threshold": 0.6,
            "escalate_failed_passes": True,
            "escalate_on_issues": True
        },
        allow_model_self_assignment=False
    )


class MockLLMClientWithDelays:
    """Mock LLM client that simulates realistic response times and behaviors."""
    
    def __init__(self, responses: Dict[str, Any], base_delay: float = 0.1):
        self.responses = responses
        self.base_delay = base_delay
        self.call_count = 0
        self.challenger_call_count = 0
        self.defender_call_count = 0
        self.judge_call_count = 0
    
    async def generate_challenger_response(self, subject_content: str, context: str, 
                                         provider: str, **kwargs) -> LLMResponse:
        """Generate challenger response with simulated delay."""
        await asyncio.sleep(self.base_delay + 0.05)  # Slight random delay
        
        responses = self.responses.get("challenger_responses", ["Generic challenger response"])
        response_idx = self.challenger_call_count % len(responses)
        content = responses[response_idx]
        
        self.challenger_call_count += 1
        self.call_count += 1
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC if "anthropic" in provider else LLMProvider.OPENAI,
            model=provider.split(":")[1] if ":" in provider else provider,
            usage={"prompt_tokens": 50, "completion_tokens": len(content.split()), "total_tokens": 50 + len(content.split())},
            response_time_seconds=self.base_delay + 0.05,
            metadata={"confidence": 0.8}
        )
    
    async def generate_defender_response(self, subject_content: str, challenger_arguments: str,
                                       context: str, provider: str, **kwargs) -> LLMResponse:
        """Generate defender response with simulated delay."""
        await asyncio.sleep(self.base_delay + 0.07)  # Defender takes slightly longer
        
        responses = self.responses.get("defender_responses", ["Generic defender response"])
        response_idx = self.defender_call_count % len(responses)
        content = responses[response_idx]
        
        self.defender_call_count += 1
        self.call_count += 1
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC if "anthropic" in provider else LLMProvider.OPENAI,
            model=provider.split(":")[1] if ":" in provider else provider,
            usage={"prompt_tokens": 60, "completion_tokens": len(content.split()), "total_tokens": 60 + len(content.split())},
            response_time_seconds=self.base_delay + 0.07,
            metadata={"confidence": 0.75}
        )
    
    async def generate_judge_response(self, subject_content: str, challenger_arguments: str,
                                    defender_arguments: str, context: str, provider: str, **kwargs) -> LLMResponse:
        """Generate judge response with simulated delay."""
        await asyncio.sleep(self.base_delay + 0.1)  # Judges take longest
        
        responses = self.responses.get("judge_responses", [])
        if responses:
            response_idx = self.judge_call_count % len(responses)
            judge_data = responses[response_idx]
            
            # Format as structured judge response
            content = json.dumps({
                "verdict": judge_data["verdict"],
                "confidence": judge_data["confidence"],
                "reasoning": judge_data["reasoning"],
                "challenger_score": judge_data["challenger_score"],
                "defender_score": judge_data["defender_score"],
                "key_points_challenger": ["Point 1", "Point 2"],
                "key_points_defender": ["Point A", "Point B"],
                "critical_weaknesses": ["Weakness identified"]
            })
        else:
            content = json.dumps({
                "verdict": "tie",
                "confidence": 0.5,
                "reasoning": "Generic judge response",
                "challenger_score": 0.5,
                "defender_score": 0.5
            })
        
        self.judge_call_count += 1
        self.call_count += 1
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC if "anthropic" in provider else LLMProvider.OPENAI,
            model=provider.split(":")[1] if ":" in provider else provider,
            usage={"prompt_tokens": 80, "completion_tokens": len(content.split()), "total_tokens": 80 + len(content.split())},
            response_time_seconds=self.base_delay + 0.1,
            metadata={"confidence": 0.9}
        )
    
    async def generate_judge_verdict(self, subject_content: str, challenger_arguments: str,
                                   defender_arguments: str, context: str, provider: str, **kwargs) -> LLMResponse:
        """Generate judge verdict with simulated delay."""
        # Use the same implementation as generate_judge_response for compatibility
        return await self.generate_judge_response(subject_content, challenger_arguments, 
                                                defender_arguments, context, provider, **kwargs)


@pytest.mark.asyncio
class TestACVFDebateWorkflows:
    """Test complete ACVF debate workflows and integration scenarios."""
    
    async def test_single_round_debate_workflow(self, acvf_integration_config, mock_llm_responses):
        """Test a complete single-round debate from start to finish."""
        mock_client = MockLLMClientWithDelays(mock_llm_responses)
        controller = ACVFController(mock_client, acvf_integration_config)
        
        # Conduct single debate round
        start_time = time.time()
        debate_round = await controller.conduct_debate_round(
            verification_task_id="test_task_001",
            subject_type="claim",
            subject_id="claim_001",
            subject_content="Climate change is accelerating due to human activities",
            context="Scientific consensus and recent temperature data",
            round_number=1
        )
        end_time = time.time()
        
        # Verify debate round completion
        assert debate_round.status == DebateStatus.COMPLETED
        assert debate_round.final_verdict is not None
        assert debate_round.consensus_confidence is not None
        assert len(debate_round.arguments) >= 2  # At least challenger and defender
        assert len(debate_round.judge_scores) >= 1
        
        # Verify timing
        assert debate_round.total_duration_seconds is not None
        assert debate_round.total_duration_seconds > 0
        assert end_time - start_time >= 0.3  # Should take at least some time due to delays
        
        # Verify argument structure
        challenger_args = debate_round.get_arguments_by_role(ACVFRole.CHALLENGER)
        defender_args = debate_round.get_arguments_by_role(ACVFRole.DEFENDER)
        assert len(challenger_args) >= 1
        assert len(defender_args) >= 1
        
        # Verify judge scores
        for score in debate_round.judge_scores:
            assert 0 <= score.confidence <= 1
            assert 0 <= score.challenger_score <= 1
            assert 0 <= score.defender_score <= 1
            assert score.reasoning is not None
            assert len(score.reasoning) > 0
        
        print(f"✅ Single round debate completed in {debate_round.total_duration_seconds:.2f}s")
        print(f"   Final verdict: {debate_round.final_verdict}")
        print(f"   Consensus confidence: {debate_round.consensus_confidence:.2f}")

    async def test_multi_round_debate_escalation(self, acvf_integration_config, mock_llm_responses):
        """Test multi-round debate with escalation scenarios."""
        # Modify responses to create escalation scenario
        escalation_responses = mock_llm_responses.copy()
        escalation_responses["judge_responses"] = [
            {
                "verdict": "tie",
                "confidence": 0.4,  # Low confidence triggers escalation
                "reasoning": "Arguments are evenly matched, need more evidence",
                "challenger_score": 0.5,
                "defender_score": 0.5
            },
            {
                "verdict": "challenger_wins",
                "confidence": 0.7,  # Higher confidence in second round
                "reasoning": "Challenger provided stronger evidence in follow-up",
                "challenger_score": 0.8,
                "defender_score": 0.4
            }
        ]
        
        mock_client = MockLLMClientWithDelays(escalation_responses)
        controller = ACVFController(mock_client, acvf_integration_config)
        
        # Create verification context that should trigger escalation
        verification_context = VerificationContext(
            document_id="doc_001",
            document_content="Controversial claim about vaccine efficacy",
            document_metadata={"source": "research_paper"},
            previous_results=[
                            VerificationResult(
                pass_id="bias_scan_001",
                pass_type=VerificationPassType.BIAS_SCAN,
                status=VerificationStatus.COMPLETED,
                started_at=datetime.now(timezone.utc),
                confidence_score=0.3,  # Low confidence
                result_data={"bias_indicators": ["confirmation_bias", "cherry_picking"]}
            )
            ]
        )
        
        # Conduct full debate
        acvf_result = await controller.conduct_full_debate(
            verification_context=verification_context,
            subject_type="claim",
            subject_id="controversial_claim_001",
            subject_content="COVID-19 vaccines show 95% efficacy in preventing severe disease"
        )
        
        # Verify multi-round behavior
        assert len(acvf_result.debate_rounds) >= 1
        assert acvf_result.total_rounds >= 1
        assert acvf_result.final_verdict is not None
        assert acvf_result.consensus_achieved is not None
        
        # Check for escalation logic
        if len(acvf_result.debate_rounds) > 1:
            first_round = acvf_result.debate_rounds[0]
            second_round = acvf_result.debate_rounds[1]
            
            # Second round should reference first round
            assert second_round.round_number > first_round.round_number
            assert len(second_round.arguments) >= 2
        
        print(f"✅ Multi-round debate completed with {acvf_result.total_rounds} rounds")
        print(f"   Final verdict: {acvf_result.final_verdict}")
        print(f"   Consensus achieved: {acvf_result.consensus_achieved}")

    async def test_acvf_trigger_conditions(self, acvf_integration_config, mock_llm_responses):
        """Test various ACVF trigger conditions."""
        mock_client = MockLLMClientWithDelays(mock_llm_responses)
        controller = ACVFController(mock_client, acvf_integration_config)
        
        # Test 1: Low confidence trigger
        low_confidence_context = VerificationContext(
            document_id="doc_low_conf",
            document_content="Ambiguous claim",
            document_metadata={},
            previous_results=[
                            VerificationResult(
                pass_id="claim_extraction_001",
                pass_type=VerificationPassType.CLAIM_EXTRACTION,
                status=VerificationStatus.COMPLETED,
                started_at=datetime.now(timezone.utc),
                confidence_score=0.3,  # Below threshold
                result_data={"claims": ["weak claim"]}
            )
            ]
        )
        
        should_trigger_low_conf = await controller.should_trigger_acvf(low_confidence_context)
        assert should_trigger_low_conf == True
        
        # Test 2: Failed pass trigger
        failed_pass_context = VerificationContext(
            document_id="doc_failed",
            document_content="Problematic content",
            document_metadata={},
            previous_results=[
                VerificationResult(
                    pass_id="citation_check_001",
                    pass_type=VerificationPassType.CITATION_CHECK,
                    status=VerificationStatus.FAILED,
                    started_at=datetime.now(timezone.utc),
                    confidence_score=0.8,  # High confidence but failed
                    result_data={"citation_errors": ["broken_link", "invalid_source"]}
                )
            ]
        )
        
        should_trigger_failed = await controller.should_trigger_acvf(failed_pass_context)
        assert should_trigger_failed == True
        
        # Test 3: High confidence, no issues - should not trigger
        high_confidence_context = VerificationContext(
            document_id="doc_high_conf",
            document_content="Well-verified content",
            document_metadata={},
            previous_results=[
                VerificationResult(
                    pass_id="logic_analysis_001",
                    pass_type=VerificationPassType.LOGIC_ANALYSIS,
                    status=VerificationStatus.COMPLETED,
                    started_at=datetime.now(timezone.utc),
                    confidence_score=0.9,  # High confidence
                    result_data={"logic_score": 0.95}
                )
            ]
        )
        
        should_trigger_high_conf = await controller.should_trigger_acvf(high_confidence_context)
        assert should_trigger_high_conf == False
        
        print("✅ ACVF trigger condition tests passed")
        print(f"   Low confidence triggers: {should_trigger_low_conf}")
        print(f"   Failed pass triggers: {should_trigger_failed}")
        print(f"   High confidence triggers: {should_trigger_high_conf}")


@pytest.mark.asyncio 
class TestACVFErrorHandling:
    """Test ACVF error handling and resilience."""
    
    async def test_llm_failure_handling(self, acvf_integration_config):
        """Test handling of LLM service failures."""
        # Create mock client that fails
        failing_client = AsyncMock()
        failing_client.generate_challenger_response.side_effect = Exception("LLM service unavailable")
        failing_client.generate_defender_response.side_effect = Exception("LLM service unavailable")
        failing_client.generate_judge_response.side_effect = Exception("LLM service unavailable")
        
        controller = ACVFController(failing_client, acvf_integration_config)
        
        # Attempt debate round - should handle gracefully
        with pytest.raises(Exception) as exc_info:
            await controller.conduct_debate_round(
                verification_task_id="failure_test",
                subject_type="claim",
                subject_id="fail_claim",
                subject_content="Test claim for failure handling",
                context="Testing error resilience",
                round_number=1
            )
        
        assert "LLM service unavailable" in str(exc_info.value)
        print("✅ LLM failure handling test completed")
    
    async def test_invalid_configuration_handling(self, mock_llm_responses):
        """Test handling of invalid ACVF configurations."""
        mock_client = MockLLMClientWithDelays(mock_llm_responses)
        
        # Test 1: No challenger models
        with pytest.raises(ValueError) as exc_info:
            invalid_config = ACVFConfiguration(
                config_id="invalid_no_challenger",
                name="Invalid Config",
                challenger_models=[],  # Empty list
                defender_models=[ModelAssignment(
                    model_id="gpt-4", provider="openai", role=ACVFRole.DEFENDER
                )],
                judge_models=[ModelAssignment(
                    model_id="claude-3-opus", provider="anthropic", role=ACVFRole.JUDGE
                )]
            )
            controller = ACVFController(mock_client, invalid_config)
        
        assert "challenger_models" in str(exc_info.value).lower() or "too_short" in str(exc_info.value).lower()
        
        print("✅ Invalid configuration handling tests passed")


@pytest.mark.asyncio
class TestACVFPerformanceScenarios:
    """Test ACVF performance under various scenarios."""
    
    async def test_concurrent_debates(self, acvf_integration_config, mock_llm_responses):
        """Test multiple concurrent debates."""
        mock_client = MockLLMClientWithDelays(mock_llm_responses, base_delay=0.05)
        controller = ACVFController(mock_client, acvf_integration_config)
        
        # Create multiple debate tasks
        debate_tasks = []
        for i in range(3):
            task = controller.conduct_debate_round(
                verification_task_id=f"concurrent_test_{i}",
                subject_type="claim",
                subject_id=f"concurrent_claim_{i}",
                subject_content=f"Test claim {i} for concurrent processing",
                context=f"Concurrent test context {i}",
                round_number=1
            )
            debate_tasks.append(task)
        
        # Run debates concurrently
        start_time = time.time()
        results = await asyncio.gather(*debate_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify all completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 3
        
        # Verify concurrent execution was faster than sequential
        total_duration = end_time - start_time
        assert total_duration < 2.0  # Should be much faster than 3 sequential debates
        
        print(f"✅ Concurrent debates test completed in {total_duration:.2f}s")
        print(f"   Successful debates: {len(successful_results)}/3")
    
    async def test_debate_complexity_scaling(self, acvf_integration_config, debate_test_documents):
        """Test debate performance with documents of varying complexity."""
        performance_results = {}
        
        for doc_type, doc_data in debate_test_documents.items():
            mock_responses = {
                "challenger_responses": [
                    f"Challenging the {doc_type}: The methodology appears flawed...",
                    f"Additional concerns about {doc_type}: Data interpretation issues..."
                ],
                "defender_responses": [
                    f"Defending {doc_type}: The approach follows established protocols...",
                    f"Further defense of {doc_type}: Independent validation supports findings..."
                ],
                "judge_responses": [
                    {
                        "verdict": "defender_wins",
                        "confidence": 0.75,
                        "reasoning": f"Analysis of {doc_type} shows robust methodology",
                        "challenger_score": 0.4,
                        "defender_score": 0.8
                    }
                ]
            }
            
            mock_client = MockLLMClientWithDelays(mock_responses)
            controller = ACVFController(mock_client, acvf_integration_config)
            
            start_time = time.time()
            debate_round = await controller.conduct_debate_round(
                verification_task_id=f"complexity_test_{doc_type}",
                subject_type="document",
                subject_id=f"doc_{doc_type}",
                subject_content=doc_data["content"],
                context=f"Analyzing {doc_type} document complexity",
                round_number=1
            )
            end_time = time.time()
            
            performance_results[doc_type] = {
                "duration": end_time - start_time,
                "content_length": len(doc_data["content"]),
                "complexity_level": doc_data["controversy_level"],
                "final_verdict": debate_round.final_verdict,
                "argument_count": len(debate_round.arguments)
            }
        
        # Verify all debates completed
        assert len(performance_results) == len(debate_test_documents)
        
        # Analysis
        print("✅ Document complexity scaling test completed")
        for doc_type, metrics in performance_results.items():
            print(f"   {doc_type}: {metrics['duration']:.2f}s, "
                  f"{metrics['content_length']} chars, "
                  f"{metrics['complexity_level']} complexity")


if __name__ == "__main__":
    # Run specific test suites
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test run
        pytest.main([__file__ + "::TestACVFDebateWorkflows::test_single_round_debate_workflow", "-v"])
    else:
        # Full test suite
        pytest.main([__file__, "-v", "--tb=short"]) 