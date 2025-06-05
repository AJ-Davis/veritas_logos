"""
Prompt templates for LLM-based verification tasks.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(str, Enum):
    """Types of prompts for different verification tasks."""
    CLAIM_EXTRACTION = "claim_extraction"
    CITATION_CHECK = "citation_check"
    LOGIC_ANALYSIS = "logic_analysis"
    BIAS_SCAN = "bias_scan"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    name: str
    version: str
    prompt_type: PromptType
    system_message: str
    user_template: str
    examples: List[Dict[str, str]]
    parameters: Dict[str, Any]


class ClaimExtractionPrompts:
    """Prompt templates for claim extraction."""
    
    SYSTEM_MESSAGE_V1 = """You are an expert document analyst specialized in identifying and extracting claims from text. Your task is to carefully read the provided document and extract all significant claims that could be verified or fact-checked.

DEFINITION OF A CLAIM:
A claim is a statement that asserts something as true or factual. Claims typically:
- Present facts, statistics, or data
- Make assertions about cause and effect
- Compare things or present rankings
- Make predictions about the future
- State policies or recommendations
- Define concepts or categorize things
- Make evaluative judgments

WHAT TO EXTRACT:
✓ Factual statements that can be verified
✓ Statistical claims and data points
✓ Causal assertions (X causes Y)
✓ Comparative statements (X is better than Y)
✓ Predictions and forecasts
✓ Policy recommendations
✓ Definitional claims
✓ Historical facts and events

WHAT NOT TO EXTRACT:
✗ Obvious statements or common knowledge
✗ Personal opinions without factual basis
✗ Questions or hypotheticals
✗ Purely subjective statements
✗ Citations or references themselves
✗ Procedural instructions
✗ Acknowledgments or meta-comments

CLASSIFICATION GUIDELINES:
- FACTUAL: Objective statements about reality
- STATISTICAL: Numbers, percentages, quantities
- CAUSAL: Claims about cause-and-effect relationships
- COMPARATIVE: Comparisons between entities
- PREDICTIVE: Forecasts or future-oriented claims
- DEFINITIONAL: Definitions or categorizations
- EVALUATIVE: Value judgments with factual basis
- POLICY: Recommendations or proposed actions
- EXISTENTIAL: Claims about existence or non-existence
- TEMPORAL: Time-based claims or sequences

CONFIDENCE SCORING:
- 0.9-1.0: Very clear, unambiguous claims
- 0.7-0.8: Clear claims with minor ambiguity
- 0.5-0.6: Somewhat unclear or implicit claims
- 0.3-0.4: Borderline claims with significant ambiguity
- 0.1-0.2: Very unclear or questionable claims

IMPORTANCE SCORING:
- 0.9-1.0: Central to document's main arguments
- 0.7-0.8: Important supporting claims
- 0.5-0.6: Moderately important details
- 0.3-0.4: Minor supporting information
- 0.1-0.2: Peripheral or trivial claims

Respond with a JSON object containing the extracted claims."""
    
    USER_TEMPLATE_V1 = """Please analyze the following document and extract all significant claims. For each claim, provide:

1. The exact claim text as it appears in the document
2. A normalized version if needed for clarity
3. Classification of the claim type
4. Location information (start/end positions)
5. Confidence score (0.0-1.0) for extraction accuracy
6. Importance score (0.0-1.0) for the claim's significance
7. Any citations or sources mentioned with the claim

Document text:
---
{document_text}
---

Extract up to {max_claims} claims. Focus on the most significant and verifiable statements."""
    
    EXAMPLES_V1 = [
        {
            "input": "Climate change has increased global temperatures by 1.1°C since pre-industrial times, according to NASA data.",
            "output": """[{
                "claim_text": "Climate change has increased global temperatures by 1.1°C since pre-industrial times",
                "claim_type": "statistical",
                "extraction_confidence": 0.95,
                "importance_score": 0.9,
                "citations": ["NASA data"]
            }]"""
        },
        {
            "input": "The new policy will reduce costs by approximately 15% over three years.",
            "output": """[{
                "claim_text": "The new policy will reduce costs by approximately 15% over three years",
                "claim_type": "predictive",
                "extraction_confidence": 0.9,
                "importance_score": 0.8,
                "citations": []
            }]"""
        }
    ]
    
    SYSTEM_MESSAGE_V2 = """You are a specialized claim extraction system designed to identify verifiable statements from documents with high precision and recall.

CORE PRINCIPLES:
1. PRECISION: Only extract statements that are genuinely claims requiring verification
2. COMPLETENESS: Capture all significant claims, including implicit ones
3. CONTEXT: Maintain enough context to understand each claim's meaning
4. CLASSIFICATION: Accurately categorize claims by type and importance

EXTRACTION STRATEGY:
1. Read the entire document first to understand context
2. Identify sentences or phrases that make factual assertions
3. Extract claims with sufficient context for verification
4. Classify each claim by type and assess its verifiability
5. Score confidence in extraction and importance to the document

QUALITY CHECKS:
- Can this claim be fact-checked against external sources?
- Is the claim specific enough to be verified or refuted?
- Does the claim contain sufficient context to be understood?
- Is this claim central to the document's argument or purpose?

Focus on extracting claims that would be most valuable for fact-checking and verification processes."""
    
    @classmethod
    def get_template(cls, version: str = "v1") -> PromptTemplate:
        """Get claim extraction prompt template by version."""
        
        if version == "v1":
            return PromptTemplate(
                name="claim_extraction",
                version="v1",
                prompt_type=PromptType.CLAIM_EXTRACTION,
                system_message=cls.SYSTEM_MESSAGE_V1,
                user_template=cls.USER_TEMPLATE_V1,
                examples=cls.EXAMPLES_V1,
                parameters={
                    "max_claims": 50,
                    "min_confidence": 0.3,
                    "include_context": True,
                    "extract_citations": True
                }
            )
        elif version == "v2":
            return PromptTemplate(
                name="claim_extraction",
                version="v2", 
                prompt_type=PromptType.CLAIM_EXTRACTION,
                system_message=cls.SYSTEM_MESSAGE_V2,
                user_template=cls.USER_TEMPLATE_V1,
                examples=cls.EXAMPLES_V1,
                parameters={
                    "max_claims": 50,
                    "min_confidence": 0.3,
                    "include_context": True,
                    "extract_citations": True
                }
            )
        else:
            raise ValueError(f"Unknown version: {version}")


class CitationCheckPrompts:
    """Prompt templates for citation verification."""
    
    SYSTEM_MESSAGE_V1 = """You are an expert citation validator. Your task is to verify whether citations properly support the claims they are associated with.

For each claim-citation pair, determine:
1. Whether the citation is accessible and real
2. Whether the citation content supports the claim
3. The strength of support (strong, moderate, weak, contradictory)
4. Any issues with the citation format or accessibility

Respond with detailed verification results."""
    
    @classmethod
    def get_template(cls, version: str = "v1") -> PromptTemplate:
        """Get citation check prompt template."""
        return PromptTemplate(
            name="citation_check",
            version=version,
            prompt_type=PromptType.CITATION_CHECK,
            system_message=cls.SYSTEM_MESSAGE_V1,
            user_template="Verify the following claim-citation pairs:\n{citation_pairs}",
            examples=[],
            parameters={}
        )


class LogicAnalysisPrompts:
    """Prompt templates for logical analysis."""
    
    SYSTEM_MESSAGE_V1 = """You are a logic and reasoning expert. Analyze the provided text for logical fallacies, invalid reasoning, and argument structure issues.

Identify:
- Logical fallacies (ad hominem, straw man, false dilemma, etc.)
- Invalid inferences and reasoning errors
- Contradictions within the text
- Unsupported conclusions
- Weak or missing evidence for claims

Provide specific examples and explanations for each issue found."""
    
    @classmethod
    def get_template(cls, version: str = "v1") -> PromptTemplate:
        """Get logic analysis prompt template."""
        return PromptTemplate(
            name="logic_analysis",
            version=version,
            prompt_type=PromptType.LOGIC_ANALYSIS,
            system_message=cls.SYSTEM_MESSAGE_V1,
            user_template="Analyze the logical structure of this text:\n{text}",
            examples=[],
            parameters={}
        )


class BiasDetectionPrompts:
    """Prompt templates for bias detection."""
    
    SYSTEM_MESSAGE_V1 = """You are a bias detection specialist. Analyze the provided text for various types of bias including:

TYPES OF BIAS TO DETECT:
- Selection bias in data or examples
- Confirmation bias in argument construction
- Cultural or demographic bias
- Political or ideological bias
- Statistical bias in data presentation
- Framing bias in how issues are presented
- Attribution bias in explanations

For each bias detected, provide:
- Type of bias
- Specific examples from the text
- Severity assessment
- Potential impact on conclusions"""
    
    @classmethod
    def get_template(cls, version: str = "v1") -> PromptTemplate:
        """Get bias detection prompt template."""
        return PromptTemplate(
            name="bias_detection", 
            version=version,
            prompt_type=PromptType.BIAS_SCAN,
            system_message=cls.SYSTEM_MESSAGE_V1,
            user_template="Analyze this text for potential bias:\n{text}",
            examples=[],
            parameters={}
        )


class PromptManager:
    """Manager for prompt templates."""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        self.templates["claim_extraction:v1"] = ClaimExtractionPrompts.get_template("v1")
        self.templates["claim_extraction:v2"] = ClaimExtractionPrompts.get_template("v2")
        self.templates["citation_check:v1"] = CitationCheckPrompts.get_template("v1")
        self.templates["logic_analysis:v1"] = LogicAnalysisPrompts.get_template("v1")
        self.templates["bias_detection:v1"] = BiasDetectionPrompts.get_template("v1")
    
    def get_template(self, prompt_type: PromptType, version: str = "v1") -> PromptTemplate:
        """Get a prompt template by type and version."""
        key = f"{prompt_type.value}:{version}"
        if key not in self.templates:
            raise ValueError(f"Template not found: {key}")
        return self.templates[key]
    
    def register_template(self, template: PromptTemplate):
        """Register a custom prompt template."""
        key = f"{template.prompt_type.value}:{template.version}"
        self.templates[key] = template
    
    def list_templates(self) -> List[str]:
        """List all available template keys."""
        return list(self.templates.keys())
    
    def create_messages(self, template: PromptTemplate, **kwargs) -> List[Dict[str, str]]:
        """Create chat messages from a template."""
        messages = [
            {"role": "system", "content": template.system_message}
        ]
        
        # Add examples if provided
        for example in template.examples:
            if "input" in example and "output" in example:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})
        
        # Add the actual user message
        user_content = template.user_template.format(**kwargs)
        messages.append({"role": "user", "content": user_content})
        
        return messages


# Global prompt manager instance
prompt_manager = PromptManager()