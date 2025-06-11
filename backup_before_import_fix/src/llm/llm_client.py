"""
Unified LLM client for making API calls to different providers.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import openai
import anthropic
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Import ACVF role definitions
from ..models.acvf import ACVFRole

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, Any]
    response_time_seconds: float
    metadata: Dict[str, Any]


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"llm.{config.provider.value}")
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate a structured response (JSON) from the LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                timeout=self.config.timeout_seconds
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model=self.config.model,
                usage=(response.usage.model_dump()  # v1 SDK
                       if hasattr(response.usage, "model_dump")
                       else dict(response.usage)              # fallback
                       if response and getattr(response, "usage", None)
                       else {}),
                response_time_seconds=response_time,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured JSON response using OpenAI API."""
        
        # Add JSON schema instruction to the system message
        schema_instruction = f"""
Please respond with valid JSON that matches this schema:
{json.dumps(response_schema, indent=2)}

Ensure your response is valid JSON and follows the schema exactly.
"""
        
        # Modify messages to include schema instruction
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += "\n\n" + schema_instruction
        else:
            enhanced_messages.insert(0, {"role": "system", "content": schema_instruction})
        
        return await self.generate_response(enhanced_messages, **kwargs)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key)
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            # Convert messages format for Anthropic while preserving chronological order
            # Anthropic API requires system messages to be handled separately, but we need
            # to preserve the chronological order by embedding system messages as user messages
            # when they appear mid-conversation
            
            system_message = ""
            anthropic_messages = []
            
            # First pass: collect initial system messages (before any user/assistant messages)
            initial_system_messages = []
            first_non_system_index = 0
            
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    initial_system_messages.append(msg["content"])
                    first_non_system_index = i + 1
                else:
                    break
            
            # Combine initial system messages for the system parameter
            if initial_system_messages:
                system_message = "\n".join(initial_system_messages)
            
            # Second pass: process remaining messages in chronological order
            for msg in messages[first_non_system_index:]:
                if msg["role"] == "system":
                    # For system messages that appear mid-conversation, 
                    # convert them to user messages to preserve chronological order
                    anthropic_messages.append({
                        "role": "user",
                        "content": f"[SYSTEM INSTRUCTION]: {msg['content']}"
                    })
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=self.config.model,
                system=system_message.strip() if system_message else None,
                messages=anthropic_messages,
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                timeout=self.config.timeout_seconds
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProvider.ANTHROPIC,
                model=self.config.model,
                usage=response.usage if hasattr(response, "usage") else {},
                response_time_seconds=response_time,
                metadata={
                    "finish_reason": response.stop_reason if hasattr(response, "stop_reason") else None,
                    "response_id": response.id if hasattr(response, "id") else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured JSON response using Anthropic API."""
        
        # Add JSON schema instruction to the system message
        schema_instruction = f"""
Please respond with valid JSON that matches this schema:
{json.dumps(response_schema, indent=2)}

Ensure your response is valid JSON and follows the schema exactly.
"""
        
        # Modify messages to include schema instruction
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += "\n\n" + schema_instruction
        else:
            enhanced_messages.insert(0, {"role": "system", "content": schema_instruction})
        
        return await self.generate_response(enhanced_messages, **kwargs)


class LLMClient:
    """Unified client for different LLM providers with ACVF role support."""
    
    def __init__(self, configs: List[LLMConfig]):
        self.providers = {}
        for config in configs:
            if config.provider == LLMProvider.OPENAI:
                provider = OpenAIProvider(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                provider = AnthropicProvider(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            # Use provider type and model as key for multiple configs of same provider
            key = f"{config.provider.value}:{config.model}"
            self.providers[key] = provider
        
        if not self.providers:
            raise ValueError("At least one provider configuration must be provided")
        
        # Default to first provider if none specified
        self.default_provider_key = list(self.providers.keys())[0]
        self.logger = logging.getLogger("llm.client")
    
    def get_provider(self, provider: Optional[str] = None) -> BaseLLMProvider:
        """Get provider instance by key."""
        if provider is None:
            provider = self.default_provider_key
        
        if provider not in self.providers:
            # Try to find by provider type only
            matching_keys = [key for key in self.providers.keys() if key.startswith(f"{provider}:")]
            if matching_keys:
                provider = matching_keys[0]
            else:
                raise ValueError(f"Provider not found: {provider}. Available: {list(self.providers.keys())}")
        
        return self.providers[provider]
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using specified or default provider."""
        llm_provider = self.get_provider(provider)
        
        for attempt in range(llm_provider.config.max_retries):
            try:
                return await llm_provider.generate_response(messages, **kwargs)
            except Exception as e:
                if attempt == llm_provider.config.max_retries - 1:
                    self.logger.error(f"All retry attempts failed. Last error: {str(e)}")
                    raise
                
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{llm_provider.config.max_retries}): {str(e)}")
                await asyncio.sleep(llm_provider.config.retry_delay_seconds * (2 ** attempt))  # Exponential backoff
    
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any],
                                         provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate structured JSON response using specified or default provider."""
        llm_provider = self.get_provider(provider)
        
        for attempt in range(llm_provider.config.max_retries):
            try:
                return await llm_provider.generate_structured_response(messages, response_schema, **kwargs)
            except Exception as e:
                if attempt == llm_provider.config.max_retries - 1:
                    self.logger.error(f"All retry attempts failed. Last error: {str(e)}")
                    raise
                
                self.logger.warning(f"Structured request failed (attempt {attempt + 1}/{llm_provider.config.max_retries}): {str(e)}")
                await asyncio.sleep(llm_provider.config.retry_delay_seconds * (2 ** attempt))
    
    async def generate_with_fallback(self, messages: List[Dict[str, str]], 
                                   preferred_provider: Optional[str] = None,
                                   fallback_providers: Optional[List[str]] = None,
                                   **kwargs) -> LLMResponse:
        """Generate response with fallback to other providers on failure."""
        providers_to_try = []
        
        # Add preferred provider first
        if preferred_provider:
            providers_to_try.append(preferred_provider)
        
        # Add fallback providers
        if fallback_providers:
            providers_to_try.extend(fallback_providers)
        
        # Add all available providers as final fallback
        for key in self.providers.keys():
            if key not in providers_to_try:
                providers_to_try.append(key)
        
        last_error = None
        for provider_key in providers_to_try:
            try:
                self.logger.info(f"Attempting to use provider: {provider_key}")
                return await self.generate_response(messages, provider=provider_key, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider_key} failed: {str(e)}")
                continue
        
        # If we get here, all providers failed
        raise Exception(f"All providers failed. Last error: {str(last_error)}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider keys."""
        return list(self.providers.keys())
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured providers."""
        return {
            key: {
                "provider": provider.config.provider.value,
                "model": provider.config.model,
                "temperature": provider.config.temperature,
                "max_tokens": provider.config.max_tokens
            }
            for key, provider in self.providers.items()
        }
    
    # ACVF-specific methods
    async def generate_challenger_response(self, subject_content: str, context: str,
                                         provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate a challenger argument for ACVF debate."""
        system_prompt = """You are a Challenger in an adversarial verification system. Your role is to:

1. Critically analyze the given content for potential issues, errors, or weaknesses
2. Question claims that lack sufficient evidence or reasoning
3. Identify logical fallacies, biases, or inconsistencies
4. Propose alternative interpretations or contradictory evidence
5. Be thorough but fair in your criticism

Focus on finding genuine problems rather than being argumentative for its own sake.
Your goal is to improve the overall quality of information verification."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Content to challenge:\n{subject_content}\n\nContext:\n{context}\n\nProvide your challenger argument:"}
        ]
        
        return await self.generate_response(messages, provider=provider, **kwargs)
    
    async def generate_defender_response(self, subject_content: str, challenger_arguments: str, 
                                       context: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate a defender response for ACVF debate."""
        system_prompt = """You are a Defender in an adversarial verification system. Your role is to:

1. Defend the accuracy and validity of the given content
2. Provide evidence and reasoning to counter challenger arguments
3. Clarify misunderstandings or misinterpretations
4. Acknowledge legitimate concerns while defending what is correct
5. Provide additional supporting evidence when available

Be honest about limitations but strong in defending valid content.
Your goal is to ensure accurate information is properly validated."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Content to defend:\n{subject_content}\n\nChallenger arguments:\n{challenger_arguments}\n\nContext:\n{context}\n\nProvide your defense:"}
        ]
        
        return await self.generate_response(messages, provider=provider, **kwargs)
    
    async def generate_judge_verdict(self, subject_content: str, challenger_arguments: str,
                                   defender_arguments: str, context: str,
                                   provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate a judge verdict for ACVF debate."""
        from ..models.acvf import JudgeVerdict, ConfidenceLevel
        
        system_prompt = """You are a Judge in an adversarial verification system. Your role is to:

1. Objectively evaluate both challenger and defender arguments
2. Assess the strength of evidence and reasoning from both sides
3. Identify which arguments are most compelling and accurate
4. Provide a fair verdict based on the quality of arguments and evidence
5. Explain your reasoning clearly and thoroughly

Be impartial and focus on factual accuracy and logical strength.
Your verdict will help determine the final verification outcome."""

        # Define the expected response schema for structured output
        response_schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["challenger_wins", "defender_wins", "tie", "insufficient_evidence", "invalid_debate"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "challenger_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "defender_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": "string"
                },
                "key_points_challenger": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "key_points_defender": {
                    "type": "array", 
                    "items": {"type": "string"}
                },
                "critical_weaknesses": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["verdict", "confidence", "challenger_score", "defender_score", "reasoning"]
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Content being debated:
{subject_content}

Challenger arguments:
{challenger_arguments}

Defender arguments:
{defender_arguments}

Context:
{context}

Please provide your verdict as a JSON response following the required schema."""}
        ]
        
        return await self.generate_structured_response(messages, response_schema, provider=provider, **kwargs)
    
    async def generate_role_response(self, role: ACVFRole, subject_content: str, 
                                   debate_context: Dict[str, Any], provider: Optional[str] = None, 
                                   **kwargs) -> LLMResponse:
        """Generate response for a specific ACVF role."""
        if role == ACVFRole.CHALLENGER:
            context = debate_context.get('context', '')
            return await self.generate_challenger_response(subject_content, context, provider, **kwargs)
        
        elif role == ACVFRole.DEFENDER:
            challenger_args = debate_context.get('challenger_arguments', '')
            context = debate_context.get('context', '')
            return await self.generate_defender_response(subject_content, challenger_args, context, provider, **kwargs)
        
        elif role == ACVFRole.JUDGE:
            challenger_args = debate_context.get('challenger_arguments', '')
            defender_args = debate_context.get('defender_arguments', '')
            context = debate_context.get('context', '')
            return await self.generate_judge_verdict(subject_content, challenger_args, defender_args, context, provider, **kwargs)
        
        else:
            raise ValueError(f"Unknown ACVF role: {role}")