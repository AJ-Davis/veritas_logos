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
                usage=response.usage.dict() if response.usage else {},
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
            # Convert messages format for Anthropic
            system_message = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message += msg["content"] + "\n"
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
                usage=response.usage.dict() if hasattr(response, 'usage') else {},
                response_time_seconds=response_time,
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any], **kwargs) -> LLMResponse:
        """Generate structured JSON response using Anthropic API."""
        
        # Add JSON schema instruction
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
    """Unified client for multiple LLM providers with retry logic and rate limiting."""
    
    def __init__(self, configs: List[LLMConfig]):
        """
        Initialize LLM client with multiple provider configurations.
        
        Args:
            configs: List of LLM configurations for different providers
        """
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[str] = None
        
        for config in configs:
            provider_key = f"{config.provider.value}:{config.model}"
            
            if config.provider == LLMProvider.OPENAI:
                self.providers[provider_key] = OpenAIProvider(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                self.providers[provider_key] = AnthropicProvider(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            # Set first provider as default
            if self.default_provider is None:
                self.default_provider = provider_key
        
        logger.info(f"Initialized LLM client with {len(self.providers)} providers")
    
    def get_provider(self, provider: Optional[str] = None) -> BaseLLMProvider:
        """Get a specific provider or the default one."""
        if provider is None:
            provider = self.default_provider
        
        if provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider '{provider}' not found. Available: {available}")
        
        return self.providers[provider]
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate a response using the specified or default provider.
        
        Args:
            messages: List of messages in chat format
            provider: Provider key (provider:model) or None for default
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLMResponse object
        """
        llm_provider = self.get_provider(provider)
        
        # Implement retry logic
        last_error = None
        for attempt in range(llm_provider.config.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(llm_provider.config.retry_delay_seconds * attempt)
                    logger.info(f"Retrying LLM request (attempt {attempt + 1})")
                
                return await llm_provider.generate_response(messages, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == llm_provider.config.max_retries - 1:
                    break
        
        logger.error(f"All LLM retry attempts failed. Last error: {str(last_error)}")
        raise last_error
    
    async def generate_structured_response(self, messages: List[Dict[str, str]], 
                                         response_schema: Dict[str, Any],
                                         provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate a structured JSON response.
        
        Args:
            messages: List of messages in chat format
            response_schema: JSON schema for the expected response
            provider: Provider key or None for default
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLMResponse object with JSON content
        """
        llm_provider = self.get_provider(provider)
        
        # Implement retry logic with JSON validation
        last_error = None
        for attempt in range(llm_provider.config.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(llm_provider.config.retry_delay_seconds * attempt)
                    logger.info(f"Retrying structured LLM request (attempt {attempt + 1})")
                
                response = await llm_provider.generate_structured_response(
                    messages, response_schema, **kwargs
                )
                
                # Validate JSON response
                try:
                    json.loads(response.content)
                    return response
                except json.JSONDecodeError as json_error:
                    raise ValueError(f"Invalid JSON response: {str(json_error)}")
                
            except Exception as e:
                last_error = e
                logger.warning(f"Structured LLM request failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == llm_provider.config.max_retries - 1:
                    break
        
        logger.error(f"All structured LLM retry attempts failed. Last error: {str(last_error)}")
        raise last_error
    
    async def generate_with_fallback(self, messages: List[Dict[str, str]], 
                                   preferred_provider: Optional[str] = None,
                                   fallback_providers: Optional[List[str]] = None,
                                   **kwargs) -> LLMResponse:
        """
        Generate response with fallback to other providers if the preferred one fails.
        
        Args:
            messages: List of messages in chat format
            preferred_provider: Preferred provider key
            fallback_providers: List of fallback provider keys
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLMResponse object
        """
        providers_to_try = []
        
        if preferred_provider:
            providers_to_try.append(preferred_provider)
        
        if fallback_providers:
            providers_to_try.extend(fallback_providers)
        
        if not providers_to_try:
            providers_to_try = list(self.providers.keys())
        
        last_error = None
        for provider in providers_to_try:
            try:
                logger.info(f"Trying provider: {provider}")
                return await self.generate_response(messages, provider=provider, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider} failed: {str(e)}")
                continue
        
        logger.error(f"All providers failed. Last error: {str(last_error)}")
        raise last_error
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider keys."""
        return list(self.providers.keys())
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured providers."""
        info = {}
        for key, provider in self.providers.items():
            info[key] = {
                "provider": provider.config.provider.value,
                "model": provider.config.model,
                "max_tokens": provider.config.max_tokens,
                "temperature": provider.config.temperature,
                "timeout_seconds": provider.config.timeout_seconds
            }
        return info