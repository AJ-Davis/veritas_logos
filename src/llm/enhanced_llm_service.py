"""
Enhanced LLM Service with Multi-Provider Support

This service provides a unified interface for multiple LLM providers:
- Anthropic Claude (via anthropic SDK)
- Google Gemini (via google-generativeai SDK) 
- DeepSeek (via OpenAI-compatible API)

Features:
- Automatic fallback between providers
- Rate limiting and error handling
- Configurable provider priority
- Mock mode for testing without API keys
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
import json
import time
from datetime import datetime, timedelta

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    enabled: bool = True
    priority: int = 0  # Lower number = higher priority


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class EnhancedLLMService:
    """
    Enhanced LLM service with multi-provider support and fallback
    """
    
    def __init__(self, configs: Optional[List[LLMConfig]] = None):
        """
        Initialize the enhanced LLM service
        
        Args:
            configs: List of LLM provider configurations. If None, auto-detect from environment
        """
        self.configs = configs or self._auto_detect_configs()
        self.clients = {}
        self._initialize_clients()
        
        # Sort configs by priority (lower number = higher priority)
        self.configs.sort(key=lambda x: x.priority)
        
        logger.info(f"Initialized Enhanced LLM Service with {len(self.configs)} providers")
        for config in self.configs:
            if config.enabled:
                logger.info(f"  - {config.provider.value}: {config.model} (priority: {config.priority})")
    
    def _auto_detect_configs(self) -> List[LLMConfig]:
        """Auto-detect available LLM providers from environment variables"""
        configs = []
        
        # Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key != "your-anthropic-api-key-here" and ANTHROPIC_AVAILABLE:
            configs.append(LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_key,
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.7,
                priority=1
            ))
        
        # Google Gemini
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key and GEMINI_AVAILABLE:
            configs.append(LLMConfig(
                provider=LLMProvider.GEMINI,
                api_key=gemini_key,
                model="gemini-1.5-pro",
                max_tokens=4096,
                temperature=0.7,
                priority=2
            ))
        
        # DeepSeek
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key and OPENAI_AVAILABLE:
            configs.append(LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                api_key=deepseek_key,
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                max_tokens=4096,
                temperature=0.7,
                priority=3
            ))
        
        # Always include mock as fallback
        configs.append(LLMConfig(
            provider=LLMProvider.MOCK,
            model="mock-model",
            max_tokens=4096,
            temperature=0.7,
            priority=999  # Lowest priority
        ))
        
        return configs
    
    def _initialize_clients(self):
        """Initialize clients for each configured provider"""
        for config in self.configs:
            if not config.enabled:
                continue
                
            try:
                if config.provider == LLMProvider.ANTHROPIC and ANTHROPIC_AVAILABLE:
                    self.clients[config.provider] = Anthropic(api_key=config.api_key)
                    
                elif config.provider == LLMProvider.GEMINI and GEMINI_AVAILABLE:
                    genai.configure(api_key=config.api_key)
                    self.clients[config.provider] = genai.GenerativeModel(config.model)
                    
                elif config.provider == LLMProvider.DEEPSEEK and OPENAI_AVAILABLE:
                    self.clients[config.provider] = openai.OpenAI(
                        api_key=config.api_key,
                        base_url=config.base_url
                    )
                    
                elif config.provider == LLMProvider.MOCK:
                    self.clients[config.provider] = None  # Mock doesn't need a real client
                    
                logger.info(f"Initialized {config.provider.value} client")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {config.provider.value} client: {e}")
                config.enabled = False
    
    async def generate_text(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          preferred_provider: Optional[LLMProvider] = None) -> LLMResponse:
        """
        Generate text using the first available LLM provider
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            preferred_provider: Preferred provider to try first
            
        Returns:
            LLMResponse with generated text and metadata
        """
        # Create ordered list of providers to try
        providers_to_try = []
        
        # Add preferred provider first if specified
        if preferred_provider:
            preferred_config = next((c for c in self.configs if c.provider == preferred_provider and c.enabled), None)
            if preferred_config:
                providers_to_try.append(preferred_config)
        
        # Add remaining providers in priority order
        for config in self.configs:
            if config.enabled and config not in providers_to_try:
                providers_to_try.append(config)
        
        last_error = None
        
        for config in providers_to_try:
            try:
                logger.debug(f"Trying {config.provider.value} for text generation")
                
                response = await self._generate_with_provider(
                    config, prompt, system_prompt, max_tokens, temperature
                )
                
                logger.info(f"Successfully generated text using {config.provider.value}")
                return response
                
            except Exception as e:
                logger.warning(f"Failed to generate text with {config.provider.value}: {e}")
                last_error = e
                continue
        
        # If all providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")
    
    async def _generate_with_provider(self, 
                                    config: LLMConfig,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    max_tokens: Optional[int] = None,
                                    temperature: Optional[float] = None) -> LLMResponse:
        """Generate text with a specific provider"""
        
        # Use config defaults if not specified
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature
        
        if config.provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(config, prompt, system_prompt, max_tokens, temperature)
        elif config.provider == LLMProvider.GEMINI:
            return await self._generate_gemini(config, prompt, system_prompt, max_tokens, temperature)
        elif config.provider == LLMProvider.DEEPSEEK:
            return await self._generate_deepseek(config, prompt, system_prompt, max_tokens, temperature)
        elif config.provider == LLMProvider.MOCK:
            return await self._generate_mock(config, prompt, system_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def _generate_anthropic(self, config: LLMConfig, prompt: str, system_prompt: Optional[str], 
                                max_tokens: int, temperature: float) -> LLMResponse:
        """Generate text using Anthropic Claude"""
        client = self.clients[config.provider]
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            provider=config.provider,
            model=config.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            metadata={"response_id": getattr(response, '_request_id', None)}
        )
    
    async def _generate_gemini(self, config: LLMConfig, prompt: str, system_prompt: Optional[str],
                             max_tokens: int, temperature: float) -> LLMResponse:
        """Generate text using Google Gemini"""
        client = self.clients[config.provider]
        
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"
        
        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        # Configure safety settings (permissive for verification use case)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = client.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return LLMResponse(
            content=response.text,
            provider=config.provider,
            model=config.model,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0
            },
            metadata={"safety_ratings": [rating.category.name for rating in response.candidates[0].safety_ratings] if response.candidates else []}
        )
    
    async def _generate_deepseek(self, config: LLMConfig, prompt: str, system_prompt: Optional[str],
                               max_tokens: int, temperature: float) -> LLMResponse:
        """Generate text using DeepSeek"""
        client = self.clients[config.provider]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=config.provider,
            model=config.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            metadata={"response_id": response.id}
        )
    
    async def _generate_mock(self, config: LLMConfig, prompt: str, system_prompt: Optional[str],
                           max_tokens: int, temperature: float) -> LLMResponse:
        """Generate mock text for testing"""
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Generate mock response based on prompt content
        mock_responses = {
            "claim": "This claim appears to be factually accurate based on available evidence.",
            "extract": "Key claims extracted: 1. Primary assertion, 2. Supporting evidence, 3. Conclusion.",
            "evidence": "Supporting evidence found from reliable sources.",
            "citation": "Citations appear to be properly formatted and accessible.",
            "logic": "Logical structure is sound with clear premises and conclusion.",
            "bias": "No significant bias detected in the language or presentation.",
            "factual": "Factual claims are consistent with verified information.",
            "credibility": "Source credibility is high with proper attribution."
        }
        
        # Select response based on prompt keywords
        response_content = "Mock LLM response: Analysis complete."
        for keyword, response in mock_responses.items():
            if keyword in prompt.lower():
                response_content = f"Mock {keyword} analysis: {response}"
                break
        
        return LLMResponse(
            content=response_content,
            provider=config.provider,
            model=config.model,
            usage={
                "input_tokens": len(prompt.split()),
                "output_tokens": len(response_content.split()),
                "total_tokens": len(prompt.split()) + len(response_content.split())
            },
            metadata={"mock": True, "timestamp": datetime.now().isoformat()}
        )
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return [config.provider for config in self.configs if config.enabled]
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured providers"""
        info = {}
        for config in self.configs:
            info[config.provider.value] = {
                "enabled": config.enabled,
                "model": config.model,
                "priority": config.priority,
                "has_client": config.provider in self.clients
            }
        return info


# Default service instance
_default_service: Optional[EnhancedLLMService] = None


def get_enhanced_llm_service() -> EnhancedLLMService:
    """Get the default enhanced LLM service instance"""
    global _default_service
    if _default_service is None:
        _default_service = EnhancedLLMService()
    return _default_service


# Convenience functions for backward compatibility
async def generate_text(prompt: str, 
                       system_prompt: Optional[str] = None,
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       preferred_provider: Optional[str] = None) -> str:
    """
    Convenience function to generate text using the default service
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        preferred_provider: Preferred provider name (anthropic, gemini, deepseek)
        
    Returns:
        Generated text content
    """
    service = get_enhanced_llm_service()
    
    # Convert string provider to enum
    provider_enum = None
    if preferred_provider:
        try:
            provider_enum = LLMProvider(preferred_provider.lower())
        except ValueError:
            logger.warning(f"Unknown provider: {preferred_provider}")
    
    response = await service.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        preferred_provider=provider_enum
    )
    
    return response.content 