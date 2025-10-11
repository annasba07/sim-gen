import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ..core.config_clean import settings
from ..models.schemas import ExtractedEntities


logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


class LLMClient:
    """Unified client for interacting with multiple LLM APIs."""
    
    def __init__(self):
        # Initialize Anthropic client (primary)
        if settings.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        else:
            self.anthropic_client = None
            logger.warning("Anthropic API key not provided")
        
        # Initialize OpenAI client (fallback)
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not provided")
        
        if not self.anthropic_client and not self.openai_client:
            raise ValueError("At least one LLM API key must be provided")
    
    async def test_connection(self) -> bool:
        """Test API connections."""
        try:
            if self.anthropic_client:
                # Test with a simple request
                await self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Test"}]
                )
                return True
        except Exception as e:
            logger.error(f"Anthropic API test failed: {e}")
            
        try:
            if self.openai_client:
                # Test with a simple request
                await self.openai_client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )
                return True
        except Exception as e:
            logger.error(f"OpenAI API test failed: {e}")
        
        return False
    
    async def complete_with_schema(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Generate completion with structured output schema."""
        
        for attempt in range(max_retries):
            try:
                # Try Anthropic first (function calling)
                if self.anthropic_client:
                    result = await self._anthropic_structured_completion(
                        prompt, schema, temperature
                    )
                    if result:
                        return result
                
                # Fallback to OpenAI
                if self.openai_client:
                    result = await self._openai_structured_completion(
                        prompt, schema, temperature
                    )
                    if result:
                        return result
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise LLMError(f"All LLM completion attempts failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise LLMError("No available LLM clients")
    
    async def _anthropic_structured_completion(
        self, 
        prompt: str, 
        schema: Dict[str, Any], 
        temperature: float
    ) -> Optional[Dict[str, Any]]:
        """Anthropic completion with function calling."""
        try:
            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "name": "structured_response",
                    "description": "Provide a structured response following the given schema",
                    "input_schema": schema
                }],
                tool_choice={"type": "tool", "name": "structured_response"}
            )
            
            # Extract structured response from tool use
            if message.content and len(message.content) > 0:
                if hasattr(message.content[0], 'input'):
                    return message.content[0].input
                elif hasattr(message.content[0], 'text'):
                    # Fallback: try to parse JSON from text
                    try:
                        return json.loads(message.content[0].text)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse JSON from Anthropic response")
            
            return None
            
        except Exception as e:
            logger.error(f"Anthropic structured completion failed: {e}")
            return None
    
    async def _openai_structured_completion(
        self, 
        prompt: str, 
        schema: Dict[str, Any], 
        temperature: float
    ) -> Optional[Dict[str, Any]]:
        """OpenAI completion with JSON mode."""
        try:
            schema_description = json.dumps(schema, indent=2)
            enhanced_prompt = f"""
{prompt}

Please respond with valid JSON that exactly matches this schema:
{schema_description}

Return only valid JSON, no additional text.
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=temperature,
                max_tokens=4000
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.error(f"OpenAI structured completion failed: {e}")
            return None
    
    async def complete(
        self, 
        prompt: str, 
        temperature: float = 0.2, 
        max_tokens: int = 4000
    ) -> str:
        """Simple text completion."""
        try:
            # Try Anthropic first
            if self.anthropic_client:
                message = await self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if message.content and len(message.content) > 0:
                    return message.content[0].text
            
            # Fallback to OpenAI
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or ""
            
            raise LLMError("No response from any LLM client")
            
        except Exception as e:
            logger.error(f"Text completion failed: {e}")
            raise LLMError(f"Completion failed: {e}")
    
    async def analyze_image(
        self, 
        image_data: bytes, 
        prompt: str, 
        temperature: float = 0.1
    ) -> str:
        """Analyze image with vision models."""
        try:
            # For now, use OpenAI GPT-4V (Anthropic vision support can be added)
            if self.openai_client:
                import base64
                
                base64_image = base64.b64encode(image_data).decode()
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }],
                    temperature=temperature,
                    max_tokens=2000
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content or ""
            
            raise LLMError("Vision model not available")
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise LLMError(f"Image analysis failed: {e}")


# Singleton instance
_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get singleton LLM client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance