"""
Test suite for LLM client with vision analysis and fallback logic
Tests the critical AI integration components
"""

import pytest
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simgen.services.llm_client import LLMClient, LLMError


class TestLLMClient:
    """Test suite for LLM client functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client"""
        mock_client = AsyncMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_message
        return mock_client

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def sample_image_bytes(self):
        """Sample image bytes for testing"""
        # Decode the base64 1x1 PNG
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        return base64.b64decode(base64_data)

    @pytest.mark.asyncio
    async def test_llm_client_initialization(self):
        """Test LLM client initialization with different API key configurations"""
        
        # Test with both API keys
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'test_anthropic_key',
            'OPENAI_API_KEY': 'test_openai_key'
        }):
            with patch('simgen.core.config.settings') as mock_settings:
                mock_settings.anthropic_api_key = 'test_anthropic_key'
                mock_settings.openai_api_key = 'test_openai_key'
                
                client = LLMClient()
                assert client.anthropic_client is not None
                assert client.openai_client is not None

    @pytest.mark.asyncio
    async def test_llm_client_no_api_keys(self):
        """Test LLM client initialization without API keys"""
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('simgen.core.config.settings') as mock_settings:
                mock_settings.anthropic_api_key = None
                mock_settings.openai_api_key = None
                
                with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
                    LLMClient()

    @pytest.mark.asyncio
    async def test_complete_with_anthropic_success(self, mock_anthropic_client):
        """Test successful text completion with Anthropic"""
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
                client = LLMClient()
                
                result = await client.complete("Test prompt")
                assert result == "Test response"
                mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_openai_fallback(self, mock_openai_client):
        """Test OpenAI fallback when Anthropic fails"""
        
        # Mock failing Anthropic client
        mock_anthropic_fail = AsyncMock()
        mock_anthropic_fail.messages.create.side_effect = Exception("Anthropic failed")
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = 'test_key'
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_fail), \
                 patch('openai.AsyncOpenAI', return_value=mock_openai_client):
                
                client = LLMClient()
                
                result = await client.complete("Test prompt")
                assert result == "Test response"
                mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_all_clients_fail(self):
        """Test when both LLM clients fail"""
        
        mock_anthropic_fail = AsyncMock()
        mock_anthropic_fail.messages.create.side_effect = Exception("Anthropic failed")
        
        mock_openai_fail = AsyncMock()
        mock_openai_fail.chat.completions.create.side_effect = Exception("OpenAI failed")
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = 'test_key'
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_fail), \
                 patch('openai.AsyncOpenAI', return_value=mock_openai_fail):
                
                client = LLMClient()
                
                with pytest.raises(LLMError, match="Completion failed"):
                    await client.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_openai_client, sample_image_bytes):
        """Test successful image analysis"""
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = 'test_key'
            
            with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
                client = LLMClient()
                
                result = await client.analyze_image(
                    image_data=sample_image_bytes,
                    prompt="Describe this image"
                )
                
                assert result == "Test response"
                mock_openai_client.chat.completions.create.assert_called_once()
                
                # Verify the call was made with correct model
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['model'] == 'gpt-4o'

    @pytest.mark.asyncio
    async def test_analyze_image_no_openai_client(self):
        """Test image analysis when OpenAI client is not available"""
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            mock_anthropic = AsyncMock()
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                client = LLMClient()
                
                with pytest.raises(LLMError, match="Vision model not available"):
                    await client.analyze_image(
                        image_data=b"test_image_data",
                        prompt="Describe this image"
                    )

    @pytest.mark.asyncio
    async def test_analyze_image_openai_failure(self, sample_image_bytes):
        """Test image analysis when OpenAI fails"""
        
        mock_openai_fail = AsyncMock()
        mock_openai_fail.chat.completions.create.side_effect = Exception("OpenAI vision failed")
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = 'test_key'
            
            with patch('openai.AsyncOpenAI', return_value=mock_openai_fail):
                client = LLMClient()
                
                with pytest.raises(LLMError, match="Image analysis failed"):
                    await client.analyze_image(
                        image_data=sample_image_bytes,
                        prompt="Describe this image"
                    )

    @pytest.mark.asyncio
    async def test_complete_with_schema_anthropic(self, mock_anthropic_client):
        """Test structured completion with Anthropic function calling"""
        
        # Mock tool use response
        mock_content = MagicMock()
        mock_content.input = {"result": "structured_data"}
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_message
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
                client = LLMClient()
                
                schema = {
                    "type": "object",
                    "properties": {"result": {"type": "string"}}
                }
                
                result = await client.complete_with_schema("Test prompt", schema)
                assert result == {"result": "structured_data"}

    @pytest.mark.asyncio
    async def test_complete_with_schema_openai_json_mode(self, mock_openai_client):
        """Test structured completion with OpenAI JSON mode"""
        
        # Mock JSON response
        mock_choice = MagicMock()
        mock_choice.message.content = '{"result": "json_data"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = 'test_key'
            
            with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
                client = LLMClient()
                
                schema = {
                    "type": "object",
                    "properties": {"result": {"type": "string"}}
                }
                
                result = await client.complete_with_schema("Test prompt", schema)
                assert result == {"result": "json_data"}
                
                # Verify JSON mode was used
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['response_format'] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_anthropic_client):
        """Test successful API connection test"""
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
                client = LLMClient()
                
                result = await client.test_connection()
                assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test API connection test failure"""
        
        mock_anthropic_fail = AsyncMock()
        mock_anthropic_fail.messages.create.side_effect = Exception("Connection failed")
        
        mock_openai_fail = AsyncMock()
        mock_openai_fail.chat.completions.create.side_effect = Exception("Connection failed")
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = 'test_key'
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_fail), \
                 patch('openai.AsyncOpenAI', return_value=mock_openai_fail):
                
                client = LLMClient()
                
                result = await client.test_connection()
                assert result is False

    @pytest.mark.asyncio
    async def test_retry_logic_with_exponential_backoff(self, mock_anthropic_client):
        """Test retry logic with exponential backoff"""
        
        # Mock client that fails twice then succeeds
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="Success")]
            return mock_message
        
        mock_anthropic_client.messages.create.side_effect = side_effect
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
                client = LLMClient()
                
                # Should succeed after retries
                result = await client.complete("Test prompt")
                assert result == "Success"
                assert call_count == 3  # Failed twice, succeeded on third try


class TestLLMClientIntegration:
    """Integration tests for LLM client with real-world scenarios"""

    @pytest.mark.asyncio
    async def test_vision_analysis_realistic_prompt(self, sample_image_bytes):
        """Test vision analysis with realistic physics sketch prompt"""
        
        mock_openai = AsyncMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "I can see a robotic arm with two joints and an end effector. The base appears to be mounted on a platform, and there are clear articulation points for movement."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai.chat.completions.create.return_value = mock_response
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = 'test_key'
            
            with patch('openai.AsyncOpenAI', return_value=mock_openai):
                client = LLMClient()
                
                prompt = """
                Analyze this hand-drawn sketch and identify all physics objects and their properties.
                Focus on: Objects, Positions, Connections, Motion Intent, Materials, Environment
                """
                
                result = await client.analyze_image(sample_image_bytes, prompt)
                
                assert len(result) > 50  # Should be a detailed response
                assert "robotic arm" in result.lower() or "robot" in result.lower()

    @pytest.mark.asyncio
    async def test_structured_physics_extraction(self):
        """Test structured extraction of physics entities"""
        
        mock_anthropic = AsyncMock()
        mock_content = MagicMock()
        mock_content.input = {
            "objects": [
                {
                    "name": "robot_arm_base",
                    "type": "rigid_body", 
                    "geometry": {"shape": "box", "dimensions": [0.2, 0.2, 0.1]},
                    "material": {"density": 1000.0, "friction": 0.8},
                    "position": [0, 0, 0]
                }
            ],
            "environment": {
                "gravity": [0, 0, -9.81],
                "ground": {"type": "plane", "friction": 0.5}
            }
        }
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_anthropic.messages.create.return_value = mock_message
        
        with patch('simgen.core.config.settings') as mock_settings:
            mock_settings.anthropic_api_key = 'test_key'
            mock_settings.openai_api_key = None
            
            with patch('anthropic.AsyncAnthropic', return_value=mock_anthropic):
                client = LLMClient()
                
                schema = {
                    "type": "object",
                    "properties": {
                        "objects": {"type": "array"},
                        "environment": {"type": "object"}
                    }
                }
                
                result = await client.complete_with_schema(
                    "Extract physics entities from: robotic arm with base", 
                    schema
                )
                
                assert "objects" in result
                assert "environment" in result
                assert len(result["objects"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])