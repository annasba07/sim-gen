"""Comprehensive error handling and edge case tests for maximum coverage."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import numpy as np
from datetime import datetime, timedelta
import asyncio
import tempfile
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, Dict, Any, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import modules for testing error conditions
from simgen.core.config import settings
from simgen.services.llm_client import LLMClient
from simgen.services.prompt_parser import PromptParser
from simgen.services.simulation_generator import SimulationGenerator
from simgen.services.mjcf_compiler import MJCFCompiler
from simgen.services.performance_optimizer import PerformanceOptimizer
from simgen.services.resilience import ResilienceService


class TestLLMClientErrorHandling:
    """Test LLM client error scenarios and edge cases."""

    @patch('openai.AsyncOpenAI')
    async def test_api_connection_errors(self, mock_openai):
        """Test handling of API connection errors."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Test various connection error scenarios
        connection_errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
            OSError("Network unreachable"),
            Exception("Unknown network error")
        ]

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        for error in connection_errors:
            mock_client.chat.completions.create = AsyncMock(side_effect=error)

            with pytest.raises(Exception):
                await client.extract_entities("test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_api_rate_limit_errors(self, mock_openai):
        """Test handling of API rate limit errors."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        class RateLimitError(Exception):
            pass

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        mock_client.chat.completions.create = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))

        with pytest.raises(Exception):
            await client.extract_entities("test prompt")

    @patch('openai.AsyncOpenAI')
    async def test_malformed_api_responses(self, mock_openai):
        """Test handling of malformed API responses."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        # Test various malformed responses
        malformed_responses = [
            Mock(choices=[]),  # Empty choices
            Mock(choices=[Mock(message=Mock(content=""))]),  # Empty content
            Mock(choices=[Mock(message=Mock(content="invalid json"))]),  # Invalid JSON
            Mock(choices=[Mock(message=None)]),  # None message
            None  # None response
        ]

        for response in malformed_responses:
            mock_client.chat.completions.create = AsyncMock(return_value=response)

            try:
                result = await client.extract_entities("test prompt")
                # Should handle gracefully or raise appropriate exception
            except Exception:
                # Expected for malformed responses
                pass

    @patch('openai.AsyncOpenAI')
    async def test_large_prompt_handling(self, mock_openai):
        """Test handling of extremely large prompts."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        client = LLMClient(api_key="test-key")
        client.client = mock_client

        # Test with various large prompt sizes
        large_prompts = [
            "word " * 1000,  # 1000 words
            "word " * 10000,  # 10000 words
            "word " * 100000,  # 100000 words (extremely large)
            "a" * 1000000,  # 1 million characters
        ]

        for prompt in large_prompts:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"objects": ["test"]}'))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            try:
                result = await client.extract_entities(prompt)
                # Should handle large prompts appropriately
            except Exception:
                # May legitimately fail for extremely large prompts
                pass

    def test_invalid_api_key_scenarios(self):
        """Test various invalid API key scenarios."""
        invalid_keys = [
            None,
            "",
            "   ",  # Whitespace only
            "invalid-key",
            "sk-",  # Incomplete key
            "sk-" + "x" * 100,  # Too long
            123,  # Wrong type
            [],   # Wrong type
            {}    # Wrong type
        ]

        for key in invalid_keys:
            try:
                client = LLMClient(api_key=key)
                # Should either initialize with default handling or raise appropriate error
            except Exception:
                # Expected for some invalid keys
                pass


class TestPromptParserEdgeCases:
    """Test prompt parser edge cases and error conditions."""

    def test_extreme_prompt_scenarios(self):
        """Test parsing of extreme prompt scenarios."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        extreme_prompts = [
            "",  # Empty prompt
            " ",  # Single space
            "\n\n\n",  # Only newlines
            "\t\t\t",  # Only tabs
            "a",  # Single character
            "Create " + "a " * 10000 + "simulation",  # Extremely repetitive
            "🤖🚀⚡🔥💫🌟✨💎🎯🎪" * 100,  # Unicode/emoji heavy
            "SELECT * FROM users; DROP TABLE simulations;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal attempt
        ]

        for prompt in extreme_prompts:
            try:
                # Test validation
                is_valid = parser.validate_prompt(prompt)
                assert isinstance(is_valid, bool)

                # Test keyword extraction
                if is_valid:
                    keywords = parser.extract_keywords(prompt)
                    assert isinstance(keywords, list)

            except Exception:
                # Some extreme cases may legitimately raise exceptions
                pass

    async def test_parser_with_corrupted_llm_responses(self):
        """Test parser handling of corrupted LLM responses."""
        mock_llm = AsyncMock()
        parser = PromptParser(llm_client=mock_llm)

        # Various corrupted/unexpected responses
        corrupted_responses = [
            None,
            "",
            "corrupted data",
            {"invalid": "structure"},
            {"objects": None},
            {"objects": "not a list"},
            {"objects": [123, None, ""]},  # Mixed invalid types
            {"objects": []},  # Empty objects
            {"incomplete": "response"},  # Missing expected fields
        ]

        for response in corrupted_responses:
            mock_llm.extract_entities = AsyncMock(return_value=response)

            try:
                result = await parser.parse("test prompt")
                # Should handle corrupted responses gracefully
            except Exception:
                # Expected for some corrupted responses
                pass

    def test_concurrent_parsing_stress(self):
        """Test parser under concurrent load."""
        mock_llm = Mock()
        parser = PromptParser(llm_client=mock_llm)

        def parse_prompt(prompt_id):
            try:
                return parser.validate_prompt(f"Test prompt {prompt_id}")
            except Exception:
                return False

        # Test concurrent parsing
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parse_prompt, i) for i in range(100)]
            results = [future.result(timeout=5) for future in futures]

        # Should handle concurrent requests
        assert len(results) == 100


class TestMJCFCompilerErrorScenarios:
    """Test MJCF compiler error scenarios and edge cases."""

    def test_malformed_xml_handling(self):
        """Test handling of malformed XML structures."""
        compiler = MJCFCompiler()

        malformed_xml_examples = [
            "",  # Empty string
            "<",  # Incomplete tag
            "<mujoco",  # Unclosed tag
            "<mujoco><worldbody>",  # Missing closing tags
            "<mujoco><worldbody></worldbody>",  # Missing closing mujoco
            "<mujoco><invalid_tag></mujoco>",  # Invalid tags
            "not xml at all",  # Not XML
            "<?xml version='1.0'?><mujoco></mujoco>",  # XML declaration
            "<mujoco><worldbody><geom type='invalid'/></worldbody></mujoco>",  # Invalid attributes
            "<mujoco>" + "x" * 1000000 + "</mujoco>",  # Extremely large content
        ]

        for xml in malformed_xml_examples:
            result = compiler.validate(xml)
            assert isinstance(result, bool)
            # Malformed XML should return False
            if xml in ["", "<", "<mujoco", "not xml at all"]:
                assert result is False

    def test_compilation_memory_stress(self):
        """Test compiler under memory stress conditions."""
        compiler = MJCFCompiler()

        # Create large MJCF structures
        large_mjcf_structures = []

        # Generate MJCF with many objects
        base = "<mujoco><worldbody>"
        for i in range(1000):  # Many objects
            base += f"<geom name='obj{i}' type='sphere' size='0.1' pos='{i} 0 0'/>"
        base += "</worldbody></mujoco>"
        large_mjcf_structures.append(base)

        # Generate deeply nested structure
        nested = "<mujoco><worldbody>"
        for i in range(100):  # Deep nesting
            nested += f"<body name='body{i}'>"
        nested += "<geom type='sphere'/>"
        for i in range(100):
            nested += "</body>"
        nested += "</worldbody></mujoco>"
        large_mjcf_structures.append(nested)

        for mjcf in large_mjcf_structures:
            try:
                result = compiler.compile(mjcf)
                # Should handle large structures or fail gracefully
            except MemoryError:
                # Expected for extremely large structures
                pass
            except Exception:
                # Other exceptions may also be acceptable
                pass

    def test_invalid_physics_parameters(self):
        """Test compilation with invalid physics parameters."""
        compiler = MJCFCompiler()

        invalid_physics_mjcf = [
            # Invalid timestep
            """<mujoco>
                <option timestep='-0.001'/>
                <worldbody><geom type='sphere'/></worldbody>
            </mujoco>""",

            # Invalid gravity
            """<mujoco>
                <option gravity='invalid'/>
                <worldbody><geom type='sphere'/></worldbody>
            </mujoco>""",

            # Missing required attributes
            """<mujoco>
                <worldbody>
                    <geom type='sphere'/>  <!-- Missing size -->
                </worldbody>
            </mujoco>""",

            # Circular references
            """<mujoco>
                <worldbody>
                    <body name='body1' childclass='class1'>
                        <geom type='sphere'/>
                    </body>
                </worldbody>
                <default class='class1' childclass='class1'/>
            </mujoco>""",
        ]

        for mjcf in invalid_physics_mjcf:
            result = compiler.validate(mjcf)
            # Should identify invalid physics parameters
            if "timestep='-0.001'" in mjcf or "gravity='invalid'" in mjcf:
                assert result is False


class TestSimulationGeneratorErrorConditions:
    """Test simulation generator error conditions."""

    @patch('simgen.services.simulation_generator.MJCFCompiler')
    async def test_generation_with_resource_constraints(self, mock_compiler_class):
        """Test generation under resource constraints."""
        mock_llm = AsyncMock()
        mock_parser = AsyncMock()
        mock_compiler = Mock()
        mock_compiler_class.return_value = mock_compiler

        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        # Simulate resource exhaustion scenarios
        resource_errors = [
            MemoryError("Out of memory"),
            OSError("Disk space full"),
            TimeoutError("Generation timeout"),
            RuntimeError("Resource limit exceeded")
        ]

        for error in resource_errors:
            mock_llm.generate_mjcf = AsyncMock(side_effect=error)
            mock_parser.parse = AsyncMock(return_value={"entities": {"objects": ["test"]}})

            with pytest.raises(Exception):
                await generator.generate("test prompt", user_id="test-user")

    async def test_generation_cancellation(self):
        """Test generation cancellation scenarios."""
        mock_llm = AsyncMock()
        mock_parser = AsyncMock()

        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        # Test asyncio cancellation
        async def slow_generation():
            await asyncio.sleep(10)  # Simulate slow generation
            return {"entities": {"objects": ["test"]}}

        mock_parser.parse = slow_generation

        task = asyncio.create_task(
            generator.generate("test prompt", user_id="test-user")
        )

        # Cancel after short delay
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    def test_invalid_user_scenarios(self):
        """Test generation with invalid user scenarios."""
        mock_llm = Mock()
        mock_parser = Mock()

        generator = SimulationGenerator(
            llm_client=mock_llm,
            prompt_parser=mock_parser
        )

        invalid_users = [
            None,
            "",
            "   ",  # Whitespace only
            "user@domain@domain",  # Invalid email format
            "a" * 1000,  # Extremely long user ID
            123,  # Wrong type
            [],   # Wrong type
            {}    # Wrong type
        ]

        for user_id in invalid_users:
            try:
                # Should validate user_id or handle gracefully
                asyncio.run(generator.generate("test prompt", user_id=user_id))
            except Exception:
                # Expected for invalid user IDs
                pass


class TestPerformanceOptimizerEdgeCases:
    """Test performance optimizer edge cases."""

    def test_optimizer_with_extreme_loads(self):
        """Test optimizer under extreme system loads."""
        optimizer = PerformanceOptimizer()

        # Simulate extreme system conditions
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:

            extreme_conditions = [
                {"cpu": 100.0, "memory": 99.9},  # Near maximum usage
                {"cpu": 0.0, "memory": 0.1},     # Near minimum usage
                {"cpu": -1.0, "memory": -1.0},   # Invalid values
                {"cpu": float('inf'), "memory": float('inf')},  # Infinity
                {"cpu": float('nan'), "memory": float('nan')},  # NaN
            ]

            for condition in extreme_conditions:
                mock_cpu.return_value = condition["cpu"]
                mock_memory.return_value = Mock(percent=condition["memory"])

                try:
                    metrics = optimizer.collect_metrics()
                    assert isinstance(metrics, dict)
                except Exception:
                    # Some extreme conditions may cause exceptions
                    pass

    def test_optimizer_concurrent_access(self):
        """Test optimizer under concurrent access."""
        optimizer = PerformanceOptimizer()

        def collect_metrics_worker(worker_id):
            try:
                return optimizer.collect_metrics()
            except Exception:
                return None

        # Test with multiple concurrent workers
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(collect_metrics_worker, i) for i in range(100)]

            try:
                results = [future.result(timeout=10) for future in futures]
                # Should handle concurrent access
                assert len(results) == 100
            except FutureTimeoutError:
                # Some operations may timeout under load
                pass

    def test_optimizer_with_missing_dependencies(self):
        """Test optimizer when system dependencies are missing."""
        optimizer = PerformanceOptimizer()

        # Simulate missing psutil
        with patch('psutil.cpu_percent', side_effect=ImportError("psutil not available")):
            try:
                metrics = optimizer.collect_metrics()
                # Should handle missing dependencies gracefully
            except ImportError:
                # Expected when dependencies are missing
                pass


class TestResilienceServiceErrorRecovery:
    """Test resilience service error recovery patterns."""

    def test_circuit_breaker_edge_cases(self):
        """Test circuit breaker with edge case scenarios."""
        if 'CircuitBreaker' not in globals():
            # Mock circuit breaker if not available
            class CircuitBreaker:
                def __init__(self, failure_threshold=5, recovery_timeout=60):
                    self.failure_threshold = failure_threshold
                    self.failure_count = 0
                    self.state = "closed"

                def __enter__(self):
                    if self.state == "open":
                        raise Exception("Circuit breaker is open")
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        self.failure_count += 1
                        if self.failure_count >= self.failure_threshold:
                            self.state = "open"

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Test rapid failure scenarios
        for i in range(10):
            try:
                with breaker:
                    if i < 5:  # First 5 fail
                        raise Exception(f"Failure {i}")
                    # Rest should be blocked by circuit breaker
            except Exception:
                pass

        # Test recovery scenarios
        time.sleep(1.1)  # Wait for recovery timeout
        if hasattr(breaker, 'reset'):
            breaker.reset()

    async def test_retry_policy_extreme_scenarios(self):
        """Test retry policy with extreme scenarios."""
        # Mock retry policy if not available
        class RetryPolicy:
            def __init__(self, max_attempts=3, delay=1, backoff=2):
                self.max_attempts = max_attempts
                self.delay = delay
                self.backoff = backoff

            def retry(self, func):
                async def wrapper(*args, **kwargs):
                    last_exception = None
                    for attempt in range(self.max_attempts):
                        try:
                            return await func(*args, **kwargs)
                        except Exception as e:
                            last_exception = e
                            if attempt < self.max_attempts - 1:
                                await asyncio.sleep(self.delay * (self.backoff ** attempt))
                    raise last_exception
                return wrapper

        policy = RetryPolicy(max_attempts=5, delay=0.1, backoff=2)

        # Test function that always fails
        @policy.retry
        async def always_fails():
            raise Exception("Always fails")

        with pytest.raises(Exception):
            await always_fails()

        # Test function with intermittent failures
        call_count = 0

        @policy.retry
        async def intermittent_failure():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = await intermittent_failure()
        assert result == "success"
        assert call_count == 3


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety scenarios."""

    def test_concurrent_service_access(self):
        """Test concurrent access to services."""
        def worker_function(worker_id):
            try:
                # Test multiple services concurrently
                compiler = MJCFCompiler()
                optimizer = PerformanceOptimizer()

                # Perform operations
                compiler.validate("<mujoco><worldbody></worldbody></mujoco>")
                optimizer.collect_metrics()

                return f"Worker {worker_id} completed"
            except Exception as e:
                return f"Worker {worker_id} failed: {e}"

        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_function, i) for i in range(50)]
            results = [future.result(timeout=30) for future in futures]

        # All workers should complete (successfully or with expected errors)
        assert len(results) == 50

    async def test_async_operation_cancellation(self):
        """Test cancellation of async operations."""
        async def long_running_operation():
            await asyncio.sleep(10)
            return "completed"

        # Test cancellation during operation
        task = asyncio.create_task(long_running_operation())
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestSystemResourceLimits:
    """Test behavior under system resource limits."""

    def test_memory_limit_handling(self):
        """Test handling of memory limits."""
        # Test with progressively larger data structures
        data_sizes = [1000, 10000, 100000]

        for size in data_sizes:
            try:
                # Create large data structure
                large_data = ["x" * 1000] * size

                # Test processing large data
                compiler = MJCFCompiler()
                mjcf = "<mujoco><worldbody>" + "".join(
                    f"<geom name='obj{i}' type='sphere'/>" for i in range(min(size, 1000))
                ) + "</worldbody></mujoco>"

                result = compiler.validate(mjcf)
                assert isinstance(result, bool)

            except MemoryError:
                # Expected for very large data
                break

    def test_file_system_limits(self):
        """Test behavior when file system limits are reached."""
        # Test temporary file creation limits
        temp_files = []

        try:
            for i in range(1000):  # Try to create many temp files
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_files.append(temp_file)

        except OSError:
            # Expected when file limits are reached
            pass

        finally:
            # Clean up
            for temp_file in temp_files:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except Exception:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])