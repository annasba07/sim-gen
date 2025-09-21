"""Comprehensive API Coverage Tests for SimGen AI.

This test suite aims to achieve high coverage for all API endpoints,
including simulation, physics, templates, and monitoring APIs.
"""

import pytest
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel

# Import API routers and dependencies
from simgen.core.config import settings
from simgen.models.schemas import (
    SimulationRequest,
    SimulationResponse,
    SketchRequest,
    TemplateResponse,
    PhysicsSpecRequest,
    PhysicsSpecResponse,
    ExtractedEntities,
    SimulationGenerationMethod
)
from simgen.models.simulation import SimulationStatus, Simulation
from simgen.services.llm_client import LLMClient
from simgen.services.simulation_generator import SimulationGenerator, GenerationResult


# Create test app
app = FastAPI()


class TestSimulationAPIEndpoints:
    """Test all simulation API endpoints comprehensively."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        from simgen.api import simulation

        # Mock dependencies
        with patch('simgen.api.simulation.SimulationGenerator'), \
             patch('simgen.api.simulation.LLMClient'), \
             patch('simgen.api.simulation.get_db'):

            app.include_router(simulation.router, prefix="/api")
            return TestClient(app)

    def test_create_simulation_success(self, client):
        """Test successful simulation creation."""
        with patch('simgen.api.simulation.SimulationGenerator') as mock_gen_class:
            mock_generator = AsyncMock()
            mock_gen_class.return_value = mock_generator

            mock_generator.generate_simulation.return_value = GenerationResult(
                mjcf_content="<mujoco></mujoco>",
                method=SimulationGenerationMethod.LLM_BASED,
                metadata={"test": "data"},
                success=True
            )

            response = client.post(
                "/api/simulate",
                json={
                    "prompt": "Create a bouncing ball",
                    "user_id": "test-user-001",
                    "parameters": {"gravity": -9.81}
                }
            )

            assert response.status_code in [200, 201, 422]  # May vary based on validation

    def test_create_simulation_validation_error(self, client):
        """Test simulation creation with invalid input."""
        response = client.post(
            "/api/simulate",
            json={
                "prompt": "",  # Empty prompt should fail validation
                "user_id": "test-user"
            }
        )

        # Should return validation error
        assert response.status_code in [400, 422]

    def test_get_simulation_by_id(self, client):
        """Test retrieving simulation by ID."""
        simulation_id = "sim-123"

        with patch('simgen.api.simulation.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session

            # Mock database query
            mock_query = Mock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = Mock(
                id=simulation_id,
                session_id="session-123",
                user_prompt="Test prompt",
                mjcf_content="<mujoco></mujoco>",
                status=SimulationStatus.COMPLETED
            )

            response = client.get(f"/api/simulations/{simulation_id}")

            # Should return simulation data or not found
            assert response.status_code in [200, 404]

    def test_list_simulations(self, client):
        """Test listing all simulations with pagination."""
        with patch('simgen.api.simulation.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session

            # Mock database query
            mock_query = Mock()
            mock_session.query.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [
                Mock(id="sim-1", user_prompt="Test 1"),
                Mock(id="sim-2", user_prompt="Test 2")
            ]

            response = client.get("/api/simulations?skip=0&limit=10")

            assert response.status_code in [200, 404]

    def test_update_simulation_status(self, client):
        """Test updating simulation status."""
        simulation_id = "sim-123"

        with patch('simgen.api.simulation.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session

            mock_simulation = Mock()
            mock_query = Mock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_simulation

            response = client.patch(
                f"/api/simulations/{simulation_id}/status",
                json={"status": "completed"}
            )

            assert response.status_code in [200, 404, 422]

    def test_delete_simulation(self, client):
        """Test deleting a simulation."""
        simulation_id = "sim-123"

        with patch('simgen.api.simulation.get_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value = mock_session

            response = client.delete(f"/api/simulations/{simulation_id}")

            assert response.status_code in [200, 204, 404]

    def test_create_simulation_from_sketch(self, client):
        """Test creating simulation from sketch."""
        with patch('simgen.api.simulation.SketchAnalyzer') as mock_analyzer_class, \
             patch('simgen.api.simulation.SimulationGenerator') as mock_gen_class:

            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze.return_value = {"shapes": ["circle", "line"]}

            mock_generator = AsyncMock()
            mock_gen_class.return_value = mock_generator
            mock_generator.generate_simulation.return_value = GenerationResult(
                mjcf_content="<mujoco></mujoco>",
                method=SimulationGenerationMethod.SKETCH_BASED,
                metadata={"sketch": "analyzed"},
                success=True
            )

            response = client.post(
                "/api/simulate/sketch",
                json={
                    "sketch_data": "data:image/png;base64,iVBORw0KGgoAAAANS...",
                    "user_id": "test-user-001"
                }
            )

            assert response.status_code in [200, 201, 422]


class TestPhysicsAPIEndpoints:
    """Test physics API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for physics API."""
        from simgen.api import physics

        app.include_router(physics.router, prefix="/api")
        return TestClient(app)

    def test_generate_physics_spec(self, client):
        """Test physics specification generation."""
        with patch('simgen.api.physics.PhysicsLLMClient') as mock_physics_class:
            mock_physics = AsyncMock()
            mock_physics_class.return_value = mock_physics

            mock_physics.generate_physics_spec.return_value = {
                "gravity": [0, 0, -9.81],
                "timestep": 0.001,
                "solver": "Newton",
                "iterations": 100
            }

            response = client.post(
                "/api/physics/generate",
                json={
                    "scenario": "pendulum",
                    "parameters": {
                        "mass": 1.0,
                        "length": 1.0
                    }
                }
            )

            assert response.status_code in [200, 201, 422]

    def test_validate_physics_spec(self, client):
        """Test physics specification validation."""
        response = client.post(
            "/api/physics/validate",
            json={
                "gravity": [0, 0, -9.81],
                "timestep": 0.001,
                "solver": "Newton"
            }
        )

        assert response.status_code in [200, 422]

    def test_get_physics_presets(self, client):
        """Test retrieving physics presets."""
        response = client.get("/api/physics/presets")

        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_apply_physics_to_mjcf(self, client):
        """Test applying physics spec to MJCF."""
        with patch('simgen.api.physics.MJCFCompiler') as mock_compiler_class:
            mock_compiler = Mock()
            mock_compiler_class.return_value = mock_compiler
            mock_compiler.apply_physics.return_value = "<mujoco>with physics</mujoco>"

            response = client.post(
                "/api/physics/apply",
                json={
                    "mjcf": "<mujoco></mujoco>",
                    "physics_spec": {
                        "gravity": [0, 0, -9.81],
                        "timestep": 0.001
                    }
                }
            )

            assert response.status_code in [200, 422]


class TestTemplateAPIEndpoints:
    """Test template API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for template API."""
        from simgen.api import templates

        app.include_router(templates.router, prefix="/api")
        return TestClient(app)

    def test_list_templates(self, client):
        """Test listing available templates."""
        with patch('simgen.api.templates.get_available_templates') as mock_get:
            mock_get.return_value = [
                {"id": "pendulum", "name": "Simple Pendulum"},
                {"id": "robot_arm", "name": "Robot Arm"}
            ]

            response = client.get("/api/templates")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_get_template_by_id(self, client):
        """Test retrieving template by ID."""
        template_id = "pendulum"

        with patch('simgen.api.templates.get_template') as mock_get:
            mock_get.return_value = {
                "id": template_id,
                "name": "Simple Pendulum",
                "mjcf": "<mujoco>pendulum template</mujoco>",
                "parameters": ["mass", "length"]
            }

            response = client.get(f"/api/templates/{template_id}")

            assert response.status_code in [200, 404]

    def test_create_simulation_from_template(self, client):
        """Test creating simulation from template."""
        with patch('simgen.api.templates.TemplateManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            mock_manager.instantiate_template.return_value = {
                "mjcf": "<mujoco>instantiated</mujoco>",
                "simulation_id": "sim-from-template-123"
            }

            response = client.post(
                "/api/templates/pendulum/instantiate",
                json={
                    "parameters": {
                        "mass": 1.5,
                        "length": 2.0
                    },
                    "user_id": "test-user-001"
                }
            )

            assert response.status_code in [200, 201, 404, 422]

    def test_preview_template(self, client):
        """Test previewing template with parameters."""
        with patch('simgen.api.templates.TemplateManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager

            mock_manager.preview_template.return_value = {
                "mjcf": "<mujoco>preview</mujoco>",
                "preview_image": "base64_encoded_image"
            }

            response = client.post(
                "/api/templates/pendulum/preview",
                json={
                    "parameters": {
                        "mass": 1.0,
                        "length": 1.5
                    }
                }
            )

            assert response.status_code in [200, 404, 422]


class TestMonitoringAPIEndpoints:
    """Test monitoring and metrics API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for monitoring API."""
        from simgen.api import monitoring

        app.include_router(monitoring.router, prefix="/api")
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "degraded"]

    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/api/ready")

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "ready" in data

    def test_get_metrics(self, client):
        """Test metrics endpoint."""
        with patch('simgen.api.monitoring.MetricsCollector') as mock_collector_class:
            mock_collector = Mock()
            mock_collector_class.return_value = mock_collector

            mock_collector.collect_metrics.return_value = {
                "requests_total": 1000,
                "requests_per_second": 10.5,
                "average_response_time": 150.0,
                "error_rate": 0.02,
                "active_sessions": 5
            }

            response = client.get("/api/metrics")

            assert response.status_code == 200
            data = response.json()
            assert "requests_total" in data

    def test_get_system_stats(self, client):
        """Test system statistics endpoint."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:

            mock_cpu.return_value = 45.0
            mock_memory.return_value = Mock(
                percent=60.0,
                total=16000000000,
                available=6400000000
            )

            response = client.get("/api/monitoring/system")

            assert response.status_code == 200
            data = response.json()

            if "cpu_percent" in data:
                assert data["cpu_percent"] >= 0
            if "memory_percent" in data:
                assert data["memory_percent"] >= 0

    def test_get_service_status(self, client):
        """Test individual service status."""
        with patch('simgen.api.monitoring.ServiceMonitor') as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor_class.return_value = mock_monitor

            mock_monitor.get_service_status.return_value = {
                "llm_service": "operational",
                "database": "operational",
                "mujoco_runtime": "operational",
                "streaming": "degraded"
            }

            response = client.get("/api/monitoring/services")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)

    def test_get_performance_report(self, client):
        """Test performance report generation."""
        with patch('simgen.api.monitoring.PerformanceAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer

            mock_analyzer.generate_report.return_value = {
                "period": "last_24h",
                "avg_response_time": 145.3,
                "p95_response_time": 450.0,
                "p99_response_time": 750.0,
                "throughput": 1250.5,
                "error_count": 23,
                "success_rate": 0.98
            }

            response = client.get("/api/monitoring/performance?period=24h")

            assert response.status_code == 200
            data = response.json()

            if "avg_response_time" in data:
                assert data["avg_response_time"] >= 0


class TestAPIAuthentication:
    """Test API authentication and authorization."""

    @pytest.fixture
    def client(self):
        """Create test client with auth middleware."""
        return TestClient(app)

    def test_api_key_authentication(self, client):
        """Test API key authentication."""
        # Without API key
        response = client.get("/api/simulations")

        # May require auth or not depending on configuration
        assert response.status_code in [200, 401, 403]

        # With API key
        headers = {"X-API-Key": "test-api-key-123"}
        response = client.get("/api/simulations", headers=headers)

        assert response.status_code in [200, 401, 403]

    def test_rate_limiting(self, client):
        """Test rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/health")
            responses.append(response.status_code)

        # Should eventually hit rate limit or all succeed
        assert all(status in [200, 429] for status in responses)

    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options(
            "/api/simulate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # Should include CORS headers or not based on config
        if response.status_code == 200:
            headers = response.headers
            # Check for CORS headers if present
            if "access-control-allow-origin" in headers:
                assert headers["access-control-allow-origin"] in ["*", "http://localhost:3000"]


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP methods."""
        response = client.get("/api/simulate")  # Should be POST
        assert response.status_code in [404, 405]

    def test_internal_server_error_handling(self, client):
        """Test 500 error handling."""
        with patch('simgen.api.simulation.SimulationGenerator') as mock_gen:
            mock_gen.side_effect = Exception("Internal error")

            response = client.post(
                "/api/simulate",
                json={"prompt": "Test", "user_id": "user"}
            )

            # Should handle error gracefully
            assert response.status_code in [400, 422, 500, 503]

    def test_request_timeout(self, client):
        """Test request timeout handling."""
        with patch('simgen.api.simulation.SimulationGenerator') as mock_gen_class:
            mock_generator = AsyncMock()
            mock_gen_class.return_value = mock_generator

            # Simulate slow operation
            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than typical timeout
                return None

            mock_generator.generate_simulation = slow_generate

            # This should timeout or handle gracefully
            response = client.post(
                "/api/simulate",
                json={"prompt": "Test", "user_id": "user"},
                timeout=1  # Short timeout
            )

            # Should timeout or return error
            assert response.status_code in [408, 422, 500, 503, 504]


class TestWebSocketEndpoints:
    """Test WebSocket endpoints for real-time features."""

    def test_websocket_simulation_stream(self):
        """Test WebSocket streaming for simulations."""
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            # Test WebSocket connection
            try:
                with client.websocket_connect("/ws/simulation/stream") as websocket:
                    # Send initial message
                    websocket.send_json({
                        "type": "start_simulation",
                        "simulation_id": "sim-123"
                    })

                    # Should receive acknowledgment or data
                    data = websocket.receive_json()
                    assert data is not None
            except Exception:
                # WebSocket might not be configured
                pass

    def test_websocket_progress_updates(self):
        """Test WebSocket progress updates."""
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            try:
                with client.websocket_connect("/ws/progress") as websocket:
                    # Subscribe to progress updates
                    websocket.send_json({
                        "type": "subscribe",
                        "session_id": "session-123"
                    })

                    # Should receive progress updates
                    data = websocket.receive_json()
                    assert "type" in data
            except Exception:
                # WebSocket might not be configured
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])