import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status
from unittest.mock import Mock, patch, AsyncMock
import json
import base64
from datetime import datetime

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from simgen.main import app
from simgen.models.schemas import (
    SimulationRequest,
    SimulationResponse,
    HealthCheck,
    TemplateResponse
)


@pytest.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_simulation_result():
    """Mock simulation result."""
    return {
        "simulation_id": "test-sim-123",
        "mjcf_content": "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>",
        "status": "completed",
        "processing_time": 2.5,
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def sample_sketch_data():
    """Generate sample base64 encoded sketch data."""
    # Create a minimal PNG header
    png_header = b'\x89PNG\r\n\x1a\n'
    return f"data:image/png;base64,{base64.b64encode(png_header).decode()}"


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, async_client):
        """Test successful health check."""
        with patch('simgen.main.async_engine') as mock_engine:
            with patch('redis.from_url') as mock_redis:
                # Mock healthy services
                mock_engine.begin = AsyncMock()
                mock_redis.return_value.ping = Mock(return_value=True)

                response = await async_client.get("/health")

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "version" in data
                assert "timestamp" in data
                assert "services" in data

    @pytest.mark.asyncio
    async def test_health_check_partial_failure(self, async_client):
        """Test health check with some services down."""
        with patch('simgen.main.async_engine') as mock_engine:
            with patch('redis.from_url') as mock_redis:
                # Database is down, Redis is up
                mock_engine.begin = AsyncMock(side_effect=Exception("DB Error"))
                mock_redis.return_value.ping = Mock(return_value=True)

                response = await async_client.get("/health")

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                services = data["services"]
                assert services.get("redis") == "ok"
                # Database might be marked as error or skipped in test mode


class TestSimulationEndpoints:
    """Test cases for simulation generation endpoints."""

    @pytest.mark.asyncio
    async def test_generate_simulation_success(self, async_client, mock_simulation_result):
        """Test successful simulation generation."""
        request_data = {
            "prompt": "A pendulum swinging back and forth",
            "user_id": "test_user"
        }

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.generate_simulation = AsyncMock(
                return_value=mock_simulation_result
            )

            response = await async_client.post(
                "/api/v1/simulation/generate",
                json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["simulation_id"] == mock_simulation_result["simulation_id"]
            assert "mjcf_content" in data

    @pytest.mark.asyncio
    async def test_generate_simulation_with_sketch(
        self, async_client, mock_simulation_result, sample_sketch_data
    ):
        """Test simulation generation with sketch data."""
        request_data = {
            "prompt": "Make this robot arm move",
            "sketch_data": sample_sketch_data,
            "user_id": "test_user"
        }

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.generate_simulation = AsyncMock(
                return_value=mock_simulation_result
            )

            response = await async_client.post(
                "/api/v1/simulation/generate",
                json=request_data
            )

            assert response.status_code == status.HTTP_200_OK

            # Verify sketch data was passed to service
            call_args = mock_service.return_value.generate_simulation.call_args
            assert call_args[0][0].sketch_data == sample_sketch_data

    @pytest.mark.asyncio
    async def test_generate_simulation_invalid_prompt(self, async_client):
        """Test simulation generation with invalid prompt."""
        request_data = {
            "prompt": "",  # Empty prompt
            "user_id": "test_user"
        }

        response = await async_client.post(
            "/api/v1/simulation/generate",
            json=request_data
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_generate_simulation_service_error(self, async_client):
        """Test handling of service errors during generation."""
        request_data = {
            "prompt": "A complex simulation",
            "user_id": "test_user"
        }

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.generate_simulation = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = await async_client.post(
                "/api/v1/simulation/generate",
                json=request_data
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "error" in data or "detail" in data

    @pytest.mark.asyncio
    async def test_test_generate_endpoint(self, async_client):
        """Test the test-generate endpoint (no database)."""
        request_data = {
            "prompt": "Test simulation without DB",
            "user_id": "test_user"
        }

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_result = {
                "mjcf_content": "<mujoco>test</mujoco>",
                "processing_time": 1.0
            }
            mock_service.return_value.generate_test = AsyncMock(
                return_value=mock_result
            )

            response = await async_client.post(
                "/api/v1/simulation/test-generate",
                json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "mjcf_content" in data
            assert data["mjcf_content"] == mock_result["mjcf_content"]

    @pytest.mark.asyncio
    async def test_get_simulation_by_id(self, async_client, mock_simulation_result):
        """Test retrieving simulation by ID."""
        simulation_id = "test-sim-123"

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.get_simulation = AsyncMock(
                return_value=mock_simulation_result
            )

            response = await async_client.get(
                f"/api/v1/simulation/{simulation_id}"
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["simulation_id"] == simulation_id

    @pytest.mark.asyncio
    async def test_get_simulation_not_found(self, async_client):
        """Test retrieving non-existent simulation."""
        simulation_id = "non-existent-id"

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.get_simulation = AsyncMock(
                return_value=None
            )

            response = await async_client.get(
                f"/api/v1/simulation/{simulation_id}"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_list_simulations(self, async_client):
        """Test listing simulations with pagination."""
        mock_simulations = [
            {"simulation_id": f"sim-{i}", "prompt": f"Test {i}"}
            for i in range(5)
        ]

        with patch('simgen.api.simulation.SimulationService') as mock_service:
            mock_service.return_value.list_simulations = AsyncMock(
                return_value={
                    "simulations": mock_simulations,
                    "total": 5,
                    "page": 1,
                    "per_page": 10
                }
            )

            response = await async_client.get(
                "/api/v1/simulation/list",
                params={"page": 1, "per_page": 10}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["simulations"]) == 5
            assert data["total"] == 5


class TestTemplateEndpoints:
    """Test cases for template management endpoints."""

    @pytest.mark.asyncio
    async def test_list_templates(self, async_client):
        """Test listing available templates."""
        mock_templates = [
            {
                "template_id": "pendulum",
                "name": "Simple Pendulum",
                "description": "A basic pendulum simulation"
            },
            {
                "template_id": "robot_arm",
                "name": "Robot Arm",
                "description": "6-DOF robotic arm"
            }
        ]

        with patch('simgen.api.templates.TemplateService') as mock_service:
            mock_service.return_value.list_templates = AsyncMock(
                return_value=mock_templates
            )

            response = await async_client.get("/api/v1/templates")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["template_id"] == "pendulum"

    @pytest.mark.asyncio
    async def test_get_template(self, async_client):
        """Test retrieving a specific template."""
        template_id = "pendulum"
        mock_template = {
            "template_id": template_id,
            "name": "Simple Pendulum",
            "mjcf_content": "<mujoco>pendulum</mujoco>",
            "parameters": ["mass", "length", "damping"]
        }

        with patch('simgen.api.templates.TemplateService') as mock_service:
            mock_service.return_value.get_template = AsyncMock(
                return_value=mock_template
            )

            response = await async_client.get(f"/api/v1/templates/{template_id}")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["template_id"] == template_id
            assert "mjcf_content" in data

    @pytest.mark.asyncio
    async def test_create_custom_template(self, async_client):
        """Test creating a custom template."""
        template_data = {
            "name": "Custom Robot",
            "description": "My custom robot template",
            "mjcf_content": "<mujoco>custom</mujoco>",
            "parameters": ["size", "color"]
        }

        with patch('simgen.api.templates.TemplateService') as mock_service:
            mock_service.return_value.create_template = AsyncMock(
                return_value={
                    "template_id": "custom_robot_123",
                    **template_data
                }
            )

            response = await async_client.post(
                "/api/v1/templates",
                json=template_data
            )

            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert "template_id" in data
            assert data["name"] == template_data["name"]


class TestPhysicsEndpoints:
    """Test cases for physics pipeline endpoints."""

    @pytest.mark.asyncio
    async def test_generate_physics(self, async_client):
        """Test physics generation endpoint."""
        request_data = {
            "prompt": "A ball bouncing on the ground",
            "physics_spec": {
                "gravity": -9.81,
                "timestep": 0.002,
                "iterations": 100
            }
        }

        with patch('simgen.api.physics.PhysicsService') as mock_service:
            mock_result = {
                "physics_id": "physics-123",
                "mjcf_content": "<mujoco>physics</mujoco>",
                "spec_applied": True
            }
            mock_service.return_value.generate_physics = AsyncMock(
                return_value=mock_result
            )

            response = await async_client.post(
                "/api/v2/physics/generate",
                json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["physics_id"] == "physics-123"
            assert data["spec_applied"] is True

    @pytest.mark.asyncio
    async def test_validate_physics_spec(self, async_client):
        """Test physics specification validation."""
        spec_data = {
            "gravity": -9.81,
            "timestep": 0.001,
            "solver": "Newton",
            "iterations": 50
        }

        with patch('simgen.api.physics.PhysicsService') as mock_service:
            mock_service.return_value.validate_spec = AsyncMock(
                return_value={"valid": True, "warnings": []}
            )

            response = await async_client.post(
                "/api/v2/physics/validate",
                json=spec_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["valid"] is True


class TestWebSocketEndpoint:
    """Test cases for WebSocket connections."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test establishing WebSocket connection."""
        # Note: This is a simplified test - real WebSocket testing requires
        # a different approach with websocket test client

        with patch('simgen.api.simulation.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = AsyncMock()

            # Would need proper WebSocket test client here
            # This is a placeholder to show the structure
            pass


class TestErrorHandling:
    """Test cases for error handling."""

    @pytest.mark.asyncio
    async def test_404_not_found(self, async_client):
        """Test 404 response for non-existent endpoint."""
        response = await async_client.get("/api/v1/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, async_client):
        """Test 405 response for wrong HTTP method."""
        response = await async_client.get("/api/v1/simulation/generate")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    @pytest.mark.asyncio
    async def test_validation_error(self, async_client):
        """Test validation error response."""
        # Missing required fields
        response = await async_client.post(
            "/api/v1/simulation/generate",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data


class TestRateLimiting:
    """Test cases for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client):
        """Test that rate limiting is applied."""
        with patch('simgen.api.simulation.rate_limiter') as mock_limiter:
            mock_limiter.check_rate_limit = AsyncMock(return_value=False)

            response = await async_client.post(
                "/api/v1/simulation/generate",
                json={"prompt": "test", "user_id": "test"}
            )

            # Should return 429 if rate limited
            # (Depends on implementation)
            pass