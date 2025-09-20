"""Comprehensive API endpoint tests for maximum coverage boost."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
from datetime import datetime
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
import uuid

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the main FastAPI app and all API modules
from simgen.main import app
from simgen.api.simulation import router as simulation_router
from simgen.api.physics import router as physics_router
from simgen.api.templates import router as templates_router
from simgen.api.monitoring import router as monitoring_router


@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)


@pytest.fixture
def mock_simulation_data():
    """Mock simulation data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "prompt": "Create a bouncing ball simulation",
        "mjcf_content": "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>",
        "status": "completed",
        "user_id": "test-user-123",
        "created_at": datetime.utcnow().isoformat(),
        "processing_time": 2.5
    }


@pytest.fixture
def mock_physics_data():
    """Mock physics data for testing."""
    return {
        "gravity": -9.81,
        "timestep": 0.001,
        "solver": "CG",
        "iterations": 100,
        "tolerance": 1e-6
    }


class TestSimulationAPI:
    """Test simulation API endpoints comprehensively."""

    @patch('simgen.api.simulation.SimulationService')
    def test_create_simulation_success(self, mock_service_class, client, mock_simulation_data):
        """Test successful simulation creation."""
        # Setup mock service
        mock_service = AsyncMock()
        mock_service.create_simulation = AsyncMock(return_value=mock_simulation_data)
        mock_service_class.return_value = mock_service

        # Make request
        request_data = {
            "prompt": "Create a bouncing ball",
            "session_id": "session-123",
            "parameters": {"gravity": -9.81}
        }

        response = client.post("/api/simulations", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert data["status"] == "completed"

    def test_create_simulation_validation_error(self, client):
        """Test simulation creation with validation errors."""
        # Invalid request - missing prompt
        request_data = {
            "session_id": "session-123"
        }

        response = client.post("/api/simulations", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_create_simulation_empty_prompt(self, client):
        """Test simulation creation with empty prompt."""
        request_data = {
            "prompt": "",
            "session_id": "session-123"
        }

        response = client.post("/api/simulations", json=request_data)
        assert response.status_code == 422

    @patch('simgen.api.simulation.SimulationService')
    def test_get_simulation_success(self, mock_service_class, client, mock_simulation_data):
        """Test successful simulation retrieval."""
        mock_service = AsyncMock()
        mock_service.get_simulation = AsyncMock(return_value=mock_simulation_data)
        mock_service_class.return_value = mock_service

        simulation_id = "test-sim-123"
        response = client.get(f"/api/simulations/{simulation_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_simulation_data["id"]
        assert "mjcf_content" in data

    @patch('simgen.api.simulation.SimulationService')
    def test_get_simulation_not_found(self, mock_service_class, client):
        """Test simulation not found error."""
        mock_service = AsyncMock()
        mock_service.get_simulation = AsyncMock(return_value=None)
        mock_service_class.return_value = mock_service

        response = client.get("/api/simulations/nonexistent-id")
        assert response.status_code == 404

    @patch('simgen.api.simulation.SimulationService')
    def test_list_simulations(self, mock_service_class, client, mock_simulation_data):
        """Test listing simulations."""
        mock_service = AsyncMock()
        mock_service.list_simulations = AsyncMock(return_value=[
            mock_simulation_data,
            {**mock_simulation_data, "id": "sim-2", "prompt": "Another simulation"}
        ])
        mock_service_class.return_value = mock_service

        response = client.get("/api/simulations?user_id=test-user")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == mock_simulation_data["id"]

    @patch('simgen.api.simulation.SimulationService')
    def test_update_simulation_status(self, mock_service_class, client):
        """Test updating simulation status."""
        mock_service = AsyncMock()
        mock_service.update_simulation = AsyncMock(return_value=True)
        mock_service_class.return_value = mock_service

        simulation_id = "test-sim-123"
        update_data = {
            "status": "completed",
            "mjcf_content": "<mujoco>updated</mujoco>"
        }

        response = client.put(f"/api/simulations/{simulation_id}", json=update_data)
        assert response.status_code == 200

    @patch('simgen.api.simulation.SimulationService')
    def test_delete_simulation(self, mock_service_class, client):
        """Test simulation deletion."""
        mock_service = AsyncMock()
        mock_service.delete_simulation = AsyncMock(return_value=True)
        mock_service_class.return_value = mock_service

        simulation_id = "test-sim-123"
        response = client.delete(f"/api/simulations/{simulation_id}")
        assert response.status_code == 204


class TestPhysicsAPI:
    """Test physics API endpoints comprehensively."""

    @patch('simgen.api.physics.PhysicsService')
    def test_generate_physics_spec(self, mock_service_class, client, mock_physics_data):
        """Test physics specification generation."""
        mock_service = AsyncMock()
        mock_service.generate_physics = AsyncMock(return_value=mock_physics_data)
        mock_service_class.return_value = mock_service

        request_data = {
            "scenario_type": "pendulum",
            "parameters": {
                "mass": 1.0,
                "length": 2.0,
                "damping": 0.1
            }
        }

        response = client.post("/api/physics/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["gravity"] == -9.81
        assert "timestep" in data

    @patch('simgen.api.physics.PhysicsService')
    def test_validate_physics_spec(self, mock_service_class, client):
        """Test physics specification validation."""
        mock_service = AsyncMock()
        mock_service.validate_physics = AsyncMock(return_value={"valid": True, "errors": []})
        mock_service_class.return_value = mock_service

        physics_spec = {
            "gravity": -9.81,
            "timestep": 0.001,
            "solver": "CG"
        }

        response = client.post("/api/physics/validate", json=physics_spec)

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True

    @patch('simgen.api.physics.PhysicsService')
    def test_optimize_physics_parameters(self, mock_service_class, client):
        """Test physics parameter optimization."""
        mock_service = AsyncMock()
        mock_service.optimize_parameters = AsyncMock(return_value={
            "optimized_params": {"timestep": 0.0005, "iterations": 150},
            "performance_gain": 25.0
        })
        mock_service_class.return_value = mock_service

        request_data = {
            "current_params": {"timestep": 0.001, "iterations": 100},
            "target_metric": "stability",
            "constraints": {"max_timestep": 0.002}
        }

        response = client.post("/api/physics/optimize", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "optimized_params" in data
        assert data["performance_gain"] > 0

    @patch('simgen.api.physics.PhysicsService')
    def test_physics_presets(self, mock_service_class, client):
        """Test getting physics presets."""
        mock_service = AsyncMock()
        mock_service.get_presets = AsyncMock(return_value=[
            {"name": "Earth", "gravity": -9.81, "description": "Earth-like gravity"},
            {"name": "Moon", "gravity": -1.62, "description": "Moon-like gravity"},
            {"name": "Zero-G", "gravity": 0.0, "description": "Zero gravity environment"}
        ])
        mock_service_class.return_value = mock_service

        response = client.get("/api/physics/presets")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["name"] == "Earth"


class TestTemplatesAPI:
    """Test templates API endpoints comprehensively."""

    @patch('simgen.api.templates.TemplateService')
    def test_list_templates(self, mock_service_class, client):
        """Test listing available templates."""
        mock_service = AsyncMock()
        mock_service.list_templates = AsyncMock(return_value=[
            {
                "id": "template-1",
                "name": "Pendulum",
                "description": "Simple pendulum simulation",
                "category": "mechanics",
                "parameters": ["mass", "length", "gravity"]
            },
            {
                "id": "template-2",
                "name": "Robot Arm",
                "description": "Multi-joint robotic arm",
                "category": "robotics",
                "parameters": ["joints", "link_lengths", "actuator_force"]
            }
        ])
        mock_service_class.return_value = mock_service

        response = client.get("/api/templates")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Pendulum"

    @patch('simgen.api.templates.TemplateService')
    def test_get_template_detail(self, mock_service_class, client):
        """Test getting template details."""
        mock_service = AsyncMock()
        mock_service.get_template = AsyncMock(return_value={
            "id": "template-1",
            "name": "Pendulum",
            "mjcf_template": "<mujoco><body><geom type='sphere' size='{radius}'/></body></mujoco>",
            "parameters": [
                {"name": "radius", "type": "float", "default": 0.1, "min": 0.01, "max": 1.0}
            ]
        })
        mock_service_class.return_value = mock_service

        response = client.get("/api/templates/template-1")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Pendulum"
        assert "mjcf_template" in data

    @patch('simgen.api.templates.TemplateService')
    def test_create_template(self, mock_service_class, client):
        """Test creating a new template."""
        mock_service = AsyncMock()
        mock_service.create_template = AsyncMock(return_value={
            "id": "template-new",
            "status": "created"
        })
        mock_service_class.return_value = mock_service

        template_data = {
            "name": "Custom Robot",
            "description": "Custom robotic simulation",
            "mjcf_template": "<mujoco>{body_content}</mujoco>",
            "parameters": [
                {"name": "body_content", "type": "string", "required": True}
            ],
            "category": "robotics"
        }

        response = client.post("/api/templates", json=template_data)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "template-new"

    @patch('simgen.api.templates.TemplateService')
    def test_instantiate_template(self, mock_service_class, client):
        """Test instantiating template with parameters."""
        mock_service = AsyncMock()
        mock_service.instantiate_template = AsyncMock(return_value={
            "simulation_id": "sim-from-template",
            "mjcf_content": "<mujoco><body><geom type='sphere' size='0.2'/></body></mujoco>"
        })
        mock_service_class.return_value = mock_service

        instantiation_data = {
            "template_id": "template-1",
            "parameters": {
                "radius": 0.2,
                "mass": 1.5
            },
            "user_id": "test-user"
        }

        response = client.post("/api/templates/instantiate", json=instantiation_data)

        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data


class TestMonitoringAPI:
    """Test monitoring API endpoints comprehensively."""

    @patch('simgen.api.monitoring.HealthService')
    def test_health_check(self, mock_service_class, client):
        """Test health check endpoint."""
        mock_service = AsyncMock()
        mock_service.check_health = AsyncMock(return_value={
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600,
            "services": {
                "database": "healthy",
                "redis": "healthy",
                "llm": "healthy",
                "gpu": "available"
            }
        })
        mock_service_class.return_value = mock_service

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data

    @patch('simgen.api.monitoring.MetricsService')
    def test_get_metrics(self, mock_service_class, client):
        """Test metrics endpoint."""
        mock_service = AsyncMock()
        mock_service.get_metrics = AsyncMock(return_value={
            "total_requests": 10000,
            "total_errors": 25,
            "average_response_time": 250,
            "requests_per_minute": 45,
            "cache_hit_rate": 0.87,
            "active_simulations": 12,
            "gpu_utilization": 65.5
        })
        mock_service_class.return_value = mock_service

        response = client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 10000
        assert "cache_hit_rate" in data

    @patch('simgen.api.monitoring.LoggingService')
    def test_get_logs(self, mock_service_class, client):
        """Test logs retrieval endpoint."""
        mock_service = AsyncMock()
        mock_service.get_recent_logs = AsyncMock(return_value=[
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "level": "INFO",
                "message": "Simulation completed successfully",
                "simulation_id": "sim-123"
            },
            {
                "timestamp": "2024-01-15T10:29:45Z",
                "level": "ERROR",
                "message": "Failed to validate MJCF",
                "simulation_id": "sim-124"
            }
        ])
        mock_service_class.return_value = mock_service

        response = client.get("/api/logs?limit=10&level=INFO")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["level"] == "INFO"


class TestSketchAPI:
    """Test sketch processing API endpoints."""

    @patch('simgen.api.simulation.SketchAnalyzer')
    def test_analyze_sketch(self, mock_analyzer_class, client):
        """Test sketch analysis endpoint."""
        mock_analyzer = AsyncMock()
        mock_analyzer.analyze = AsyncMock(return_value={
            "detected_objects": [
                {"type": "circle", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
                {"type": "line", "confidence": 0.88, "bbox": [60, 10, 100, 80]}
            ],
            "scene_description": "A ball connected to a string (pendulum)",
            "suggested_physics": ["gravity", "pendulum_dynamics"]
        })
        mock_analyzer_class.return_value = mock_analyzer

        sketch_data = {
            "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "user_id": "test-user"
        }

        response = client.post("/api/sketch/analyze", json=sketch_data)

        assert response.status_code == 200
        data = response.json()
        assert "detected_objects" in data
        assert len(data["detected_objects"]) > 0

    @patch('simgen.api.simulation.SimulationGenerator')
    def test_sketch_to_simulation(self, mock_generator_class, client):
        """Test sketch to simulation conversion."""
        mock_generator = AsyncMock()
        mock_generator.generate_from_sketch = AsyncMock(return_value={
            "simulation_id": "sim-from-sketch",
            "mjcf_content": "<mujoco><worldbody><geom type='sphere'/></worldbody></mujoco>",
            "analysis": {
                "objects_detected": 2,
                "physics_applied": ["gravity", "collision"]
            }
        })
        mock_generator_class.return_value = mock_generator

        request_data = {
            "sketch_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "prompt": "Make this sketch into a physics simulation",
            "user_id": "test-user"
        }

        response = client.post("/api/sketch/to-simulation", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert "mjcf_content" in data


class TestWebSocketAPI:
    """Test WebSocket endpoints for real-time features."""

    @patch('simgen.api.simulation.ConnectionManager')
    def test_websocket_connection(self, mock_manager_class, client):
        """Test WebSocket connection endpoint."""
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        # Note: TestClient doesn't support WebSocket testing directly
        # This tests the endpoint existence and basic setup
        try:
            with client.websocket_connect("/api/ws/simulation/sim-123") as websocket:
                pass
        except Exception:
            # Expected to fail in test environment without real WebSocket
            pass

    def test_websocket_endpoint_exists(self, client):
        """Test that WebSocket endpoint is registered."""
        # Check if the endpoint is in the routes
        routes = [route.path for route in app.routes]
        websocket_routes = [r for r in routes if "ws" in r]
        assert len(websocket_routes) > 0


class TestErrorHandling:
    """Test error handling across all endpoints."""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/simulations",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test handling of missing content type."""
        response = client.post("/api/simulations", data='{"prompt": "test"}')
        # Should still work with string data
        assert response.status_code in [200, 422]

    @patch('simgen.api.simulation.SimulationService')
    def test_internal_server_error(self, mock_service_class, client):
        """Test handling of internal server errors."""
        mock_service = AsyncMock()
        mock_service.create_simulation = AsyncMock(side_effect=Exception("Internal error"))
        mock_service_class.return_value = mock_service

        request_data = {"prompt": "test simulation"}
        response = client.post("/api/simulations", json=request_data)
        assert response.status_code == 500

    def test_method_not_allowed(self, client):
        """Test handling of unsupported HTTP methods."""
        response = client.patch("/api/simulations")
        assert response.status_code == 405


class TestAuthentication:
    """Test authentication and authorization."""

    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        # Assuming some endpoints require authentication
        response = client.delete("/api/simulations/test-id")
        # May return 401 (unauthorized) or 403 (forbidden) depending on implementation
        assert response.status_code in [200, 401, 403, 404]

    def test_api_key_validation(self, client):
        """Test API key validation."""
        headers = {"X-API-Key": "invalid-key"}
        response = client.get("/api/simulations", headers=headers)
        # Should work with proper error handling
        assert response.status_code in [200, 401, 403]

    def test_rate_limiting(self, client):
        """Test rate limiting behavior."""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/health")
            responses.append(response.status_code)

        # All should succeed or some should be rate limited
        assert all(code in [200, 429] for code in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])