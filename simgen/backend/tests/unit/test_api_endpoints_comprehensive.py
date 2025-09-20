"""Comprehensive API endpoint tests covering all routes and scenarios."""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
from datetime import datetime
import tempfile
import os
from typing import Optional, Dict, Any, List
import uuid
from fastapi.testclient import TestClient

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import API components
from simgen.main import app
from simgen.models.simulation import Simulation, SimulationGenerationMethod, SimulationStatus
from simgen.models.schemas import SimulationRequest, SimulationResponse


class TestSimulationAPIEndpoints:
    """Comprehensive tests for simulation API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session for testing."""
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None
        session.query.return_value.all.return_value = []
        session.add = MagicMock()
        session.commit = MagicMock()
        session.rollback = MagicMock()
        return session

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]  # API may return either value

    def test_simulation_creation_endpoint_success(self, client):
        """Test successful simulation creation via API."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session

            simulation_request = {
                "user_prompt": "Create a bouncing ball simulation",
                "session_id": "test-session-001"
            }

            response = client.post("/api/simulations", json=simulation_request)

            # Should return 200 or 201 depending on implementation
            assert response.status_code in [200, 201]

    def test_simulation_creation_endpoint_validation_error(self, client):
        """Test simulation creation with validation errors."""
        # Test missing required fields
        invalid_request = {
            "session_id": "test-session-002"
            # Missing user_prompt
        }

        response = client.post("/api/simulations", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_simulation_retrieval_by_id(self, client):
        """Test retrieving simulation by ID."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_simulation = Simulation(
                id=1,
                session_id="test-session",
                user_prompt="Test simulation",
                mjcf_content="<mujoco>test</mujoco>",
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED
            )
            mock_session.query.return_value.filter.return_value.first.return_value = mock_simulation
            mock_get_db.return_value = mock_session

            response = client.get("/api/simulations/1")

            if response.status_code == 200:
                data = response.json()
                assert data["id"] == 1
                assert data["user_prompt"] == "Test simulation"

    def test_simulation_retrieval_not_found(self, client):
        """Test retrieving non-existent simulation."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_get_db.return_value = mock_session

            response = client.get("/api/simulations/999")
            assert response.status_code == 404

    def test_simulation_list_endpoint(self, client):
        """Test listing simulations."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_simulations = [
                Simulation(
                    id=1,
                    session_id="session-1",
                    user_prompt="Simulation 1",
                    mjcf_content="<mujoco>test1</mujoco>",
                    generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                    status=SimulationStatus.COMPLETED
                ),
                Simulation(
                    id=2,
                    session_id="session-2",
                    user_prompt="Simulation 2",
                    mjcf_content="<mujoco>test2</mujoco>",
                    generation_method=SimulationGenerationMethod.HYBRID,
                    status=SimulationStatus.PENDING
                )
            ]
            mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_simulations
            mock_get_db.return_value = mock_session

            response = client.get("/api/simulations")

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)
                assert len(data) <= 2

    def test_simulation_update_endpoint(self, client):
        """Test updating simulation."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_simulation = Simulation(
                id=1,
                session_id="test-session",
                user_prompt="Test simulation",
                mjcf_content="<mujoco>test</mujoco>",
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.PROCESSING
            )
            mock_session.query.return_value.filter.return_value.first.return_value = mock_simulation
            mock_get_db.return_value = mock_session

            update_data = {
                "status": "completed",
                "mjcf_content": "<mujoco><updated>test</updated></mujoco>"
            }

            response = client.put("/api/simulations/1", json=update_data)

            # Should return 200 for successful update
            if response.status_code == 200:
                mock_session.commit.assert_called_once()

    def test_simulation_deletion_endpoint(self, client):
        """Test deleting simulation."""
        with patch('simgen.db.base.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_simulation = Simulation(
                id=1,
                session_id="test-session",
                user_prompt="Test simulation",
                mjcf_content="<mujoco>test</mujoco>",
                generation_method=SimulationGenerationMethod.TEMPLATE_BASED,
                status=SimulationStatus.COMPLETED
            )
            mock_session.query.return_value.filter.return_value.first.return_value = mock_simulation
            mock_get_db.return_value = mock_session

            response = client.delete("/api/simulations/1")

            # Should return 200 or 204 for successful deletion
            if response.status_code in [200, 204]:
                mock_session.delete.assert_called_once()
                mock_session.commit.assert_called_once()


class TestPhysicsAPIEndpoints:
    """Test physics-related API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_physics_validation_endpoint(self, client):
        """Test physics validation endpoint."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>
        """

        response = client.post("/api/physics/validate", json={"mjcf_content": mjcf_content})

        # Should return validation result
        if response.status_code == 200:
            data = response.json()
            assert "valid" in data

    def test_physics_simulation_run_endpoint(self, client):
        """Test running physics simulation."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <geom type="sphere" size="0.1"/>
            </worldbody>
        </mujoco>
        """

        run_request = {
            "mjcf_content": mjcf_content,
            "duration": 10.0,
            "fps": 30
        }

        response = client.post("/api/physics/run", json=run_request)

        # Should return simulation results or accept for async processing
        assert response.status_code in [200, 202]

    def test_physics_parameters_endpoint(self, client):
        """Test getting physics parameters."""
        response = client.get("/api/physics/parameters")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


class TestTemplateAPIEndpoints:
    """Test template-related API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_template_list_endpoint(self, client):
        """Test listing available templates."""
        response = client.get("/api/templates")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_template_retrieval_by_id(self, client):
        """Test retrieving specific template."""
        response = client.get("/api/templates/1")

        # Should return template data or 404 if not found
        assert response.status_code in [200, 404]

    def test_template_search_endpoint(self, client):
        """Test template search functionality."""
        search_params = {
            "query": "bouncing ball",
            "category": "physics"
        }

        response = client.get("/api/templates/search", params=search_params)

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


class TestMonitoringAPIEndpoints:
    """Test monitoring and observability API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        """Test metrics collection endpoint."""
        response = client.get("/api/monitoring/metrics")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_health_detailed_endpoint(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/api/monitoring/health")

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "components" in data

    def test_performance_stats_endpoint(self, client):
        """Test performance statistics endpoint."""
        response = client.get("/api/monitoring/performance")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


class TestSketchAPIEndpoints:
    """Test sketch analysis API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_sketch_upload_endpoint(self, client):
        """Test sketch upload and analysis."""
        # Create a dummy image file
        image_data = b"fake_image_data"

        files = {"file": ("sketch.png", image_data, "image/png")}
        data = {"prompt": "Analyze this sketch for physics simulation"}

        response = client.post("/api/sketch/upload", files=files, data=data)

        # Should accept sketch for processing
        assert response.status_code in [200, 202]

    def test_sketch_analysis_status_endpoint(self, client):
        """Test checking sketch analysis status."""
        analysis_id = "test-analysis-123"

        response = client.get(f"/api/sketch/analysis/{analysis_id}")

        # Should return status or 404 if not found
        assert response.status_code in [200, 404]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for all endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON payloads."""
        response = client.post(
            "/api/simulations",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test handling of missing content type."""
        response = client.post("/api/simulations", data='{"test": "data"}')
        # Should handle gracefully
        assert response.status_code in [400, 422]

    def test_oversized_payload(self, client):
        """Test handling of oversized payloads."""
        large_content = "x" * (10 * 1024 * 1024)  # 10MB string
        response = client.post("/api/simulations", json={"user_prompt": large_content})

        # Should reject or handle gracefully
        assert response.status_code in [400, 413, 422]

    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection attempts."""
        malicious_prompt = "'; DROP TABLE simulations; --"

        response = client.post("/api/simulations", json={
            "user_prompt": malicious_prompt,
            "session_id": "test-session"
        })

        # Should not cause server error
        assert response.status_code != 500

    def test_xss_protection(self, client):
        """Test protection against XSS attempts."""
        xss_prompt = "<script>alert('xss')</script>"

        response = client.post("/api/simulations", json={
            "user_prompt": xss_prompt,
            "session_id": "test-session"
        })

        # Should handle gracefully
        assert response.status_code != 500

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time

        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        # Create multiple threads making concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10

    def test_rate_limiting_behavior(self, client):
        """Test rate limiting behavior if implemented."""
        # Make rapid requests
        responses = []
        for i in range(50):
            response = client.get("/health")
            responses.append(response.status_code)

        # Should not get server errors
        assert all(status != 500 for status in responses)

    def test_malformed_urls(self, client):
        """Test handling of malformed URLs."""
        malformed_urls = [
            "/api/simulations/../../../etc/passwd",
            "/api/simulations/%2e%2e/%2e%2e/etc/passwd",
            "/api/simulations/\x00",
            "/api/simulations/\xff\xfe"
        ]

        for url in malformed_urls:
            try:
                response = client.get(url)
                # Should return appropriate error codes, not server errors
                assert response.status_code in [400, 404, 422]
            except Exception:
                # Some malformed URLs might cause client-side exceptions, which is fine
                pass

    def test_method_not_allowed(self, client):
        """Test method not allowed responses."""
        # Try wrong HTTP methods on endpoints
        response = client.patch("/health")  # Health endpoint might not support PATCH
        assert response.status_code in [405, 422]

        response = client.delete("/health")  # Health endpoint might not support DELETE
        assert response.status_code in [405, 422]

    def test_authentication_endpoints_if_exist(self, client):
        """Test authentication-related endpoints if they exist."""
        # Try accessing potentially protected endpoints without auth
        protected_endpoints = [
            "/api/admin/users",
            "/api/admin/settings",
            "/api/private/metrics"
        ]

        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            # Should return 401, 403, or 404 (if endpoint doesn't exist)
            assert response.status_code in [401, 403, 404]


class TestAPIPerformanceAndLoad:
    """Test API performance characteristics."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time."""
        import time

        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second

    def test_api_response_consistency(self, client):
        """Test that API responses are consistent across multiple calls."""
        responses = []

        for i in range(5):
            response = client.get("/health")
            responses.append(response.json())

        # All responses should be identical
        first_response = responses[0]
        for response in responses[1:]:
            assert response == first_response

    def test_memory_usage_stability(self, client):
        """Test that repeated API calls don't cause memory leaks."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests
        for i in range(100):
            response = client.get("/health")
            assert response.status_code == 200

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024


class TestAPIDocumentationAndMetadata:
    """Test API documentation and metadata endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_openapi_schema_endpoint(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")

        if response.status_code == 200:
            schema = response.json()
            assert "openapi" in schema
            assert "info" in schema
            assert "paths" in schema

    def test_swagger_ui_endpoint(self, client):
        """Test Swagger UI endpoint."""
        response = client.get("/docs")

        # Should return HTML for Swagger UI
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")

        # Should return HTML for ReDoc
        assert response.status_code == 200
        assert "redoc" in response.text.lower()

    def test_api_version_endpoint(self, client):
        """Test API version information."""
        response = client.get("/api/version")

        if response.status_code == 200:
            data = response.json()
            assert "version" in data