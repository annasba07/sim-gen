"""
Test suite for simulation API endpoints
Tests the core sketch-to-physics functionality
"""

import pytest
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient

# Adjust imports based on the actual structure
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simgen.api.simulation import router
from simgen.models.schemas import SimulationRequest, SketchGenerationRequest
from simgen.services.sketch_analyzer import SketchAnalysisResult
from simgen.services.multimodal_enhancer import EnhancedPromptResult
from simgen.services.simulation_generator import GenerationResult, SimulationGenerationError
from simgen.models.schemas import ExtractedEntities, ObjectSchema, ConstraintSchema, EnvironmentSchema, GeometrySchema, MaterialSchema

# Test fixtures
@pytest.fixture
def sample_sketch_base64():
    """Sample base64 encoded image for testing"""
    # Create a small 1x1 PNG image in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

@pytest.fixture
def mock_sketch_analysis():
    """Mock successful sketch analysis result"""
    mock_entities = ExtractedEntities(
        objects=[
            ObjectSchema(
                name="robot_arm",
                type="robot",
                geometry=GeometrySchema(shape="cylinder", dimensions=[0.1, 0.5]),
                material=MaterialSchema(density=1000.0, friction=0.5, restitution=0.1),
                position=[0, 0, 0],
                orientation=[0, 0, 0]
            )
        ],
        constraints=[],
        environment=EnvironmentSchema(
            gravity=[0, 0, -9.81],
            ground={"type": "plane", "friction": 0.8}
        )
    )
    
    return SketchAnalysisResult(
        success=True,
        physics_description="A robotic arm system with joint articulation",
        extracted_entities=mock_entities,
        confidence_score=0.85,
        raw_vision_output="Robot arm detected with base and joints"
    )

@pytest.fixture
def mock_enhanced_result():
    """Mock enhanced prompt result"""
    mock_entities = ExtractedEntities(
        objects=[
            ObjectSchema(
                name="enhanced_robot_arm",
                type="robot",
                geometry=GeometrySchema(shape="cylinder", dimensions=[0.1, 0.5]),
                material=MaterialSchema(density=1000.0, friction=0.5, restitution=0.1),
                position=[0, 0, 0],
                orientation=[0, 0, 0]
            )
        ]
    )
    
    return EnhancedPromptResult(
        success=True,
        enhanced_prompt="Create a professional robotic arm simulation with realistic materials",
        confidence_score=0.9,
        sketch_contribution=0.7,
        text_contribution=0.3,
        combined_entities=mock_entities
    )

@pytest.fixture
def mock_generation_result():
    """Mock simulation generation result"""
    return GenerationResult(
        success=True,
        mjcf_content='<mujoco><worldbody><body name="robot_arm"><geom type="cylinder" size="0.1 0.5"/></body></worldbody></mujoco>',
        method=GenerationMethod.DYNAMIC_COMPOSITION,
        confidence_score=0.88,
        generation_time=2.5,
        error_message=None
    )


class TestSimulationAPI:
    """Test suite for simulation API endpoints"""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test basic API health check"""
        from fastapi.testclient import TestClient
        from simgen.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sketch_generate_missing_data(self):
        """Test sketch generation with missing required data"""
        from fastapi.testclient import TestClient
        from simgen.main import app
        
        client = TestClient(app)
        
        # Test with empty request
        response = client.post("/api/v1/simulation/sketch-generate", json={
            "sketch_data": "",
            "prompt": ""
        })
        
        # Should return error for missing data
        assert response.status_code in [400, 422]  # Bad request or validation error

    @pytest.mark.asyncio
    async def test_sketch_generate_success_flow(self, sample_sketch_base64, mock_sketch_analysis, 
                                                 mock_enhanced_result, mock_generation_result):
        """Test successful sketch-to-physics generation flow"""
        
        with patch('simgen.services.sketch_analyzer.get_sketch_analyzer') as mock_analyzer, \
             patch('simgen.services.multimodal_enhancer.get_multimodal_enhancer') as mock_enhancer, \
             patch('simgen.services.llm_client.get_llm_client') as mock_llm_client, \
             patch('simgen.services.simulation_generator.SimulationGenerator') as mock_generator:
            
            # Setup mocks
            analyzer_instance = AsyncMock()
            analyzer_instance.analyze_sketch.return_value = mock_sketch_analysis
            mock_analyzer.return_value = analyzer_instance
            
            enhancer_instance = AsyncMock()
            enhancer_instance.enhance_prompt.return_value = mock_enhanced_result
            mock_enhancer.return_value = enhancer_instance
            
            generator_instance = AsyncMock()
            generator_instance.generate_simulation.return_value = mock_generation_result
            mock_generator.return_value = generator_instance
            
            # Test the endpoint
            from fastapi.testclient import TestClient
            from simgen.main import app
            
            client = TestClient(app)
            
            response = client.post("/api/v1/simulation/sketch-generate", json={
                "sketch_data": sample_sketch_base64,
                "prompt": "Make this robot arm pick up a ball"
            })
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["status"] == "success"
            assert "generation_result" in data
            assert data["generation_result"]["success"] == True

    @pytest.mark.asyncio  
    async def test_sketch_generate_vision_failure(self, sample_sketch_base64):
        """Test sketch generation when vision analysis fails"""
        
        failed_analysis = SketchAnalysisResult(
            success=False,
            physics_description="",
            extracted_entities=None,
            confidence_score=0.0,
            raw_vision_output="",
            error_message="Vision analysis failed"
        )
        
        with patch('simgen.services.sketch_analyzer.get_sketch_analyzer') as mock_analyzer:
            analyzer_instance = AsyncMock()
            analyzer_instance.analyze_sketch.return_value = failed_analysis
            mock_analyzer.return_value = analyzer_instance
            
            from fastapi.testclient import TestClient
            from simgen.main import app
            
            client = TestClient(app)
            
            response = client.post("/api/v1/simulation/sketch-generate", json={
                "sketch_data": sample_sketch_base64,
                "prompt": "Test prompt"
            })
            
            # Should return partial success or error
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["partial", "error"]

    @pytest.mark.asyncio
    async def test_text_only_generation(self):
        """Test generation with text prompt only (no sketch)"""
        from fastapi.testclient import TestClient
        from simgen.main import app
        
        client = TestClient(app)
        
        response = client.post("/api/v1/simulation/test-generate", json={
            "prompt": "Create a simple pendulum simulation"
        })
        
        # Should handle text-only requests
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_base64_image(self):
        """Test handling of invalid base64 image data"""
        from fastapi.testclient import TestClient
        from simgen.main import app
        
        client = TestClient(app)
        
        response = client.post("/api/v1/simulation/sketch-generate", json={
            "sketch_data": "invalid_base64_data",
            "prompt": "Test prompt"
        })
        
        # Should handle invalid base64 gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "error" in data

    @pytest.mark.asyncio
    async def test_simulation_generation_timeout(self, sample_sketch_base64, mock_sketch_analysis, mock_enhanced_result):
        """Test handling of simulation generation timeout"""
        
        # Mock a timeout scenario
        timeout_result = GenerationResult(
            success=False,
            mjcf_content="",
            method=GenerationMethod.DYNAMIC_COMPOSITION,
            confidence_score=0.0,
            generation_time=0.0,
            error_message="Generation timeout"
        )
        
        with patch('simgen.services.sketch_analyzer.get_sketch_analyzer') as mock_analyzer, \
             patch('simgen.services.multimodal_enhancer.get_multimodal_enhancer') as mock_enhancer, \
             patch('simgen.services.simulation_generator.SimulationGenerator') as mock_generator:
            
            analyzer_instance = AsyncMock()
            analyzer_instance.analyze_sketch.return_value = mock_sketch_analysis
            mock_analyzer.return_value = analyzer_instance
            
            enhancer_instance = AsyncMock()
            enhancer_instance.enhance_prompt.return_value = mock_enhanced_result
            mock_enhancer.return_value = enhancer_instance
            
            generator_instance = AsyncMock()
            generator_instance.generate_simulation.return_value = timeout_result
            mock_generator.return_value = generator_instance
            
            from fastapi.testclient import TestClient
            from simgen.main import app
            
            client = TestClient(app)
            
            response = client.post("/api/v1/simulation/sketch-generate", json={
                "sketch_data": sample_sketch_base64,
                "prompt": "Test prompt"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == False


class TestDataValidation:
    """Test data validation and edge cases"""

    def test_sketch_request_validation(self):
        """Test SketchGenerationRequest validation"""
        
        # Valid request
        valid_request = SketchGenerationRequest(
            sketch_data="valid_base64_data",
            prompt="Test prompt"
        )
        assert valid_request.sketch_data == "valid_base64_data"
        
        # Test with optional fields
        request_with_optional = SketchGenerationRequest(
            sketch_data="data",
            prompt="prompt",
            style_preferences={"style": "realistic"}
        )
        assert request_with_optional.style_preferences == {"style": "realistic"}

    def test_simulation_request_validation(self):
        """Test SimulationRequest validation"""
        
        # Valid request
        valid_request = SimulationRequest(
            prompt="Create a pendulum"
        )
        assert valid_request.prompt == "Create a pendulum"
        
        # Test prompt length validation
        with pytest.raises(ValueError):
            SimulationRequest(prompt="short")  # Too short
            
        with pytest.raises(ValueError):
            SimulationRequest(prompt="x" * 2001)  # Too long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])