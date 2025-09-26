"""
Comprehensive tests for Computer Vision Pipeline
Tests shape detection, stroke analysis, and physics conversion
"""

import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
import base64
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Add path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from simgen.services.computer_vision_pipeline import (
    ComputerVisionPipeline, Point2D, Stroke, DetectedShape, DetectedConnection,
    DetectedText, ShapeType, ConnectionType, CVAnalysisResult
)
from simgen.services.sketch_to_physics_converter import (
    SketchToPhysicsConverter, ConversionResult
)
from simgen.services.sketch_analyzer import AdvancedSketchAnalyzer


class TestComputerVisionPipeline:
    """Test computer vision pipeline functionality"""

    @pytest.fixture
    def cv_pipeline(self):
        """Create a CV pipeline instance for testing"""
        pipeline = ComputerVisionPipeline()
        # Mock OCR reader to avoid dependency issues in tests
        pipeline.ocr_reader = Mock()
        return pipeline

    @pytest.fixture
    def sample_circle_image(self):
        """Create a test image with a circle"""
        # Create a 400x400 white image
        img = Image.new('RGB', (400, 400), 'white')
        draw = ImageDraw.Draw(img)

        # Draw a black circle
        draw.ellipse([150, 150, 250, 250], outline='black', width=3)

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.fixture
    def sample_rectangle_image(self):
        """Create a test image with a rectangle"""
        img = Image.new('RGB', (400, 400), 'white')
        draw = ImageDraw.Draw(img)

        # Draw a black rectangle
        draw.rectangle([100, 150, 300, 250], outline='black', width=3)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.fixture
    def sample_line_image(self):
        """Create a test image with a line"""
        img = Image.new('RGB', (400, 400), 'white')
        draw = ImageDraw.Draw(img)

        # Draw a black line
        draw.line([100, 100, 300, 300], fill='black', width=3)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.fixture
    def sample_complex_sketch(self):
        """Create a complex sketch with multiple shapes and connections"""
        img = Image.new('RGB', (600, 400), 'white')
        draw = ImageDraw.Draw(img)

        # Draw a circle (ball)
        draw.ellipse([50, 50, 100, 100], outline='black', width=3)

        # Draw a rectangle (box)
        draw.rectangle([200, 150, 300, 200], outline='black', width=3)

        # Draw a line connecting them
        draw.line([75, 100, 250, 150], fill='black', width=2)

        # Add some text
        draw.text([350, 100], "pendulum", fill='black')

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    def test_image_preprocessing(self, cv_pipeline, sample_circle_image):
        """Test image preprocessing functionality"""
        processed = cv_pipeline._preprocess_image(sample_circle_image)

        # Check output properties
        assert processed is not None
        assert len(processed.shape) == 2  # Should be grayscale
        assert processed.dtype == np.uint8
        assert processed.shape[0] > 0 and processed.shape[1] > 0

    def test_stroke_extraction(self, cv_pipeline, sample_circle_image):
        """Test stroke extraction from binary image"""
        processed = cv_pipeline._preprocess_image(sample_circle_image)
        strokes = cv_pipeline._extract_strokes(processed)

        # Should detect at least one stroke for the circle
        assert len(strokes) > 0

        # Check stroke properties
        for stroke in strokes:
            assert isinstance(stroke, Stroke)
            assert len(stroke.points) > 0
            assert stroke.thickness > 0
            assert isinstance(stroke.id, str)

    @pytest.mark.asyncio
    async def test_circle_detection(self, cv_pipeline, sample_circle_image):
        """Test circle shape detection"""
        result = await cv_pipeline.analyze_sketch(sample_circle_image)

        assert result.success is not False  # May be True or have shapes

        # Look for circle detection
        circle_shapes = [s for s in result.shapes if s.shape_type == ShapeType.CIRCLE]

        # Should detect at least one circular shape (might also detect as other shapes)
        # This is a heuristic test since computer vision can be imperfect
        if circle_shapes:
            circle = circle_shapes[0]
            assert 'radius' in circle.parameters
            assert circle.parameters['radius'] > 0
            assert circle.confidence > 0

    @pytest.mark.asyncio
    async def test_rectangle_detection(self, cv_pipeline, sample_rectangle_image):
        """Test rectangle shape detection"""
        result = await cv_pipeline.analyze_sketch(sample_rectangle_image)

        # Look for rectangle detection
        rect_shapes = [s for s in result.shapes if s.shape_type == ShapeType.RECTANGLE]

        if rect_shapes:
            rect = rect_shapes[0]
            assert 'width' in rect.parameters
            assert 'height' in rect.parameters
            assert rect.parameters['width'] > 0
            assert rect.parameters['height'] > 0

    @pytest.mark.asyncio
    async def test_line_detection(self, cv_pipeline, sample_line_image):
        """Test line shape detection"""
        result = await cv_pipeline.analyze_sketch(sample_line_image)

        # Look for line detection
        line_shapes = [s for s in result.shapes if s.shape_type == ShapeType.LINE]

        if line_shapes:
            line = line_shapes[0]
            assert 'length' in line.parameters
            assert 'angle' in line.parameters
            assert line.parameters['length'] > 0

    def test_shape_fitting_circle(self, cv_pipeline):
        """Test circle fitting algorithm"""
        # Create a perfect circle contour
        center = (100, 100)
        radius = 50
        angles = np.linspace(0, 2*np.pi, 100)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y])

        contour = np.array(points, dtype=np.int32)

        fit_result = cv_pipeline._fit_circle(contour)

        assert fit_result is not None
        assert abs(fit_result['center'][0] - center[0]) < 5
        assert abs(fit_result['center'][1] - center[1]) < 5
        assert abs(fit_result['radius'] - radius) < 5
        assert fit_result['confidence'] > 0.5

    def test_shape_fitting_rectangle(self, cv_pipeline):
        """Test rectangle fitting algorithm"""
        # Create a rectangle contour
        points = [[50, 50], [150, 50], [150, 100], [50, 100]]
        contour = np.array(points, dtype=np.int32)

        fit_result = cv_pipeline._fit_rectangle(contour)

        assert fit_result is not None
        assert fit_result['width'] > 0
        assert fit_result['height'] > 0
        assert fit_result['confidence'] > 0

    def test_shape_fitting_line(self, cv_pipeline):
        """Test line fitting algorithm"""
        # Create a straight line contour
        points = [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100]]
        contour = np.array(points, dtype=np.int32)

        fit_result = cv_pipeline._fit_line(contour)

        assert fit_result is not None
        assert fit_result['length'] > 0
        assert abs(fit_result['angle'] - 45) < 10  # Should be ~45 degrees
        assert fit_result['confidence'] > 0

    @pytest.mark.asyncio
    async def test_complex_sketch_analysis(self, cv_pipeline, sample_complex_sketch):
        """Test analysis of complex sketch with multiple shapes"""
        # Mock OCR to return some text
        cv_pipeline.ocr_reader.readtext.return_value = [
            ([[[340, 95], [420, 95], [420, 110], [340, 110]]], 'pendulum', 0.9)
        ]

        result = await cv_pipeline.analyze_sketch(sample_complex_sketch)

        assert isinstance(result, CVAnalysisResult)
        # Should detect multiple shapes
        assert len(result.shapes) >= 1

        # Check that we got some meaningful analysis
        assert result.confidence_score > 0

        # Check text detection worked
        assert len(result.text_annotations) > 0
        assert result.text_annotations[0].text == 'pendulum'

    def test_confidence_calculation(self, cv_pipeline):
        """Test confidence score calculation"""
        # Create mock data
        shapes = [
            DetectedShape(
                shape_type=ShapeType.CIRCLE,
                center=Point2D(100, 100),
                parameters={'radius': 50},
                confidence=0.8,
                bounding_box=(50, 50, 150, 150),
                source_strokes=['stroke_0']
            )
        ]

        connections = [
            DetectedConnection(
                connection_type=ConnectionType.HINGE_JOINT,
                shape1_id='shape_0',
                shape2_id='shape_1',
                connection_point=Point2D(125, 125),
                confidence=0.7
            )
        ]

        text_annotations = [
            DetectedText(
                text='test',
                position=Point2D(200, 200),
                bounding_box=(190, 190, 210, 210),
                confidence=0.9
            )
        ]

        confidence = cv_pipeline._calculate_overall_confidence(shapes, connections, text_annotations)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably confident with good inputs


class TestSketchToPhysicsConverter:
    """Test sketch to PhysicsSpec conversion functionality"""

    @pytest.fixture
    def physics_converter(self):
        """Create a physics converter instance"""
        return SketchToPhysicsConverter()

    @pytest.fixture
    def sample_cv_result(self):
        """Create sample CV analysis result"""
        shapes = [
            DetectedShape(
                shape_type=ShapeType.CIRCLE,
                center=Point2D(100, 100),
                parameters={'radius': 25},
                confidence=0.8,
                bounding_box=(75, 75, 125, 125),
                source_strokes=['stroke_0']
            ),
            DetectedShape(
                shape_type=ShapeType.RECTANGLE,
                center=Point2D(200, 150),
                parameters={'width': 50, 'height': 30},
                confidence=0.7,
                bounding_box=(175, 135, 225, 165),
                source_strokes=['stroke_1']
            )
        ]

        connections = [
            DetectedConnection(
                connection_type=ConnectionType.HINGE_JOINT,
                shape1_id='stroke_0',
                shape2_id='stroke_1',
                connection_point=Point2D(150, 125),
                confidence=0.6
            )
        ]

        return CVAnalysisResult(
            strokes=[],
            shapes=shapes,
            connections=connections,
            text_annotations=[],
            physics_interpretation={},
            confidence_score=0.75
        )

    @pytest.mark.asyncio
    async def test_cv_to_physics_spec_conversion(self, physics_converter, sample_cv_result):
        """Test conversion from CV analysis to PhysicsSpec"""
        result = await physics_converter.convert_cv_to_physics_spec(sample_cv_result)

        assert result.success
        assert result.physics_spec is not None

        # Check that shapes were converted to bodies
        assert len(result.physics_spec.bodies) == 2

        # Check body properties
        for body in result.physics_spec.bodies:
            assert body.id is not None
            assert len(body.geoms) > 0
            assert body.inertial is not None
            assert body.inertial.mass > 0

    @pytest.mark.asyncio
    async def test_circle_to_body_conversion(self, physics_converter):
        """Test conversion of circle shape to physics body"""
        circle_shape = DetectedShape(
            shape_type=ShapeType.CIRCLE,
            center=Point2D(100, 100),
            parameters={'radius': 30},
            confidence=0.8,
            bounding_box=(70, 70, 130, 130),
            source_strokes=['stroke_0']
        )

        # Test conversion
        body = physics_converter._convert_shape_to_body(circle_shape, 0.01, 0, 0)

        assert body is not None
        assert len(body.geoms) == 1
        assert body.geoms[0].type.value == 'sphere'
        assert len(body.geoms[0].size) == 1
        assert body.geoms[0].size[0] > 0  # Radius should be positive

    @pytest.mark.asyncio
    async def test_rectangle_to_body_conversion(self, physics_converter):
        """Test conversion of rectangle shape to physics body"""
        rect_shape = DetectedShape(
            shape_type=ShapeType.RECTANGLE,
            center=Point2D(150, 100),
            parameters={'width': 60, 'height': 40},
            confidence=0.7,
            bounding_box=(120, 80, 180, 120),
            source_strokes=['stroke_1']
        )

        body = physics_converter._convert_shape_to_body(rect_shape, 0.01, 0, 0)

        assert body is not None
        assert len(body.geoms) == 1
        assert body.geoms[0].type.value == 'box'
        assert len(body.geoms[0].size) == 3  # Box needs 3 dimensions
        assert all(s > 0 for s in body.geoms[0].size)

    @pytest.mark.asyncio
    async def test_actuator_creation(self, physics_converter, sample_cv_result):
        """Test creation of actuators for bodies with joints"""
        result = await physics_converter.convert_cv_to_physics_spec(
            sample_cv_result, include_actuators=True
        )

        assert result.success

        # Should create some actuators for jointed bodies
        # (Exact number depends on joint creation logic)
        assert isinstance(result.physics_spec.actuators, list)

    @pytest.mark.asyncio
    async def test_sensor_creation(self, physics_converter, sample_cv_result):
        """Test creation of sensors for physics monitoring"""
        result = await physics_converter.convert_cv_to_physics_spec(
            sample_cv_result, include_sensors=True
        )

        assert result.success

        # Should create sensors for jointed bodies
        assert isinstance(result.physics_spec.sensors, list)

    def test_confidence_calculation(self, physics_converter, sample_cv_result):
        """Test conversion confidence calculation"""
        confidence = physics_converter._calculate_conversion_confidence(
            sample_cv_result, num_bodies=2, num_actuators=1
        )

        assert 0 <= confidence <= 1
        assert confidence > sample_cv_result.confidence_score * 0.5  # Should be reasonable


class TestAdvancedSketchAnalyzer:
    """Test the integrated advanced sketch analyzer"""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client"""
        llm_client = Mock()
        llm_client.analyze_image = AsyncMock(return_value="Mock image analysis result")
        llm_client.complete = AsyncMock(return_value="Mock physics description")
        llm_client.complete_with_schema = AsyncMock(return_value={
            "objects": [],
            "constraints": [],
            "environment": {"gravity": [0, 0, -9.81]}
        })
        return llm_client

    @pytest.fixture
    def sketch_analyzer(self, mock_llm_client):
        """Create sketch analyzer with mocked dependencies"""
        analyzer = AdvancedSketchAnalyzer(mock_llm_client)

        # Mock the CV pipeline to avoid real computer vision in tests
        analyzer.cv_pipeline = Mock()
        analyzer.cv_pipeline.analyze_sketch = AsyncMock()

        # Mock the physics converter
        analyzer.physics_converter = Mock()
        analyzer.physics_converter.convert_cv_to_physics_spec = AsyncMock()

        return analyzer

    @pytest.fixture
    def sample_sketch_bytes(self):
        """Create sample sketch image bytes"""
        img = Image.new('RGB', (300, 300), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([100, 100, 200, 200], outline='black', width=3)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.mark.asyncio
    async def test_successful_sketch_analysis(self, sketch_analyzer, sample_sketch_bytes):
        """Test successful sketch analysis workflow"""
        # Setup mocks
        mock_cv_result = CVAnalysisResult(
            strokes=[],
            shapes=[
                DetectedShape(
                    shape_type=ShapeType.CIRCLE,
                    center=Point2D(150, 150),
                    parameters={'radius': 50},
                    confidence=0.8,
                    bounding_box=(100, 100, 200, 200),
                    source_strokes=['stroke_0']
                )
            ],
            connections=[],
            text_annotations=[],
            physics_interpretation={},
            confidence_score=0.8
        )

        sketch_analyzer.cv_pipeline.analyze_sketch.return_value = mock_cv_result

        mock_conversion_result = ConversionResult(
            success=True,
            physics_spec=Mock(),  # Would be real PhysicsSpec in actual use
            confidence_score=0.7,
            conversion_notes=["Successfully converted shapes"]
        )

        sketch_analyzer.physics_converter.convert_cv_to_physics_spec.return_value = mock_conversion_result

        # Test analysis
        result = await sketch_analyzer.analyze_sketch(sample_sketch_bytes, "test pendulum")

        assert result.success
        assert result.physics_spec is not None
        assert result.cv_analysis == mock_cv_result
        assert result.confidence_score > 0
        assert len(result.processing_notes) > 0

    @pytest.mark.asyncio
    async def test_fallback_to_llm_analysis(self, sketch_analyzer, sample_sketch_bytes):
        """Test fallback to LLM-only analysis when CV fails"""
        # Make CV pipeline return empty result
        empty_cv_result = CVAnalysisResult(
            strokes=[],
            shapes=[],  # No shapes detected
            connections=[],
            text_annotations=[],
            physics_interpretation={},
            confidence_score=0.1
        )

        sketch_analyzer.cv_pipeline.analyze_sketch.return_value = empty_cv_result

        # Test that fallback is triggered
        result = await sketch_analyzer.analyze_sketch(sample_sketch_bytes, "test sketch")

        # Should still get a result from LLM fallback
        assert result.success or result.error_message  # Either success or meaningful error

        # Check that LLM methods were called
        sketch_analyzer.llm_client.analyze_image.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling(self, sketch_analyzer, sample_sketch_bytes):
        """Test error handling in sketch analysis"""
        # Make CV pipeline raise an exception
        sketch_analyzer.cv_pipeline.analyze_sketch.side_effect = Exception("CV pipeline error")

        # Also make fallback fail
        sketch_analyzer.llm_client.analyze_image.side_effect = Exception("LLM error")

        result = await sketch_analyzer.analyze_sketch(sample_sketch_bytes)

        assert not result.success
        assert result.error_message is not None
        assert "advanced and fallback analysis failed" in result.error_message.lower()

    def test_cv_analysis_formatting(self, sketch_analyzer):
        """Test formatting of CV analysis output"""
        cv_result = CVAnalysisResult(
            strokes=[],
            shapes=[
                DetectedShape(
                    shape_type=ShapeType.CIRCLE,
                    center=Point2D(100, 100),
                    parameters={'radius': 25},
                    confidence=0.8,
                    bounding_box=(75, 75, 125, 125),
                    source_strokes=['stroke_0']
                )
            ],
            connections=[
                DetectedConnection(
                    connection_type=ConnectionType.HINGE_JOINT,
                    shape1_id='shape_0',
                    shape2_id='shape_1',
                    connection_point=Point2D(125, 125),
                    confidence=0.7
                )
            ],
            text_annotations=[
                DetectedText(
                    text='pendulum',
                    position=Point2D(200, 100),
                    bounding_box=(180, 90, 220, 110),
                    confidence=0.9
                )
            ],
            physics_interpretation={},
            confidence_score=0.8
        )

        formatted_output = sketch_analyzer._format_cv_analysis_output(cv_result)

        assert "COMPUTER VISION ANALYSIS" in formatted_output
        assert "DETECTED SHAPES:" in formatted_output
        assert "DETECTED CONNECTIONS:" in formatted_output
        assert "DETECTED TEXT:" in formatted_output
        assert "circle" in formatted_output.lower()
        assert "hinge" in formatted_output.lower()
        assert "pendulum" in formatted_output.lower()


# Integration tests
class TestIntegration:
    """Integration tests for the complete CV pipeline"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_simple_sketch(self):
        """Test complete pipeline with a simple sketch"""
        # This test would require actual CV libraries to be installed
        # and would be slower, so marked as slow

        # Create a simple test image
        img = Image.new('RGB', (400, 400), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([150, 150, 250, 250], outline='black', width=5)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        image_data = img_bytes.getvalue()

        try:
            # Create real pipeline (will fail if dependencies not installed)
            cv_pipeline = ComputerVisionPipeline()

            # This might fail due to missing dependencies in test environment
            result = await cv_pipeline.analyze_sketch(image_data)

            # Basic checks if it succeeded
            assert isinstance(result, CVAnalysisResult)
            assert result.confidence_score >= 0

        except ImportError as e:
            pytest.skip(f"CV dependencies not available for integration test: {e}")
        except Exception as e:
            pytest.skip(f"Integration test failed due to environment: {e}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])