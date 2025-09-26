"""
Advanced Computer Vision Pipeline for Sketch Analysis
Transforms hand-drawn sketches into structured physics representations using
sophisticated image processing, shape detection, and stroke analysis.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import base64
from io import BytesIO
from PIL import Image, ImageOps
import easyocr
from scipy import ndimage
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import json

logger = logging.getLogger(__name__)

class ShapeType(Enum):
    """Detected shape types"""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    LINE = "line"
    ARC = "arc"
    POLYGON = "polygon"
    SPRING = "spring"
    ARROW = "arrow"
    UNKNOWN = "unknown"

class ConnectionType(Enum):
    """Types of connections between objects"""
    HINGE_JOINT = "hinge"
    SLIDER_JOINT = "slide"
    FIXED_JOINT = "fixed"
    SPRING = "spring"
    CONTACT = "contact"
    FREE = "free"

@dataclass
class Point2D:
    """2D point with additional properties"""
    x: float
    y: float
    pressure: float = 1.0
    timestamp: float = 0.0

    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Stroke:
    """A single stroke (continuous pen movement)"""
    points: List[Point2D]
    id: str
    thickness: float = 2.0
    color: Tuple[int, int, int] = (0, 0, 0)
    is_closed: bool = False

    @property
    def length(self) -> float:
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i].distance_to(self.points[i-1])
        return total

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)"""
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

@dataclass
class DetectedShape:
    """A detected geometric shape"""
    shape_type: ShapeType
    center: Point2D
    parameters: Dict[str, float]  # radius for circle, width/height for rect, etc.
    confidence: float
    bounding_box: Tuple[float, float, float, float]
    source_strokes: List[str]  # IDs of strokes that form this shape
    physics_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedConnection:
    """A detected connection between shapes"""
    connection_type: ConnectionType
    shape1_id: str
    shape2_id: str
    connection_point: Point2D
    parameters: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class DetectedText:
    """Detected text annotation"""
    text: str
    position: Point2D
    bounding_box: Tuple[float, float, float, float]
    confidence: float
    associated_shape_id: Optional[str] = None

@dataclass
class CVAnalysisResult:
    """Complete computer vision analysis result"""
    strokes: List[Stroke]
    shapes: List[DetectedShape]
    connections: List[DetectedConnection]
    text_annotations: List[DetectedText]
    physics_interpretation: Dict[str, Any]
    confidence_score: float
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class ComputerVisionPipeline:
    """Advanced computer vision pipeline for sketch analysis"""

    def __init__(self):
        self.ocr_reader = None
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize OCR reader"""
        try:
            # Initialize EasyOCR with English
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}. Text detection will be disabled.")

    async def analyze_sketch(self, image_data: Union[bytes, np.ndarray]) -> CVAnalysisResult:
        """
        Main entry point for sketch analysis

        Args:
            image_data: Either raw bytes or numpy array of the image

        Returns:
            Complete computer vision analysis result
        """
        try:
            # Step 1: Preprocess image
            processed_image = self._preprocess_image(image_data)

            # Step 2: Extract strokes
            strokes = self._extract_strokes(processed_image)

            # Step 3: Detect shapes
            shapes = self._detect_shapes(strokes, processed_image)

            # Step 4: Detect connections
            connections = self._detect_connections(shapes, strokes)

            # Step 5: Extract text annotations
            text_annotations = self._extract_text(processed_image)

            # Step 6: Associate text with shapes
            self._associate_text_with_shapes(text_annotations, shapes)

            # Step 7: Interpret physics
            physics_interpretation = self._interpret_physics(shapes, connections, text_annotations)

            # Step 8: Calculate confidence
            confidence = self._calculate_overall_confidence(shapes, connections, text_annotations)

            return CVAnalysisResult(
                strokes=strokes,
                shapes=shapes,
                connections=connections,
                text_annotations=text_annotations,
                physics_interpretation=physics_interpretation,
                confidence_score=confidence,
                processing_metadata={
                    'image_size': processed_image.shape[:2],
                    'num_strokes': len(strokes),
                    'num_shapes': len(shapes),
                    'processing_time': 0.0  # Could add timing
                }
            )

        except Exception as e:
            logger.error(f"Computer vision analysis failed: {e}")
            return CVAnalysisResult(
                strokes=[],
                shapes=[],
                connections=[],
                text_annotations=[],
                physics_interpretation={},
                confidence_score=0.0,
                processing_metadata={'error': str(e)}
            )

    def _preprocess_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess the input image for analysis
        """
        if isinstance(image_data, bytes):
            # Decode base64 if needed
            if image_data.startswith(b'data:image'):
                # Remove data URL prefix
                image_data = base64.b64decode(image_data.split(b',')[1])

            # Load image from bytes
            image = Image.open(BytesIO(image_data))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image_data.copy()

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Noise reduction
        denoised = cv2.medianBlur(enhanced, 3)

        # Ensure binary image (sketch should be mostly black on white)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (make drawing pixels white)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        return binary

    def _extract_strokes(self, binary_image: np.ndarray) -> List[Stroke]:
        """
        Extract individual strokes from the binary image
        Uses skeletonization and contour analysis
        """
        strokes = []

        # Invert for processing (make drawing pixels white)
        inverted = cv2.bitwise_not(binary_image)

        # Skeletonize to get stroke centerlines
        skeleton = self._skeletonize(inverted)

        # Find contours in the skeleton
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        stroke_id = 0
        for contour in contours:
            if len(contour) < 5:  # Skip very short contours
                continue

            # Convert contour to stroke points
            points = []
            for point in contour:
                x, y = point[0]
                points.append(Point2D(float(x), float(y)))

            # Determine if stroke is closed
            is_closed = len(points) > 10 and points[0].distance_to(points[-1]) < 10

            # Estimate thickness from original image
            thickness = self._estimate_stroke_thickness(binary_image, contour)

            stroke = Stroke(
                points=points,
                id=f"stroke_{stroke_id}",
                thickness=thickness,
                is_closed=is_closed
            )
            strokes.append(stroke)
            stroke_id += 1

        return strokes

    def _skeletonize(self, image: np.ndarray) -> np.ndarray:
        """
        Skeletonize binary image to extract stroke centerlines
        """
        # Morphological thinning
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Zhang-Suen thinning algorithm approximation
        size = np.size(image)
        skel = np.zeros(image.shape, np.uint8)

        ret, img = cv2.threshold(image, 127, 255, 0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        return skel

    def _estimate_stroke_thickness(self, image: np.ndarray, contour: np.ndarray) -> float:
        """
        Estimate the thickness of a stroke from its contour
        """
        # Create mask for this contour
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Find the stroke in the original image
        stroke_region = cv2.bitwise_and(cv2.bitwise_not(image), mask)

        # Use distance transform to find thickness
        dist_transform = cv2.distanceTransform(stroke_region, cv2.DIST_L2, 5)

        # Average the non-zero distances as thickness estimate
        non_zero_dists = dist_transform[dist_transform > 0]
        if len(non_zero_dists) > 0:
            return float(np.mean(non_zero_dists) * 2)  # Multiply by 2 for diameter
        else:
            return 2.0  # Default thickness

    def _detect_shapes(self, strokes: List[Stroke], image: np.ndarray) -> List[DetectedShape]:
        """
        Detect geometric shapes from strokes
        """
        shapes = []
        shape_id = 0

        for stroke in strokes:
            detected_shape = self._classify_stroke_as_shape(stroke, image)
            if detected_shape:
                detected_shape.source_strokes = [stroke.id]
                shapes.append(detected_shape)
                shape_id += 1

        # Merge similar shapes that might be drawn as multiple strokes
        merged_shapes = self._merge_similar_shapes(shapes)

        return merged_shapes

    def _classify_stroke_as_shape(self, stroke: Stroke, image: np.ndarray) -> Optional[DetectedShape]:
        """
        Classify a single stroke as a geometric shape
        """
        points = [(p.x, p.y) for p in stroke.points]
        if len(points) < 3:
            return None

        # Convert to numpy array for processing
        contour = np.array(points, dtype=np.int32)

        # Calculate various shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 50:  # Skip very small shapes
            return None

        # Fit different shapes and determine best match
        circle_fit = self._fit_circle(contour)
        rectangle_fit = self._fit_rectangle(contour)
        line_fit = self._fit_line(contour)

        # Determine best fit based on error metrics
        best_shape = None
        best_confidence = 0.0

        # Check circle fit
        if circle_fit and circle_fit['confidence'] > best_confidence:
            best_shape = DetectedShape(
                shape_type=ShapeType.CIRCLE,
                center=Point2D(circle_fit['center'][0], circle_fit['center'][1]),
                parameters={'radius': circle_fit['radius']},
                confidence=circle_fit['confidence'],
                bounding_box=stroke.bounding_box,
                source_strokes=[stroke.id]
            )
            best_confidence = circle_fit['confidence']

        # Check rectangle fit
        if rectangle_fit and rectangle_fit['confidence'] > best_confidence:
            best_shape = DetectedShape(
                shape_type=ShapeType.RECTANGLE,
                center=Point2D(rectangle_fit['center'][0], rectangle_fit['center'][1]),
                parameters={
                    'width': rectangle_fit['width'],
                    'height': rectangle_fit['height'],
                    'angle': rectangle_fit['angle']
                },
                confidence=rectangle_fit['confidence'],
                bounding_box=stroke.bounding_box,
                source_strokes=[stroke.id]
            )
            best_confidence = rectangle_fit['confidence']

        # Check line fit
        if line_fit and line_fit['confidence'] > best_confidence:
            best_shape = DetectedShape(
                shape_type=ShapeType.LINE,
                center=Point2D(line_fit['center'][0], line_fit['center'][1]),
                parameters={
                    'length': line_fit['length'],
                    'angle': line_fit['angle'],
                    'start_x': line_fit['start'][0],
                    'start_y': line_fit['start'][1],
                    'end_x': line_fit['end'][0],
                    'end_y': line_fit['end'][1]
                },
                confidence=line_fit['confidence'],
                bounding_box=stroke.bounding_box,
                source_strokes=[stroke.id]
            )
            best_confidence = line_fit['confidence']

        return best_shape

    def _fit_circle(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fit a circle to a contour and return fit parameters
        """
        if len(contour) < 5:
            return None

        # Fit enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Calculate how well the contour matches a circle
        center = np.array([x, y])
        distances = [np.linalg.norm(point - center) for point in contour.reshape(-1, 2)]

        # Check circularity
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Confidence based on how consistent the distances are
        confidence = max(0, 1.0 - (std_dist / mean_dist) if mean_dist > 0 else 0)

        # Additional check: ratio of area to perimeter should be close to circle
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence *= circularity  # Multiply by circularity factor

        if confidence > 0.3:  # Threshold for accepting as circle
            return {
                'center': (float(x), float(y)),
                'radius': float(radius),
                'confidence': float(confidence)
            }

        return None

    def _fit_rectangle(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fit a rectangle to a contour
        """
        if len(contour) < 4:
            return None

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rect

        # Get the rectangle vertices
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate how well the contour matches the rectangle
        rect_area = width * height
        contour_area = cv2.contourArea(contour)

        if rect_area == 0:
            return None

        # Confidence based on area match and contour approximation
        area_ratio = contour_area / rect_area

        # Check if contour can be approximated by 4 points (rectangle-like)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Confidence increases if approximation has ~4 corners
        corner_confidence = 1.0 - abs(len(approx) - 4) / 10.0
        corner_confidence = max(0, corner_confidence)

        confidence = area_ratio * corner_confidence

        if confidence > 0.3:  # Threshold for accepting as rectangle
            return {
                'center': (float(center_x), float(center_y)),
                'width': float(width),
                'height': float(height),
                'angle': float(angle),
                'confidence': float(confidence)
            }

        return None

    def _fit_line(self, contour: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Fit a line to a contour
        """
        if len(contour) < 2:
            return None

        # Fit line using least squares
        points = contour.reshape(-1, 2).astype(np.float32)

        # Use cv2.fitLine
        vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate line endpoints
        lefty = int((-x * vy / vx) + y) if vx != 0 else int(y)
        righty = int(((points[:, 0].max() - x) * vy / vx) + y) if vx != 0 else int(y)

        start_point = (int(points[:, 0].min()), lefty)
        end_point = (int(points[:, 0].max()), righty)

        # Calculate line length
        length = np.linalg.norm(np.array(end_point) - np.array(start_point))

        # Calculate fit quality
        # Distance of each point from the fitted line
        line_point = np.array([x, y])
        line_direction = np.array([vx, vy])

        distances = []
        for point in points:
            # Vector from line point to contour point
            to_point = point - line_point
            # Distance from line (perpendicular distance)
            distance = np.abs(np.cross(to_point, line_direction))
            distances.append(distance)

        mean_distance = np.mean(distances)

        # Confidence based on how close points are to the fitted line
        confidence = max(0, 1.0 - (mean_distance / max(1.0, length * 0.1)))

        # Check if contour is roughly linear (aspect ratio check)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        if bbox_width > 0 and bbox_height > 0:
            aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height)
            if aspect_ratio > 3:  # Line-like aspect ratio
                confidence *= 1.5  # Boost confidence for line-like shapes

        if confidence > 0.4:  # Threshold for accepting as line
            angle = math.degrees(math.atan2(vy, vx))
            center_x = (start_point[0] + end_point[0]) / 2
            center_y = (start_point[1] + end_point[1]) / 2

            return {
                'center': (float(center_x), float(center_y)),
                'length': float(length),
                'angle': float(angle),
                'start': start_point,
                'end': end_point,
                'confidence': float(confidence)
            }

        return None

    def _merge_similar_shapes(self, shapes: List[DetectedShape]) -> List[DetectedShape]:
        """
        Merge shapes that are likely the same object drawn with multiple strokes
        """
        if len(shapes) < 2:
            return shapes

        merged_shapes = []
        used_indices = set()

        for i, shape1 in enumerate(shapes):
            if i in used_indices:
                continue

            # Look for similar shapes to merge
            merge_candidates = [shape1]
            used_indices.add(i)

            for j, shape2 in enumerate(shapes[i+1:], i+1):
                if j in used_indices:
                    continue

                if self._should_merge_shapes(shape1, shape2):
                    merge_candidates.append(shape2)
                    used_indices.add(j)

            # If multiple candidates, merge them
            if len(merge_candidates) > 1:
                merged_shape = self._merge_shape_group(merge_candidates)
                merged_shapes.append(merged_shape)
            else:
                merged_shapes.append(shape1)

        return merged_shapes

    def _should_merge_shapes(self, shape1: DetectedShape, shape2: DetectedShape) -> bool:
        """
        Determine if two shapes should be merged
        """
        # Only merge shapes of the same type
        if shape1.shape_type != shape2.shape_type:
            return False

        # Check if centers are close
        center_distance = shape1.center.distance_to(shape2.center)

        # Distance threshold based on shape size
        if shape1.shape_type == ShapeType.CIRCLE:
            max_distance = max(shape1.parameters.get('radius', 50), shape2.parameters.get('radius', 50)) * 0.5
        elif shape1.shape_type == ShapeType.RECTANGLE:
            max_distance = max(
                shape1.parameters.get('width', 50) * 0.3,
                shape1.parameters.get('height', 50) * 0.3,
                shape2.parameters.get('width', 50) * 0.3,
                shape2.parameters.get('height', 50) * 0.3
            )
        else:
            max_distance = 50  # Default threshold

        return center_distance < max_distance

    def _merge_shape_group(self, shapes: List[DetectedShape]) -> DetectedShape:
        """
        Merge a group of similar shapes into one
        """
        # Use the highest confidence shape as base
        best_shape = max(shapes, key=lambda s: s.confidence)

        # Average the centers
        avg_x = sum(s.center.x for s in shapes) / len(shapes)
        avg_y = sum(s.center.y for s in shapes) / len(shapes)

        # Merge parameters (average where appropriate)
        merged_params = best_shape.parameters.copy()
        if best_shape.shape_type == ShapeType.CIRCLE:
            avg_radius = sum(s.parameters.get('radius', 0) for s in shapes) / len(shapes)
            merged_params['radius'] = avg_radius
        elif best_shape.shape_type == ShapeType.RECTANGLE:
            avg_width = sum(s.parameters.get('width', 0) for s in shapes) / len(shapes)
            avg_height = sum(s.parameters.get('height', 0) for s in shapes) / len(shapes)
            merged_params['width'] = avg_width
            merged_params['height'] = avg_height

        # Combine bounding boxes
        all_bboxes = [s.bounding_box for s in shapes]
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)

        # Combine source strokes
        all_strokes = []
        for shape in shapes:
            all_strokes.extend(shape.source_strokes)

        return DetectedShape(
            shape_type=best_shape.shape_type,
            center=Point2D(avg_x, avg_y),
            parameters=merged_params,
            confidence=max(s.confidence for s in shapes),  # Use best confidence
            bounding_box=(min_x, min_y, max_x, max_y),
            source_strokes=all_strokes
        )

    def _detect_connections(self, shapes: List[DetectedShape], strokes: List[Stroke]) -> List[DetectedConnection]:
        """
        Detect connections between shapes (joints, constraints)
        """
        connections = []

        # Look for shapes that are close to each other
        for i, shape1 in enumerate(shapes):
            for j, shape2 in enumerate(shapes[i+1:], i+1):
                connection = self._analyze_shape_connection(shape1, shape2, strokes)
                if connection:
                    connections.append(connection)

        # Look for explicit connection drawings (lines between shapes, springs, etc.)
        line_shapes = [s for s in shapes if s.shape_type == ShapeType.LINE]
        for line in line_shapes:
            connection = self._analyze_line_connection(line, shapes)
            if connection:
                connections.append(connection)

        return connections

    def _analyze_shape_connection(self, shape1: DetectedShape, shape2: DetectedShape, strokes: List[Stroke]) -> Optional[DetectedConnection]:
        """
        Analyze if two shapes are connected
        """
        # Calculate distance between shape centers
        distance = shape1.center.distance_to(shape2.center)

        # Determine connection threshold based on shape sizes
        threshold1 = self._get_shape_connection_threshold(shape1)
        threshold2 = self._get_shape_connection_threshold(shape2)
        max_threshold = max(threshold1, threshold2)

        if distance < max_threshold:
            # Determine connection type based on shape types and relative positions
            connection_type = self._infer_connection_type(shape1, shape2, distance)

            # Calculate connection point (midpoint or edge intersection)
            connection_point = self._calculate_connection_point(shape1, shape2)

            # Calculate confidence based on distance and shape types
            confidence = max(0, 1.0 - (distance / max_threshold))

            return DetectedConnection(
                connection_type=connection_type,
                shape1_id=shape1.source_strokes[0] if shape1.source_strokes else "unknown",
                shape2_id=shape2.source_strokes[0] if shape2.source_strokes else "unknown",
                connection_point=connection_point,
                confidence=confidence
            )

        return None

    def _get_shape_connection_threshold(self, shape: DetectedShape) -> float:
        """
        Get the connection distance threshold for a shape
        """
        if shape.shape_type == ShapeType.CIRCLE:
            return shape.parameters.get('radius', 25) + 20
        elif shape.shape_type == ShapeType.RECTANGLE:
            return max(shape.parameters.get('width', 50), shape.parameters.get('height', 50)) / 2 + 20
        else:
            return 40  # Default threshold

    def _infer_connection_type(self, shape1: DetectedShape, shape2: DetectedShape, distance: float) -> ConnectionType:
        """
        Infer the type of connection between two shapes
        """
        # Heuristics for connection type based on shapes and distance

        # Very close shapes - likely fixed connection
        if distance < 10:
            return ConnectionType.FIXED_JOINT

        # Circle to line/rectangle - likely hinge
        if (shape1.shape_type == ShapeType.CIRCLE and shape2.shape_type in [ShapeType.LINE, ShapeType.RECTANGLE]) or \
           (shape2.shape_type == ShapeType.CIRCLE and shape1.shape_type in [ShapeType.LINE, ShapeType.RECTANGLE]):
            return ConnectionType.HINGE_JOINT

        # Line to line - could be slider or hinge
        if shape1.shape_type == ShapeType.LINE and shape2.shape_type == ShapeType.LINE:
            # Check angles to determine if parallel (slider) or perpendicular (hinge)
            angle1 = shape1.parameters.get('angle', 0)
            angle2 = shape2.parameters.get('angle', 0)
            angle_diff = abs(angle1 - angle2) % 180

            if angle_diff < 20 or angle_diff > 160:  # Parallel lines
                return ConnectionType.SLIDER_JOINT
            else:  # Perpendicular or angled
                return ConnectionType.HINGE_JOINT

        # Default to contact for other cases
        return ConnectionType.CONTACT

    def _calculate_connection_point(self, shape1: DetectedShape, shape2: DetectedShape) -> Point2D:
        """
        Calculate the connection point between two shapes
        """
        # Simple midpoint for now - could be more sophisticated
        mid_x = (shape1.center.x + shape2.center.x) / 2
        mid_y = (shape1.center.y + shape2.center.y) / 2
        return Point2D(mid_x, mid_y)

    def _analyze_line_connection(self, line_shape: DetectedShape, all_shapes: List[DetectedShape]) -> Optional[DetectedConnection]:
        """
        Analyze if a line shape represents a connection between other shapes
        """
        if line_shape.shape_type != ShapeType.LINE:
            return None

        # Get line endpoints
        start_x = line_shape.parameters.get('start_x', line_shape.center.x)
        start_y = line_shape.parameters.get('start_y', line_shape.center.y)
        end_x = line_shape.parameters.get('end_x', line_shape.center.x)
        end_y = line_shape.parameters.get('end_y', line_shape.center.y)

        start_point = Point2D(start_x, start_y)
        end_point = Point2D(end_x, end_y)

        # Find shapes near the line endpoints
        start_shape = self._find_nearest_shape(start_point, all_shapes, exclude=line_shape)
        end_shape = self._find_nearest_shape(end_point, all_shapes, exclude=line_shape)

        if start_shape and end_shape and start_shape != end_shape:
            # Line connects two different shapes
            connection_type = ConnectionType.FIXED_JOINT  # Lines typically represent fixed connections

            return DetectedConnection(
                connection_type=connection_type,
                shape1_id=start_shape.source_strokes[0] if start_shape.source_strokes else "unknown",
                shape2_id=end_shape.source_strokes[0] if end_shape.source_strokes else "unknown",
                connection_point=Point2D(line_shape.center.x, line_shape.center.y),
                confidence=0.8  # High confidence for explicit line connections
            )

        return None

    def _find_nearest_shape(self, point: Point2D, shapes: List[DetectedShape], exclude: Optional[DetectedShape] = None) -> Optional[DetectedShape]:
        """
        Find the nearest shape to a given point
        """
        min_distance = float('inf')
        nearest_shape = None

        for shape in shapes:
            if exclude and shape == exclude:
                continue

            distance = point.distance_to(shape.center)

            # Adjust distance based on shape size
            if shape.shape_type == ShapeType.CIRCLE:
                distance = max(0, distance - shape.parameters.get('radius', 0))
            elif shape.shape_type == ShapeType.RECTANGLE:
                # Approximate distance to rectangle edge
                half_width = shape.parameters.get('width', 50) / 2
                half_height = shape.parameters.get('height', 50) / 2
                distance = max(0, distance - max(half_width, half_height))

            if distance < min_distance and distance < 50:  # Within reasonable range
                min_distance = distance
                nearest_shape = shape

        return nearest_shape

    def _extract_text(self, image: np.ndarray) -> List[DetectedText]:
        """
        Extract text annotations from the image using OCR
        """
        text_annotations = []

        if not self.ocr_reader:
            return text_annotations

        try:
            # Use EasyOCR to detect text
            results = self.ocr_reader.readtext(image)

            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Confidence threshold
                    # Calculate center of bounding box
                    bbox_array = np.array(bbox)
                    center_x = np.mean(bbox_array[:, 0])
                    center_y = np.mean(bbox_array[:, 1])

                    # Calculate bounding box
                    min_x = np.min(bbox_array[:, 0])
                    min_y = np.min(bbox_array[:, 1])
                    max_x = np.max(bbox_array[:, 0])
                    max_y = np.max(bbox_array[:, 1])

                    text_annotation = DetectedText(
                        text=text.strip(),
                        position=Point2D(center_x, center_y),
                        bounding_box=(min_x, min_y, max_x, max_y),
                        confidence=confidence
                    )
                    text_annotations.append(text_annotation)

        except Exception as e:
            logger.warning(f"OCR text extraction failed: {e}")

        return text_annotations

    def _associate_text_with_shapes(self, text_annotations: List[DetectedText], shapes: List[DetectedShape]):
        """
        Associate text annotations with nearby shapes
        """
        for text in text_annotations:
            nearest_shape = self._find_nearest_shape(text.position, shapes)
            if nearest_shape:
                distance = text.position.distance_to(nearest_shape.center)
                threshold = self._get_shape_connection_threshold(nearest_shape)

                if distance < threshold:
                    text.associated_shape_id = nearest_shape.source_strokes[0] if nearest_shape.source_strokes else None

                    # Add text information to shape physics properties
                    if 'annotations' not in nearest_shape.physics_properties:
                        nearest_shape.physics_properties['annotations'] = []
                    nearest_shape.physics_properties['annotations'].append(text.text)

    def _interpret_physics(self, shapes: List[DetectedShape], connections: List[DetectedConnection], text_annotations: List[DetectedText]) -> Dict[str, Any]:
        """
        Interpret the detected shapes and connections as physics objects
        """
        physics_objects = []
        physics_constraints = []

        # Convert shapes to physics objects
        for i, shape in enumerate(shapes):
            physics_obj = self._shape_to_physics_object(shape, i)
            physics_objects.append(physics_obj)

        # Convert connections to physics constraints
        for connection in connections:
            physics_constraint = self._connection_to_physics_constraint(connection)
            physics_constraints.append(physics_constraint)

        # Extract environment parameters from text
        environment_params = self._extract_environment_from_text(text_annotations)

        return {
            'objects': physics_objects,
            'constraints': physics_constraints,
            'environment': environment_params,
            'metadata': {
                'total_shapes': len(shapes),
                'total_connections': len(connections),
                'total_annotations': len(text_annotations)
            }
        }

    def _shape_to_physics_object(self, shape: DetectedShape, object_id: int) -> Dict[str, Any]:
        """
        Convert a detected shape to a physics object description
        """
        # Base object properties
        physics_obj = {
            'id': f'object_{object_id}',
            'name': f'{shape.shape_type.value}_{object_id}',
            'type': 'rigid_body',
            'position': [shape.center.x / 100.0, shape.center.y / 100.0, 0.0],  # Scale to meters
            'material': {
                'density': 1000.0,  # Default density
                'friction': 0.6,
                'restitution': 0.3
            }
        }

        # Shape-specific geometry
        if shape.shape_type == ShapeType.CIRCLE:
            radius = shape.parameters.get('radius', 25) / 100.0  # Scale to meters
            physics_obj['geometry'] = {
                'type': 'sphere',
                'radius': radius
            }
            physics_obj['inertial'] = {
                'mass': 4/3 * np.pi * radius**3 * 1000  # Sphere volume * density
            }

        elif shape.shape_type == ShapeType.RECTANGLE:
            width = shape.parameters.get('width', 50) / 100.0
            height = shape.parameters.get('height', 50) / 100.0
            depth = min(width, height) * 0.1  # Assume some depth

            physics_obj['geometry'] = {
                'type': 'box',
                'size': [width/2, height/2, depth/2]  # Half-sizes for MuJoCo
            }
            physics_obj['inertial'] = {
                'mass': width * height * depth * 1000
            }

        elif shape.shape_type == ShapeType.LINE:
            length = shape.parameters.get('length', 100) / 100.0
            radius = 0.01  # Thin rod

            physics_obj['geometry'] = {
                'type': 'capsule',
                'size': [radius, length/2]
            }
            physics_obj['inertial'] = {
                'mass': np.pi * radius**2 * length * 1000
            }

        # Add text annotations as properties
        if shape.physics_properties.get('annotations'):
            physics_obj['annotations'] = shape.physics_properties['annotations']
            # Try to extract physical properties from text
            self._parse_physics_properties_from_text(physics_obj, shape.physics_properties['annotations'])

        return physics_obj

    def _connection_to_physics_constraint(self, connection: DetectedConnection) -> Dict[str, Any]:
        """
        Convert a detected connection to a physics constraint
        """
        constraint = {
            'type': 'joint',
            'body1': connection.shape1_id,
            'body2': connection.shape2_id,
            'position': [connection.connection_point.x / 100.0, connection.connection_point.y / 100.0, 0.0]
        }

        # Map connection types to joint types
        joint_type_mapping = {
            ConnectionType.HINGE_JOINT: 'hinge',
            ConnectionType.SLIDER_JOINT: 'slide',
            ConnectionType.FIXED_JOINT: 'weld',
            ConnectionType.SPRING: 'spring',
            ConnectionType.CONTACT: 'contact',
            ConnectionType.FREE: 'free'
        }

        constraint['joint_type'] = joint_type_mapping.get(connection.connection_type, 'hinge')
        constraint['parameters'] = connection.parameters

        return constraint

    def _extract_environment_from_text(self, text_annotations: List[DetectedText]) -> Dict[str, Any]:
        """
        Extract environment parameters from text annotations
        """
        environment = {
            'gravity': [0.0, 0.0, -9.81],  # Default gravity
            'ground': {'type': 'plane', 'friction': 0.8},
            'boundaries': {'type': 'none'}
        }

        # Look for physics-related keywords in text
        all_text = ' '.join([text.text.lower() for text in text_annotations])

        # Gravity keywords
        if 'no gravity' in all_text or 'zero gravity' in all_text or 'space' in all_text:
            environment['gravity'] = [0.0, 0.0, 0.0]
        elif 'high gravity' in all_text:
            environment['gravity'] = [0.0, 0.0, -19.62]  # 2x Earth gravity
        elif 'low gravity' in all_text or 'moon' in all_text:
            environment['gravity'] = [0.0, 0.0, -1.62]  # Moon gravity

        # Surface properties
        if 'ice' in all_text or 'slippery' in all_text:
            environment['ground']['friction'] = 0.1
        elif 'rough' in all_text or 'sand' in all_text:
            environment['ground']['friction'] = 1.5

        return environment

    def _parse_physics_properties_from_text(self, physics_obj: Dict[str, Any], annotations: List[str]):
        """
        Parse physics properties from text annotations
        """
        all_text = ' '.join(annotations).lower()

        # Mass keywords
        if 'heavy' in all_text:
            physics_obj['inertial']['mass'] *= 5
        elif 'light' in all_text:
            physics_obj['inertial']['mass'] *= 0.2

        # Material keywords
        if 'metal' in all_text or 'steel' in all_text:
            physics_obj['material']['density'] = 7800  # Steel density
            physics_obj['material']['friction'] = 0.8
        elif 'wood' in all_text:
            physics_obj['material']['density'] = 600
            physics_obj['material']['friction'] = 0.6
        elif 'rubber' in all_text:
            physics_obj['material']['density'] = 1200
            physics_obj['material']['restitution'] = 0.8
        elif 'ice' in all_text:
            physics_obj['material']['friction'] = 0.05
            physics_obj['material']['restitution'] = 0.1

    def _calculate_overall_confidence(self, shapes: List[DetectedShape], connections: List[DetectedConnection], text_annotations: List[DetectedText]) -> float:
        """
        Calculate overall confidence score for the analysis
        """
        if not shapes:
            return 0.0

        # Average shape confidence
        shape_confidence = sum(s.confidence for s in shapes) / len(shapes)

        # Connection confidence (if any connections detected)
        connection_confidence = 0.0
        if connections:
            connection_confidence = sum(c.confidence for c in connections) / len(connections)

        # Text confidence (if any text detected)
        text_confidence = 0.0
        if text_annotations:
            text_confidence = sum(t.confidence for t in text_annotations) / len(text_annotations)

        # Weighted average
        if connections and text_annotations:
            overall_confidence = (shape_confidence * 0.6 + connection_confidence * 0.25 + text_confidence * 0.15)
        elif connections:
            overall_confidence = (shape_confidence * 0.8 + connection_confidence * 0.2)
        elif text_annotations:
            overall_confidence = (shape_confidence * 0.9 + text_confidence * 0.1)
        else:
            overall_confidence = shape_confidence

        return float(overall_confidence)


# Factory function for easy instantiation
def create_cv_pipeline() -> ComputerVisionPipeline:
    """Create a computer vision pipeline instance"""
    return ComputerVisionPipeline()