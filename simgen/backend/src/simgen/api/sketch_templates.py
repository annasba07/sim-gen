"""
Sketch Templates API
Provides pre-made physics system templates to solve the "blank canvas" problem.

Addresses core UX issue: Users need inspiration and examples of what
"good" physics sketches look like to succeed with the system.
"""

import json
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["templates"])


@dataclass
class SketchTemplate:
    """Predefined sketch template"""
    id: str
    name: str
    description: str
    category: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    physics_concepts: List[str]
    sketch_data: Dict[str, Any]  # Canvas drawing data
    expected_objects: List[str]
    learning_notes: List[str]
    preview_image: str  # Base64 or URL to preview image


class SketchTemplateManager:
    """Manages physics sketch templates"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, SketchTemplate]:
        """Initialize predefined sketch templates"""
        templates = {}

        # Template 1: Simple Pendulum (Beginner)
        templates["simple_pendulum"] = SketchTemplate(
            id="simple_pendulum",
            name="Simple Pendulum",
            description="A classic physics pendulum with a ball suspended by a string",
            category="oscillation",
            difficulty="beginner",
            physics_concepts=["gravity", "oscillation", "angular motion", "energy conservation"],
            sketch_data=self._create_pendulum_sketch(),
            expected_objects=["circle (ball)", "line (string)", "fixed point (pivot)"],
            learning_notes=[
                "Draw a circle for the bob (hanging mass)",
                "Draw a straight line from the pivot to the bob",
                "Mark the pivot point clearly",
                "The system will add gravity automatically"
            ],
            preview_image="data:image/svg+xml;base64,..." # SVG preview
        )

        # Template 2: Ball and Ramp (Beginner)
        templates["ball_ramp"] = SketchTemplate(
            id="ball_ramp",
            name="Ball Rolling Down Ramp",
            description="A ball rolling down an inclined plane",
            category="motion",
            difficulty="beginner",
            physics_concepts=["gravity", "rolling motion", "friction", "acceleration"],
            sketch_data=self._create_ball_ramp_sketch(),
            expected_objects=["circle (ball)", "line (ramp)", "rectangle (ground)"],
            learning_notes=[
                "Draw a circle for the ball at the top",
                "Draw an angled line for the ramp",
                "Add a horizontal line for the ground",
                "The ball will roll down due to gravity"
            ],
            preview_image="data:image/svg+xml;base64,..."
        )

        # Template 3: Double Pendulum (Intermediate)
        templates["double_pendulum"] = SketchTemplate(
            id="double_pendulum",
            name="Double Pendulum",
            description="Two connected pendulums showing chaotic motion",
            category="oscillation",
            difficulty="intermediate",
            physics_concepts=["chaos theory", "coupled oscillation", "complex dynamics"],
            sketch_data=self._create_double_pendulum_sketch(),
            expected_objects=["circle (mass 1)", "circle (mass 2)", "line (rod 1)", "line (rod 2)", "pivot"],
            learning_notes=[
                "Draw first pendulum (circle + line to pivot)",
                "Draw second pendulum hanging from first mass",
                "Connect them clearly at the joint",
                "Expect chaotic, unpredictable motion"
            ],
            preview_image="data:image/svg+xml;base64,..."
        )

        # Template 4: Catapult (Intermediate)
        templates["catapult"] = SketchTemplate(
            id="catapult",
            name="Simple Catapult",
            description="A lever-based catapult for launching projectiles",
            category="machines",
            difficulty="intermediate",
            physics_concepts=["levers", "torque", "projectile motion", "energy transfer"],
            sketch_data=self._create_catapult_sketch(),
            expected_objects=["rectangle (base)", "line (arm)", "circle (fulcrum)", "circle (projectile)"],
            learning_notes=[
                "Draw a base platform (rectangle)",
                "Add a lever arm (line) with a pivot point",
                "Place a ball at the end of the arm",
                "The system will simulate launching motion"
            ],
            preview_image="data:image/svg+xml;base64,..."
        )

        # Template 5: Spring-Mass System (Intermediate)
        templates["spring_mass"] = SketchTemplate(
            id="spring_mass",
            name="Spring-Mass Oscillator",
            description="A mass attached to a spring showing harmonic motion",
            category="oscillation",
            difficulty="intermediate",
            physics_concepts=["springs", "harmonic motion", "Hooke's law", "resonance"],
            sketch_data=self._create_spring_mass_sketch(),
            expected_objects=["rectangle (mass)", "spring (zigzag line)", "fixed wall"],
            learning_notes=[
                "Draw a rectangle for the mass",
                "Draw a zigzag line for the spring",
                "Connect spring to a fixed wall",
                "Add 'spring' label for recognition"
            ],
            preview_image="data:image/svg+xml;base64,..."
        )

        # Template 6: Newton's Cradle (Advanced)
        templates["newtons_cradle"] = SketchTemplate(
            id="newtons_cradle",
            name="Newton's Cradle",
            description="Multiple pendulums demonstrating momentum conservation",
            category="conservation",
            difficulty="advanced",
            physics_concepts=["momentum conservation", "elastic collision", "energy transfer"],
            sketch_data=self._create_newtons_cradle_sketch(),
            expected_objects=["5 circles (balls)", "5 lines (strings)", "frame (support)"],
            learning_notes=[
                "Draw 5 circles in a row (touching)",
                "Draw strings from each ball to the top",
                "Ensure balls are at the same height",
                "The leftmost ball transfers momentum through the chain"
            ],
            preview_image="data:image/svg+xml;base64,..."
        )

        return templates

    def _create_pendulum_sketch(self) -> Dict[str, Any]:
        """Create sketch data for simple pendulum"""
        return {
            "strokes": [
                {
                    "id": "pivot",
                    "type": "point",
                    "points": [{"x": 400, "y": 100}],
                    "style": {"color": "black", "size": 8}
                },
                {
                    "id": "string",
                    "type": "line",
                    "points": [
                        {"x": 400, "y": 100},
                        {"x": 300, "y": 300}
                    ],
                    "style": {"color": "black", "width": 2}
                },
                {
                    "id": "bob",
                    "type": "circle",
                    "center": {"x": 300, "y": 300},
                    "radius": 20,
                    "style": {"color": "red", "fill": True}
                }
            ],
            "labels": [
                {"text": "pivot", "position": {"x": 410, "y": 100}},
                {"text": "bob", "position": {"x": 320, "y": 330}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def _create_ball_ramp_sketch(self) -> Dict[str, Any]:
        """Create sketch data for ball and ramp"""
        return {
            "strokes": [
                {
                    "id": "ball",
                    "type": "circle",
                    "center": {"x": 150, "y": 200},
                    "radius": 15,
                    "style": {"color": "blue", "fill": True}
                },
                {
                    "id": "ramp",
                    "type": "line",
                    "points": [
                        {"x": 150, "y": 230},
                        {"x": 600, "y": 400}
                    ],
                    "style": {"color": "brown", "width": 5}
                },
                {
                    "id": "ground",
                    "type": "line",
                    "points": [
                        {"x": 600, "y": 400},
                        {"x": 800, "y": 400}
                    ],
                    "style": {"color": "green", "width": 5}
                }
            ],
            "labels": [
                {"text": "ball", "position": {"x": 120, "y": 180}},
                {"text": "ramp", "position": {"x": 350, "y": 300}},
                {"text": "ground", "position": {"x": 700, "y": 420}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def _create_double_pendulum_sketch(self) -> Dict[str, Any]:
        """Create sketch data for double pendulum"""
        return {
            "strokes": [
                {
                    "id": "pivot",
                    "type": "point",
                    "points": [{"x": 400, "y": 100}],
                    "style": {"color": "black", "size": 8}
                },
                {
                    "id": "rod1",
                    "type": "line",
                    "points": [
                        {"x": 400, "y": 100},
                        {"x": 350, "y": 250}
                    ],
                    "style": {"color": "black", "width": 3}
                },
                {
                    "id": "mass1",
                    "type": "circle",
                    "center": {"x": 350, "y": 250},
                    "radius": 15,
                    "style": {"color": "red", "fill": True}
                },
                {
                    "id": "rod2",
                    "type": "line",
                    "points": [
                        {"x": 350, "y": 250},
                        {"x": 450, "y": 400}
                    ],
                    "style": {"color": "black", "width": 3}
                },
                {
                    "id": "mass2",
                    "type": "circle",
                    "center": {"x": 450, "y": 400},
                    "radius": 15,
                    "style": {"color": "blue", "fill": True}
                }
            ],
            "labels": [
                {"text": "pivot", "position": {"x": 410, "y": 100}},
                {"text": "mass 1", "position": {"x": 360, "y": 270}},
                {"text": "mass 2", "position": {"x": 460, "y": 420}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def _create_catapult_sketch(self) -> Dict[str, Any]:
        """Create sketch data for catapult"""
        return {
            "strokes": [
                {
                    "id": "base",
                    "type": "rectangle",
                    "corner1": {"x": 200, "y": 400},
                    "corner2": {"x": 600, "y": 450},
                    "style": {"color": "brown", "fill": True}
                },
                {
                    "id": "fulcrum",
                    "type": "circle",
                    "center": {"x": 300, "y": 400},
                    "radius": 10,
                    "style": {"color": "black", "fill": True}
                },
                {
                    "id": "arm",
                    "type": "line",
                    "points": [
                        {"x": 150, "y": 380},
                        {"x": 500, "y": 300}
                    ],
                    "style": {"color": "brown", "width": 8}
                },
                {
                    "id": "projectile",
                    "type": "circle",
                    "center": {"x": 500, "y": 290},
                    "radius": 8,
                    "style": {"color": "red", "fill": True}
                }
            ],
            "labels": [
                {"text": "base", "position": {"x": 400, "y": 470}},
                {"text": "pivot", "position": {"x": 310, "y": 420}},
                {"text": "arm", "position": {"x": 320, "y": 350}},
                {"text": "projectile", "position": {"x": 510, "y": 290}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def _create_spring_mass_sketch(self) -> Dict[str, Any]:
        """Create sketch data for spring-mass system"""
        return {
            "strokes": [
                {
                    "id": "wall",
                    "type": "line",
                    "points": [
                        {"x": 100, "y": 200},
                        {"x": 100, "y": 400}
                    ],
                    "style": {"color": "gray", "width": 10}
                },
                {
                    "id": "spring",
                    "type": "zigzag",
                    "points": [
                        {"x": 100, "y": 300},
                        {"x": 120, "y": 290},
                        {"x": 140, "y": 310},
                        {"x": 160, "y": 290},
                        {"x": 180, "y": 310},
                        {"x": 200, "y": 290},
                        {"x": 220, "y": 310},
                        {"x": 240, "y": 300}
                    ],
                    "style": {"color": "blue", "width": 3}
                },
                {
                    "id": "mass",
                    "type": "rectangle",
                    "corner1": {"x": 240, "y": 280},
                    "corner2": {"x": 300, "y": 320},
                    "style": {"color": "red", "fill": True}
                }
            ],
            "labels": [
                {"text": "wall", "position": {"x": 70, "y": 300}},
                {"text": "spring", "position": {"x": 170, "y": 270}},
                {"text": "mass", "position": {"x": 270, "y": 340}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def _create_newtons_cradle_sketch(self) -> Dict[str, Any]:
        """Create sketch data for Newton's cradle"""
        balls = []
        strings = []

        # Create 5 hanging balls
        for i in range(5):
            x = 300 + i * 40  # 40px spacing
            y = 350

            # String
            strings.append({
                "id": f"string_{i}",
                "type": "line",
                "points": [
                    {"x": x, "y": 100},
                    {"x": x, "y": y}
                ],
                "style": {"color": "black", "width": 1}
            })

            # Ball
            balls.append({
                "id": f"ball_{i}",
                "type": "circle",
                "center": {"x": x, "y": y},
                "radius": 15,
                "style": {"color": "silver", "fill": True}
            })

        return {
            "strokes": [
                # Support frame
                {
                    "id": "frame_top",
                    "type": "line",
                    "points": [
                        {"x": 250, "y": 100},
                        {"x": 550, "y": 100}
                    ],
                    "style": {"color": "brown", "width": 5}
                }
            ] + strings + balls,
            "labels": [
                {"text": "Newton's Cradle", "position": {"x": 350, "y": 80}},
                {"text": "lift this ball", "position": {"x": 200, "y": 370}}
            ],
            "canvas_size": {"width": 800, "height": 600}
        }

    def get_all_templates(self) -> List[Dict[str, Any]]:
        """Get all available templates"""
        return [asdict(template) for template in self.templates.values()]

    def get_template_by_id(self, template_id: str) -> Dict[str, Any]:
        """Get specific template by ID"""
        if template_id not in self.templates:
            raise ValueError(f"Template '{template_id}' not found")
        return asdict(self.templates[template_id])

    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get templates filtered by category"""
        return [
            asdict(template) for template in self.templates.values()
            if template.category == category
        ]

    def get_templates_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get templates filtered by difficulty"""
        return [
            asdict(template) for template in self.templates.values()
            if template.difficulty == difficulty
        ]


# Global template manager
template_manager = SketchTemplateManager()


@router.get("/", summary="Get all sketch templates")
async def get_all_templates():
    """Get all available physics sketch templates"""
    return {
        "templates": template_manager.get_all_templates(),
        "total_count": len(template_manager.templates),
        "categories": list(set(t.category for t in template_manager.templates.values())),
        "difficulties": ["beginner", "intermediate", "advanced"]
    }


@router.get("/{template_id}", summary="Get specific template")
async def get_template(template_id: str):
    """Get a specific sketch template by ID"""
    try:
        template = template_manager.get_template_by_id(template_id)
        return template
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/category/{category}", summary="Get templates by category")
async def get_templates_by_category(category: str):
    """Get templates filtered by category"""
    templates = template_manager.get_templates_by_category(category)
    if not templates:
        raise HTTPException(status_code=404, detail=f"No templates found for category '{category}'")

    return {
        "category": category,
        "templates": templates,
        "count": len(templates)
    }


@router.get("/difficulty/{difficulty}", summary="Get templates by difficulty")
async def get_templates_by_difficulty(difficulty: str):
    """Get templates filtered by difficulty level"""
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Difficulty must be 'beginner', 'intermediate', or 'advanced'")

    templates = template_manager.get_templates_by_difficulty(difficulty)

    return {
        "difficulty": difficulty,
        "templates": templates,
        "count": len(templates)
    }


@router.get("/concepts/physics", summary="Get all physics concepts covered")
async def get_physics_concepts():
    """Get all physics concepts covered by templates"""
    all_concepts = set()
    for template in template_manager.templates.values():
        all_concepts.update(template.physics_concepts)

    return {
        "physics_concepts": sorted(list(all_concepts)),
        "total_concepts": len(all_concepts),
        "templates_by_concept": {
            concept: [
                t.id for t in template_manager.templates.values()
                if concept in t.physics_concepts
            ]
            for concept in sorted(all_concepts)
        }
    }