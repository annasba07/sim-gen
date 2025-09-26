"""
Physics API: New endpoints using PhysicsSpec pipeline
Replaces direct MJCF generation with structured intermediate representation
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import json
import base64

from ..models.physics_spec import PhysicsSpec, Body, Geom, GeomType, JointType, Actuator, ActuatorType
from ..services.mjcf_compiler import MJCFCompiler
from ..services.mujoco_runtime import MuJoCoRuntime, SimulationStatus
from ..services.streaming_protocol import streaming_manager
from ..services.llm_client import get_llm_client

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response models
class PhysicsSpecRequest(BaseModel):
    """Request to compile PhysicsSpec to MJCF"""
    spec: PhysicsSpec
    validate_spec: bool = Field(default=True, description="Validate spec before compilation")
    return_binary: bool = Field(default=False, description="Return as binary MJCF")

class CompileResponse(BaseModel):
    """Response from compilation"""
    success: bool
    mjcf_xml: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    model_stats: Optional[Dict[str, Any]] = None

class PromptToPhysicsRequest(BaseModel):
    """Request to generate physics from text prompt"""
    prompt: str = Field(description="Natural language description")
    sketch_data: Optional[str] = Field(default=None, description="Base64 encoded sketch image")
    use_multimodal: bool = Field(default=True, description="Use vision model for sketch")
    max_bodies: int = Field(default=10, description="Maximum number of bodies")
    include_actuators: bool = Field(default=True)
    include_sensors: bool = Field(default=True)

class SimulationRequest(BaseModel):
    """Request to run a simulation"""
    mjcf_xml: Optional[str] = None
    physics_spec: Optional[PhysicsSpec] = None
    duration: float = Field(default=10.0, le=30.0)
    render_video: bool = Field(default=False)
    return_frames: bool = Field(default=False)

# Compiler instance
compiler = MJCFCompiler()

@router.post("/compile", response_model=CompileResponse)
async def compile_physics_spec(request: PhysicsSpecRequest):
    """
    Compile PhysicsSpec to MJCF XML
    This is the core transformation: structured spec → valid MuJoCo XML
    """
    try:
        # Compile to MJCF
        mjcf_xml = compiler.compile(request.spec)

        # Optionally validate with MuJoCo parser
        model_stats = None
        if request.validate_spec:
            try:
                # Quick validation by loading in MuJoCo
                from mujoco import MjModel
                model = MjModel.from_xml_string(mjcf_xml)
                model_stats = {
                    "nbody": model.nbody,
                    "nq": model.nq,
                    "nu": model.nu,
                    "ngeom": model.ngeom,
                    "nsensor": model.nsensor
                }
            except Exception as e:
                return CompileResponse(
                    success=False,
                    errors=[f"MuJoCo validation failed: {str(e)}"]
                )

        return CompileResponse(
            success=True,
            mjcf_xml=mjcf_xml,
            model_stats=model_stats
        )

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        return CompileResponse(
            success=False,
            errors=[str(e)]
        )

@router.post("/generate-from-prompt")
async def generate_from_prompt(request: PromptToPhysicsRequest):
    """
    Generate PhysicsSpec from natural language prompt
    Uses LLM with structured output to create valid spec
    """
    try:
        llm_client = get_llm_client()

        # Build the generation prompt
        system_prompt = """You are a physics simulation expert. Generate a valid PhysicsSpec JSON that represents the described physics scenario.

Rules:
1. Use SI units (meters, kg, seconds, Newtons)
2. Prefer capsule geometries for links and joints
3. Keep dimensions realistic (0.01m to 10m scale)
4. Include proper inertial properties
5. Add actuators and sensors as appropriate
6. Return ONLY valid JSON matching the PhysicsSpec schema
7. No commentary or explanation - just the JSON

PhysicsSpec must include:
- meta: simulation metadata with name and gravity
- bodies: list of rigid bodies with geometry and joints
- actuators: motors/forces to control the system
- sensors: measurements from the system"""

        # Enhanced sketch analysis with computer vision pipeline
        physics_spec = None
        user_content = request.prompt

        if request.sketch_data and request.use_multimodal:
            # Use advanced CV pipeline for sketch analysis
            try:
                from ..services.sketch_analyzer import get_sketch_analyzer
                import base64

                sketch_analyzer = get_sketch_analyzer()

                # Decode sketch data
                sketch_data = request.sketch_data
                if sketch_data.startswith('data:image'):
                    sketch_data = sketch_data.split(',')[1]
                image_bytes = base64.b64decode(sketch_data)

                # Analyze sketch with CV pipeline
                sketch_result = await sketch_analyzer.analyze_sketch(
                    image_data=image_bytes,
                    user_text=request.prompt,
                    include_actuators=request.include_actuators,
                    include_sensors=request.include_sensors
                )

                if sketch_result.success and sketch_result.physics_spec:
                    # Use PhysicsSpec from CV pipeline directly
                    physics_spec = sketch_result.physics_spec
                    logger.info(f"Successfully generated PhysicsSpec from sketch with {len(physics_spec.bodies)} bodies")
                else:
                    # Fall back to enhanced prompt for LLM
                    user_content = f"""Sketch analysis result:
{sketch_result.physics_description}

Computer vision detected:
- {len(sketch_result.cv_analysis.shapes) if sketch_result.cv_analysis else 0} shapes
- {len(sketch_result.cv_analysis.connections) if sketch_result.cv_analysis else 0} connections
- {len(sketch_result.cv_analysis.text_annotations) if sketch_result.cv_analysis else 0} text annotations

Text description: {request.prompt}

Generate a PhysicsSpec that captures this physics system based on the sketch analysis and description."""

            except Exception as e:
                logger.warning(f"CV sketch analysis failed, falling back to LLM-only: {e}")
                # Fall back to basic multimodal prompt
                user_content = f"""Sketch provided (analyze the shapes and structure).
Text description: {request.prompt}

Based on the sketch and description, generate a PhysicsSpec that captures the intended physics system."""

        # Generate PhysicsSpec (either from CV pipeline or LLM)
        if physics_spec is None:
            # Generate with LLM structured output
            response = await llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_content,
                response_schema=PhysicsSpec.schema(),
                max_tokens=4000
            )

            # Parse and validate
            spec = PhysicsSpec.parse_raw(response)
        else:
            # Use PhysicsSpec from CV pipeline
            spec = physics_spec

        # Compile to MJCF
        mjcf_xml = compiler.compile(spec)

        return {
            "success": True,
            "physics_spec": spec.dict(),
            "mjcf_xml": mjcf_xml
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_physics_templates():
    """
    Get pre-built PhysicsSpec templates
    These are golden examples for common physics scenarios
    """
    templates = {
        "pendulum": {
            "name": "Simple Pendulum",
            "description": "Single pendulum with hinge joint",
            "spec": {
                "meta": {
                    "name": "pendulum",
                    "gravity": [0, 0, -9.81]
                },
                "bodies": [
                    {
                        "id": "pendulum",
                        "pos": [0, 0, 1],
                        "joint": {
                            "type": "hinge",
                            "axis": [0, 1, 0],
                            "limited": False
                        },
                        "geoms": [
                            {
                                "type": "capsule",
                                "fromto": [0, 0, 0, 0, 0, -0.5],
                                "size": [0.05],
                                "material": {
                                    "rgba": [0.8, 0.2, 0.2, 1]
                                }
                            }
                        ],
                        "inertial": {
                            "mass": 1.0,
                            "pos": [0, 0, -0.25]
                        }
                    }
                ],
                "sensors": [
                    {
                        "type": "jointpos",
                        "source": "pendulum_joint"
                    }
                ]
            }
        },
        "double_pendulum": {
            "name": "Double Pendulum",
            "description": "Chaotic double pendulum system",
            "spec": {
                "meta": {
                    "name": "double_pendulum",
                    "gravity": [0, 0, -9.81]
                },
                "bodies": [
                    {
                        "id": "link1",
                        "pos": [0, 0, 1],
                        "joint": {
                            "type": "hinge",
                            "axis": [0, 1, 0],
                            "limited": False
                        },
                        "geoms": [
                            {
                                "type": "capsule",
                                "fromto": [0, 0, 0, 0, 0, -0.3],
                                "size": [0.04],
                                "material": {
                                    "rgba": [0.2, 0.2, 0.8, 1]
                                }
                            }
                        ],
                        "inertial": {
                            "mass": 0.5
                        },
                        "children": [
                            {
                                "id": "link2",
                                "pos": [0, 0, -0.3],
                                "joint": {
                                    "type": "hinge",
                                    "axis": [0, 1, 0],
                                    "limited": False
                                },
                                "geoms": [
                                    {
                                        "type": "capsule",
                                        "fromto": [0, 0, 0, 0, 0, -0.3],
                                        "size": [0.03],
                                        "material": {
                                            "rgba": [0.8, 0.2, 0.2, 1]
                                        }
                                    }
                                ],
                                "inertial": {
                                    "mass": 0.3
                                }
                            }
                        ]
                    }
                ]
            }
        },
        "cart_pole": {
            "name": "Cart-Pole",
            "description": "Classic control problem",
            "spec": {
                "meta": {
                    "name": "cart_pole",
                    "gravity": [0, 0, -9.81]
                },
                "bodies": [
                    {
                        "id": "cart",
                        "pos": [0, 0, 0.5],
                        "joint": {
                            "type": "slide",
                            "axis": [1, 0, 0],
                            "limited": True,
                            "range": [-2, 2]
                        },
                        "geoms": [
                            {
                                "type": "box",
                                "size": [0.2, 0.1, 0.05],
                                "material": {
                                    "rgba": [0.3, 0.3, 0.7, 1]
                                }
                            }
                        ],
                        "inertial": {
                            "mass": 1.0
                        },
                        "children": [
                            {
                                "id": "pole",
                                "pos": [0, 0, 0.05],
                                "joint": {
                                    "type": "hinge",
                                    "axis": [0, 1, 0],
                                    "limited": False
                                },
                                "geoms": [
                                    {
                                        "type": "capsule",
                                        "fromto": [0, 0, 0, 0, 0, 0.6],
                                        "size": [0.025],
                                        "material": {
                                            "rgba": [0.8, 0.4, 0.2, 1]
                                        }
                                    }
                                ],
                                "inertial": {
                                    "mass": 0.1,
                                    "pos": [0, 0, 0.3]
                                }
                            }
                        ]
                    }
                ],
                "actuators": [
                    {
                        "id": "cart_motor",
                        "type": "motor",
                        "target": "cart_joint",
                        "gear": 100,
                        "ctrlrange": [-1, 1]
                    }
                ],
                "sensors": [
                    {
                        "type": "jointpos",
                        "source": "cart_joint"
                    },
                    {
                        "type": "jointpos",
                        "source": "pole_joint"
                    },
                    {
                        "type": "jointvel",
                        "source": "cart_joint"
                    },
                    {
                        "type": "jointvel",
                        "source": "pole_joint"
                    }
                ]
            }
        },
        "robot_arm": {
            "name": "2-DOF Robot Arm",
            "description": "Simple planar robot arm",
            "spec": {
                "meta": {
                    "name": "robot_arm",
                    "gravity": [0, 0, -9.81]
                },
                "bodies": [
                    {
                        "id": "shoulder",
                        "pos": [0, 0, 0.5],
                        "joint": {
                            "type": "hinge",
                            "axis": [0, 0, 1],
                            "limited": True,
                            "range": [-3.14, 3.14],
                            "damping": 0.1
                        },
                        "geoms": [
                            {
                                "type": "cylinder",
                                "fromto": [0, 0, 0, 0.3, 0, 0],
                                "size": [0.04],
                                "material": {
                                    "rgba": [0.4, 0.4, 0.4, 1]
                                }
                            }
                        ],
                        "inertial": {
                            "mass": 2.0
                        },
                        "children": [
                            {
                                "id": "elbow",
                                "pos": [0.3, 0, 0],
                                "joint": {
                                    "type": "hinge",
                                    "axis": [0, 0, 1],
                                    "limited": True,
                                    "range": [-2.5, 2.5],
                                    "damping": 0.05
                                },
                                "geoms": [
                                    {
                                        "type": "cylinder",
                                        "fromto": [0, 0, 0, 0.25, 0, 0],
                                        "size": [0.03],
                                        "material": {
                                            "rgba": [0.5, 0.5, 0.5, 1]
                                        }
                                    }
                                ],
                                "inertial": {
                                    "mass": 1.0
                                },
                                "children": [
                                    {
                                        "id": "end_effector",
                                        "pos": [0.25, 0, 0],
                                        "geoms": [
                                            {
                                                "type": "sphere",
                                                "size": [0.03],
                                                "material": {
                                                    "rgba": [0.8, 0.2, 0.2, 1]
                                                }
                                            }
                                        ],
                                        "inertial": {
                                            "mass": 0.1
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "actuators": [
                    {
                        "id": "shoulder_motor",
                        "type": "motor",
                        "target": "shoulder_joint",
                        "gear": 50,
                        "ctrlrange": [-2, 2]
                    },
                    {
                        "id": "elbow_motor",
                        "type": "motor",
                        "target": "elbow_joint",
                        "gear": 30,
                        "ctrlrange": [-2, 2]
                    }
                ]
            }
        }
    }

    return templates

@router.post("/simulate")
async def run_simulation(request: SimulationRequest):
    """
    Run a simulation and return results
    Can accept either MJCF XML or PhysicsSpec
    """
    try:
        # Get MJCF XML
        if request.physics_spec:
            mjcf_xml = compiler.compile(request.physics_spec)
        elif request.mjcf_xml:
            mjcf_xml = request.mjcf_xml
        else:
            raise ValueError("Either mjcf_xml or physics_spec must be provided")

        # Create runtime
        runtime = MuJoCoRuntime(headless=True)
        manifest = runtime.load_mjcf(mjcf_xml)

        # Run simulation
        frames = []
        frame_callback = lambda f: frames.append(f.to_dict()) if request.return_frames else None

        await runtime.run_async(
            duration=request.duration,
            callback=frame_callback
        )

        # Generate video if requested
        video_url = None
        if request.render_video:
            # TODO: Implement video rendering
            pass

        return {
            "success": True,
            "manifest": manifest.to_dict(),
            "sim_time": runtime.sim_time,
            "frame_count": runtime.frame_count,
            "frames": frames if request.return_frames else None,
            "video_url": video_url
        }

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/stream")
async def stream_simulation(websocket: WebSocket):
    """
    WebSocket endpoint for real-time physics streaming
    Uses binary protocol for efficient frame transmission
    """
    session_id = str(uuid.uuid4())
    session = None

    try:
        # Connect session
        session = await streaming_manager.connect(session_id, websocket)

        # Handle messages
        await streaming_manager.handle_session(session)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        streaming_manager.disconnect(session_id)

@router.get("/stream/stats/{session_id}")
async def get_stream_stats(session_id: str):
    """Get streaming session statistics"""
    stats = streaming_manager.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats

@router.post("/validate-spec")
async def validate_physics_spec(spec: PhysicsSpec):
    """
    Validate a PhysicsSpec without compiling
    Checks for common issues and provides warnings
    """
    warnings = []
    errors = []

    # Check for unrealistic values
    for body in spec.bodies:
        if body.inertial and body.inertial.mass > 1000:
            warnings.append(f"Body '{body.id}' has very large mass: {body.inertial.mass} kg")
        if body.inertial and body.inertial.mass < 0.001:
            warnings.append(f"Body '{body.id}' has very small mass: {body.inertial.mass} kg")

        for geom in body.geoms:
            if geom.type == GeomType.BOX and geom.size:
                if any(s > 10 for s in geom.size):
                    warnings.append(f"Geometry in '{body.id}' has large dimensions")

    # Check actuator ranges
    for actuator in spec.actuators:
        if actuator.gear > 1000:
            warnings.append(f"Actuator '{actuator.id}' has very high gear ratio: {actuator.gear}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "body_count": len(spec.bodies),
        "actuator_count": len(spec.actuators),
        "sensor_count": len(spec.sensors)
    }