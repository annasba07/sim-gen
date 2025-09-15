"""
Golden Test Fixtures: Validated PhysicsSpec examples
These serve as reference implementations and test cases
"""

from typing import Dict, Any
from simgen.models.physics_spec import PhysicsSpec

# Collection of golden PhysicsSpec fixtures
GOLDEN_SPECS: Dict[str, Dict[str, Any]] = {
    "pendulum": {
        "description": "Simple pendulum with gravity",
        "spec": {
            "meta": {
                "name": "simple_pendulum",
                "description": "Single pendulum under gravity",
                "gravity": [0, 0, -9.81],
                "timestep": 0.002
            },
            "bodies": [
                {
                    "id": "pendulum_bob",
                    "pos": [0, 0, 1],
                    "joint": {
                        "type": "hinge",
                        "axis": [0, 1, 0],
                        "limited": False,
                        "damping": 0.01
                    },
                    "geoms": [
                        {
                            "type": "capsule",
                            "fromto": [0, 0, 0, 0, 0, -0.5],
                            "size": [0.05],
                            "density": 1000,
                            "material": {
                                "rgba": [0.8, 0.2, 0.2, 1.0]
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
                {"type": "jointpos", "source": "pendulum_bob_joint"},
                {"type": "jointvel", "source": "pendulum_bob_joint"}
            ]
        },
        "expected_properties": {
            "nbody": 2,  # world + pendulum
            "nq": 1,     # 1 hinge joint
            "energy_conserved": True,
            "period_approx": 1.42  # seconds for small angles
        }
    },

    "double_pendulum": {
        "description": "Chaotic double pendulum",
        "spec": {
            "meta": {
                "name": "double_pendulum",
                "description": "Two-link pendulum exhibiting chaos",
                "gravity": [0, 0, -9.81],
                "timestep": 0.001
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
                            "material": {"rgba": [0.2, 0.4, 0.8, 1.0]}
                        }
                    ],
                    "inertial": {"mass": 0.5},
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
                                    "material": {"rgba": [0.8, 0.2, 0.2, 1.0]}
                                }
                            ],
                            "inertial": {"mass": 0.3}
                        }
                    ]
                }
            ]
        },
        "expected_properties": {
            "nbody": 3,
            "nq": 2,
            "chaotic": True,
            "sensitive_to_initial_conditions": True
        }
    },

    "cart_pole": {
        "description": "Classic control benchmark",
        "spec": {
            "meta": {
                "name": "cart_pole",
                "description": "Inverted pendulum on cart",
                "gravity": [0, 0, -9.81]
            },
            "bodies": [
                {
                    "id": "cart",
                    "pos": [0, 0, 0.1],
                    "joint": {
                        "type": "slide",
                        "axis": [1, 0, 0],
                        "limited": True,
                        "range": [-2.4, 2.4]
                    },
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.2, 0.1, 0.05],
                            "material": {"rgba": [0.3, 0.3, 0.7, 1.0]}
                        }
                    ],
                    "inertial": {"mass": 1.0},
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
                                    "material": {"rgba": [0.8, 0.4, 0.2, 1.0]}
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
                    "id": "cart_drive",
                    "type": "motor",
                    "target": "cart_joint",
                    "gear": 100,
                    "ctrlrange": [-1, 1],
                    "forcerange": [-100, 100]
                }
            ],
            "sensors": [
                {"type": "jointpos", "source": "cart_joint"},
                {"type": "jointpos", "source": "pole_joint"},
                {"type": "jointvel", "source": "cart_joint"},
                {"type": "jointvel", "source": "pole_joint"}
            ]
        },
        "expected_properties": {
            "nbody": 3,
            "nq": 2,
            "nu": 1,
            "controllable": True,
            "unstable_equilibrium": True
        }
    },

    "box_stack": {
        "description": "Stack of boxes for collision testing",
        "spec": {
            "meta": {
                "name": "box_stack",
                "description": "Falling and stacking boxes",
                "gravity": [0, 0, -9.81]
            },
            "bodies": [
                {
                    "id": "box1",
                    "pos": [0, 0, 0.5],
                    "joint": {"type": "free"},
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.1, 0.1, 0.1],
                            "material": {"rgba": [0.8, 0.2, 0.2, 1.0]},
                            "friction": {"slide": 0.8, "spin": 0.05, "roll": 0.01}
                        }
                    ],
                    "inertial": {"mass": 0.5}
                },
                {
                    "id": "box2",
                    "pos": [0.05, 0, 1.0],
                    "joint": {"type": "free"},
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.1, 0.1, 0.1],
                            "material": {"rgba": [0.2, 0.8, 0.2, 1.0]},
                            "friction": {"slide": 0.8, "spin": 0.05, "roll": 0.01}
                        }
                    ],
                    "inertial": {"mass": 0.5}
                },
                {
                    "id": "box3",
                    "pos": [-0.03, 0.02, 1.5],
                    "joint": {"type": "free"},
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.1, 0.1, 0.1],
                            "material": {"rgba": [0.2, 0.2, 0.8, 1.0]},
                            "friction": {"slide": 0.8, "spin": 0.05, "roll": 0.01}
                        }
                    ],
                    "inertial": {"mass": 0.5}
                }
            ]
        },
        "expected_properties": {
            "nbody": 4,
            "nq": 21,  # 3 free joints × 7 DOF
            "collision_handling": True,
            "stable_stack": True
        }
    },

    "robot_arm_2dof": {
        "description": "Two-link planar robot arm",
        "spec": {
            "meta": {
                "name": "robot_arm_2dof",
                "description": "Planar robot for reaching tasks",
                "gravity": [0, 0, -9.81]
            },
            "bodies": [
                {
                    "id": "base",
                    "pos": [0, 0, 0.5],
                    "geoms": [
                        {
                            "type": "cylinder",
                            "fromto": [0, 0, -0.05, 0, 0, 0],
                            "size": [0.08],
                            "material": {"rgba": [0.2, 0.2, 0.2, 1.0]}
                        }
                    ],
                    "children": [
                        {
                            "id": "shoulder",
                            "pos": [0, 0, 0],
                            "joint": {
                                "type": "hinge",
                                "axis": [0, 0, 1],
                                "limited": True,
                                "range": [-3.14159, 3.14159],
                                "damping": 0.1,
                                "armature": 0.01
                            },
                            "geoms": [
                                {
                                    "type": "capsule",
                                    "fromto": [0, 0, 0, 0.3, 0, 0],
                                    "size": [0.04],
                                    "material": {"rgba": [0.5, 0.5, 0.5, 1.0]}
                                }
                            ],
                            "inertial": {"mass": 2.0},
                            "children": [
                                {
                                    "id": "elbow",
                                    "pos": [0.3, 0, 0],
                                    "joint": {
                                        "type": "hinge",
                                        "axis": [0, 0, 1],
                                        "limited": True,
                                        "range": [-2.5, 2.5],
                                        "damping": 0.05,
                                        "armature": 0.005
                                    },
                                    "geoms": [
                                        {
                                            "type": "capsule",
                                            "fromto": [0, 0, 0, 0.25, 0, 0],
                                            "size": [0.03],
                                            "material": {"rgba": [0.6, 0.6, 0.6, 1.0]}
                                        }
                                    ],
                                    "inertial": {"mass": 1.0},
                                    "children": [
                                        {
                                            "id": "end_effector",
                                            "pos": [0.25, 0, 0],
                                            "geoms": [
                                                {
                                                    "type": "sphere",
                                                    "size": [0.03],
                                                    "material": {"rgba": [0.9, 0.1, 0.1, 1.0]}
                                                }
                                            ],
                                            "inertial": {"mass": 0.1}
                                        }
                                    ]
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
                    "ctrlrange": [-2, 2],
                    "forcerange": [-50, 50]
                },
                {
                    "id": "elbow_motor",
                    "type": "motor",
                    "target": "elbow_joint",
                    "gear": 30,
                    "ctrlrange": [-2, 2],
                    "forcerange": [-30, 30]
                }
            ],
            "sensors": [
                {"type": "jointpos", "source": "shoulder_joint"},
                {"type": "jointpos", "source": "elbow_joint"},
                {"type": "jointvel", "source": "shoulder_joint"},
                {"type": "jointvel", "source": "elbow_joint"},
                {"type": "actuatorfrc", "source": "shoulder_motor"},
                {"type": "actuatorfrc", "source": "elbow_motor"}
            ]
        },
        "expected_properties": {
            "nbody": 5,
            "nq": 2,
            "nu": 2,
            "workspace_radius": 0.55,
            "singularities": True
        }
    },

    "pulley_system": {
        "description": "Simple pulley with masses",
        "spec": {
            "meta": {
                "name": "pulley",
                "description": "Two masses connected by pulley",
                "gravity": [0, 0, -9.81]
            },
            "bodies": [
                {
                    "id": "mass1",
                    "pos": [-0.3, 0, 0.5],
                    "joint": {
                        "type": "slide",
                        "axis": [0, 0, 1],
                        "limited": True,
                        "range": [-1, 1]
                    },
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.05, 0.05, 0.1],
                            "material": {"rgba": [0.8, 0.2, 0.2, 1.0]}
                        }
                    ],
                    "inertial": {"mass": 2.0}
                },
                {
                    "id": "mass2",
                    "pos": [0.3, 0, 0.5],
                    "joint": {
                        "type": "slide",
                        "axis": [0, 0, 1],
                        "limited": True,
                        "range": [-1, 1]
                    },
                    "geoms": [
                        {
                            "type": "box",
                            "size": [0.05, 0.05, 0.1],
                            "material": {"rgba": [0.2, 0.2, 0.8, 1.0]}
                        }
                    ],
                    "inertial": {"mass": 1.0}
                }
            ],
            "equality": [
                {
                    "type": "joint",
                    "body1": "mass1_joint",
                    "body2": "mass2_joint",
                    "polycoef": [0, -1, 1, 0, 0]  # Pulley constraint
                }
            ]
        },
        "expected_properties": {
            "nbody": 3,
            "nq": 2,
            "constrained": True,
            "acceleration_ratio": 2.0  # Due to mass difference
        }
    }
}

def get_golden_spec(name: str) -> PhysicsSpec:
    """Get a validated golden PhysicsSpec by name"""
    if name not in GOLDEN_SPECS:
        raise ValueError(f"Unknown golden spec: {name}")

    spec_data = GOLDEN_SPECS[name]["spec"]
    return PhysicsSpec(**spec_data)

def get_all_golden_names() -> list[str]:
    """Get list of all golden spec names"""
    return list(GOLDEN_SPECS.keys())

def validate_golden_spec(name: str) -> tuple[bool, list[str]]:
    """
    Validate a golden spec compiles correctly
    Returns (success, errors)
    """
    try:
        spec = get_golden_spec(name)
        from simgen.services.mjcf_compiler import MJCFCompiler
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        # Try loading in MuJoCo
        import mujoco
        model = mujoco.MjModel.from_xml_string(mjcf_xml)

        # Check expected properties
        expected = GOLDEN_SPECS[name].get("expected_properties", {})
        errors = []

        if "nbody" in expected and model.nbody != expected["nbody"]:
            errors.append(f"Expected {expected['nbody']} bodies, got {model.nbody}")

        if "nq" in expected and model.nq != expected["nq"]:
            errors.append(f"Expected {expected['nq']} DOFs, got {model.nq}")

        if "nu" in expected and model.nu != expected["nu"]:
            errors.append(f"Expected {expected['nu']} actuators, got {model.nu}")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [str(e)]