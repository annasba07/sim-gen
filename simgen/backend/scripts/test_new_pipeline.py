#!/usr/bin/env python3
"""
Test script for the new PhysicsSpec pipeline
Demonstrates the complete flow from prompt to simulation
"""

import asyncio
import json
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simgen.models.physics_spec import PhysicsSpec, Body, Geom, GeomType, SimulationMeta
from simgen.services.mjcf_compiler import MJCFCompiler
from simgen.services.mujoco_runtime import MuJoCoRuntime

# Import test fixtures directly
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from fixtures.golden_specs import get_golden_spec, get_all_golden_names

async def test_basic_compilation():
    """Test basic PhysicsSpec → MJCF compilation"""
    print("\n=== Testing Basic Compilation ===")

    # Create a simple spec
    spec = PhysicsSpec(
        meta=SimulationMeta(
            name="test_box",
            description="Simple falling box"
        ),
        bodies=[
            Body(
                id="box",
                pos=[0, 0, 1],
                joint={"type": "free"},
                geoms=[
                    Geom(
                        type=GeomType.BOX,
                        size=[0.1, 0.1, 0.1],
                        material={"rgba": [1, 0, 0, 1]}
                    )
                ],
                inertial={"mass": 1.0}
            )
        ]
    )

    # Compile to MJCF
    compiler = MJCFCompiler()
    mjcf_xml = compiler.compile(spec)

    print(f"[OK] Generated MJCF ({len(mjcf_xml)} chars)")
    print(f"  First 200 chars: {mjcf_xml[:200]}...")

    return mjcf_xml

async def test_golden_specs():
    """Test all golden specifications"""
    print("\n=== Testing Golden Specs ===")

    compiler = MJCFCompiler()
    results = []

    for spec_name in get_all_golden_names():
        try:
            spec = get_golden_spec(spec_name)
            mjcf_xml = compiler.compile(spec)
            results.append((spec_name, True, len(mjcf_xml)))
            print(f"[OK] {spec_name}: SUCCESS ({len(mjcf_xml)} chars)")
        except Exception as e:
            results.append((spec_name, False, str(e)))
            print(f"[FAIL] {spec_name}: FAILED - {e}")

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    print(f"\nSummary: {successful}/{len(results)} specs compiled successfully")

    return results

async def test_simulation():
    """Test running a simulation"""
    print("\n=== Testing Simulation Runtime ===")

    try:
        # Get pendulum spec
        spec = get_golden_spec("pendulum")
        compiler = MJCFCompiler()
        mjcf_xml = compiler.compile(spec)

        # Create runtime
        runtime = MuJoCoRuntime(headless=True)
        manifest = runtime.load_mjcf(mjcf_xml)

        print(f"[OK] Loaded model: {manifest.model_name}")
        print(f"  Bodies: {manifest.nbody}")
        print(f"  DOFs: {manifest.nq}")
        print(f"  Timestep: {manifest.timestep}s")

        # Run for 1 second
        print("\nRunning simulation...")
        frames_received = []

        def frame_callback(frame):
            frames_received.append(frame)
            if len(frames_received) % 30 == 0:  # Every 30 frames
                print(f"  Frame {frame.frame_id}: t={frame.sim_time:.2f}s")

        await runtime.run_async(duration=1.0, callback=frame_callback)

        print(f"[OK] Simulation complete: {len(frames_received)} frames")
        print(f"  Final time: {runtime.sim_time:.2f}s")
        print(f"  Real-time factor: {runtime.real_time_factor:.1f}x")

        # Test binary serialization
        if frames_received:
            frame = frames_received[0]
            binary_data = frame.to_binary()
            print(f"[OK] Binary frame size: {len(binary_data)} bytes")

    except ImportError:
        print("[FAIL] MuJoCo not installed - skipping simulation test")
    except Exception as e:
        print(f"[FAIL] Simulation failed: {e}")

async def test_api_endpoints():
    """Test the new API endpoints"""
    print("\n=== Testing API Endpoints ===")

    try:
        # Import FastAPI test client
        from fastapi.testclient import TestClient
        from simgen.main import app

        client = TestClient(app)

        # Test compile endpoint
        spec_data = {
            "spec": {
                "meta": {"name": "api_test"},
                "bodies": [{"id": "test_body"}]
            }
        }

        response = client.post("/api/v2/physics/compile", json=spec_data)
        if response.status_code == 200:
            print("[OK] POST /api/v2/physics/compile")
            result = response.json()
            if result.get("success"):
                print(f"  MJCF length: {len(result.get('mjcf_xml', ''))}")
        else:
            print(f"[FAIL] Compile endpoint failed: {response.status_code}")

        # Test templates endpoint
        response = client.get("/api/v2/physics/templates")
        if response.status_code == 200:
            templates = response.json()
            print(f"[OK] GET /api/v2/physics/templates")
            print(f"  Available templates: {', '.join(templates.keys())}")
        else:
            print(f"[FAIL] Templates endpoint failed: {response.status_code}")

    except Exception as e:
        print(f"[FAIL] API test failed: {e}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing New PhysicsSpec Pipeline")
    print("=" * 60)

    # Run tests
    await test_basic_compilation()
    await test_golden_specs()
    await test_simulation()
    await test_api_endpoints()

    print("\n" + "=" * 60)
    print("Pipeline Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())