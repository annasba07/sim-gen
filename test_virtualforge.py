"""
Quick test script for VirtualForge mode system
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simgen', 'backend', 'src'))

def test_mode_system():
    """Test the mode registry and configuration"""
    print("=" * 60)
    print("Testing VirtualForge Mode System")
    print("=" * 60)

    from simgen.core.modes import mode_registry, get_all_mode_info

    print("\n1. Testing Mode Registry...")
    modes = mode_registry.get_all_modes()
    print(f"   Found {len(modes)} modes:")
    for mode in modes:
        status = "AVAILABLE" if mode.enabled else "COMING SOON"
        beta = " (BETA)" if mode.beta else ""
        print(f"   - {mode.name} [{status}]{beta}")
        print(f"     Engines: {', '.join(mode.engines)}")
        print(f"     Features: {len(mode.features)} ({', '.join(mode.features[:2])}...)")
        print()

    print("\n2. Testing Mode Info API...")
    mode_info = get_all_mode_info()
    print(f"   Generated {len(mode_info)} mode info objects")
    for info in mode_info:
        print(f"   - {info.name}: {len(info.features)} features, {len(info.engines)} engines")

    print("\n3. Testing Mode Availability...")
    physics_available = mode_registry.is_mode_available('physics')
    games_available = mode_registry.is_mode_available('games')
    vr_available = mode_registry.is_mode_available('vr')

    print(f"   Physics Mode: {'YES' if physics_available else 'NO'}")
    print(f"   Games Mode: {'YES' if games_available else 'NO'}")
    print(f"   VR Mode: {'YES' if vr_available else 'NO'}")

    assert physics_available, "Physics mode should be available"
    assert games_available, "Games mode should be available"
    assert not vr_available, "VR mode should not be available yet"

    print("\n" + "=" * 60)
    print("SUCCESS: All mode system tests passed!")
    print("=" * 60)

    return True

def test_api_imports():
    """Test that API modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing API Imports")
    print("=" * 60)

    print("\n1. Testing Unified Creation API...")
    from simgen.api.unified_creation import router
    print(f"   Router prefix: {router.prefix}")
    print(f"   Routes: {len(router.routes)}")
    for route in router.routes:
        print(f"     - {route.methods} {route.path}")

    print("\n2. Testing Mode System API...")
    from simgen.core.modes import CreationRequest, CreationResponse, ModeInfo
    print(f"   CreationRequest: {CreationRequest}")
    print(f"   CreationResponse: {CreationResponse}")
    print(f"   ModeInfo: {ModeInfo}")

    print("\n" + "=" * 60)
    print("SUCCESS: All API imports passed!")
    print("=" * 60)

    return True

def test_model_imports():
    """Test that models can be imported"""
    print("\n" + "=" * 60)
    print("Testing Model Imports")
    print("=" * 60)

    from simgen.models import PhysicsSpec, CVResult

    print("\n1. Testing PhysicsSpec...")
    print(f"   PhysicsSpec class: {PhysicsSpec}")
    print(f"   Fields: {list(PhysicsSpec.model_fields.keys())}")

    print("\n2. Testing CVResult...")
    print(f"   CVResult class: {CVResult}")
    cv_result = CVResult(objects=[], text_annotations=[], confidence=0.8)
    print(f"   Created instance: {cv_result}")

    print("\n" + "=" * 60)
    print("SUCCESS: All model imports passed!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    try:
        # Run all tests
        test_model_imports()
        test_mode_system()
        test_api_imports()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("VirtualForge Mode System is Working!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
