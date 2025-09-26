"""
TEST WORKING IMPORTS - Find what actually works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set minimal environment
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["SECRET_KEY"] = "test-key"


def test_find_working_imports():
    """Find which modules can be imported without errors."""
    working = []
    failing = []

    modules_to_test = [
        "simgen.core.config",
        "simgen.models.physics_spec",
        "simgen.models.simulation",
        "simgen.models.schemas",
        "simgen.services.resilience",
        "simgen.services.streaming_protocol",
        "simgen.services.prompt_parser",
        "simgen.services.mjcf_compiler",
        "simgen.monitoring.observability",
        "simgen.api.simulation",
        "simgen.api.physics",
        "simgen.api.monitoring",
    ]

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            working.append(module_name)
            print(f"[SUCCESS] {module_name}")
        except Exception as e:
            failing.append((module_name, str(e)))
            print(f"[FAILED] {module_name}: {e}")

    print(f"\n\nWORKING MODULES ({len(working)}):")
    for m in working:
        print(f"  - {m}")

    print(f"\n\nFAILING MODULES ({len(failing)}):")
    for m, e in failing:
        print(f"  - {m}: {e}")

    assert len(working) > 0, "No modules could be imported!"


if __name__ == "__main__":
    test_find_working_imports()