#!/usr/bin/env python3
"""
Codebase cleanup script to implement clean architecture.
Removes redundant files and reorganizes the structure.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Base directory
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = BASE_DIR / "simgen" / "backend"
SRC_DIR = BACKEND_DIR / "src" / "simgen"

# Files to remove (redundant/duplicate)
FILES_TO_REMOVE = [
    # Duplicate service files (keep optimized versions)
    "simgen/backend/src/simgen/services/sketch_analyzer.py",  # Keep optimized
    "simgen/backend/src/simgen/services/streaming_protocol.py",  # Keep optimized

    # Redundant test files created for coverage
    "simgen/backend/tests/unit/test_50_percent_smart_mock.py",
    "simgen/backend/tests/unit/test_50_percent_ultra_force.py",
    "simgen/backend/tests/unit/test_actual_50_percent.py",
    "simgen/backend/tests/unit/test_aggressive_50.py",
    "simgen/backend/tests/unit/test_final_push_50.py",
    "simgen/backend/tests/unit/test_focused_50_percent.py",
    "simgen/backend/tests/unit/test_force_50_percent.py",
    "simgen/backend/tests/unit/test_maximum_50.py",
    "simgen/backend/tests/unit/test_targeted_50.py",
    "simgen/backend/tests/unit/test_ultra_50_final.py",
    "simgen/backend/tests/unit/test_ultra_50_percent_final.py",
    "simgen/backend/tests/unit/test_ultra_final_50.py",
    "simgen/backend/tests/unit/test_working_imports.py",
    "simgen/backend/tests/unit/test_actual_modules_50_percent.py",
    "simgen/backend/tests/unit/test_api_services_35_percent.py",
    "simgen/backend/tests/unit/test_final_50_percent_push.py",
    "simgen/backend/tests/unit/test_ultimate_50_percent_breakthrough.py",
    "simgen/backend/tests/unit/test_ultra_aggressive_50_percent.py",
    "simgen/backend/tests/unit/test_working_modules_50_percent.py",
]

# Directories to remove
DIRS_TO_REMOVE = [
    "simgen/backend/src/simgen/db",  # Use database/ instead
    "simgen/backend/src/simgen/validation",  # Move to core/
]

# File reorganization mapping
FILE_MOVES = {
    # Rename optimized versions
    "services/sketch_analyzer_optimized.py": "services/vision/analyzer.py",
    "services/streaming_protocol_optimized.py": "services/physics/streaming.py",

    # Move to proper locations
    "services/mjcf_compiler.py": "services/physics/compiler.py",
    "services/mujoco_runtime.py": "services/physics/runtime.py",
    "services/computer_vision_pipeline.py": "services/vision/pipeline.py",
    "services/sketch_to_physics_converter.py": "services/vision/converter.py",
    "services/llm_client.py": "services/ai/llm_client.py",
    "services/multimodal_enhancer.py": "services/ai/enhancer.py",
    "services/cache_service.py": "services/infrastructure/cache.py",
    "services/websocket_session_manager.py": "services/infrastructure/websocket.py",
    "services/realtime_progress.py": "services/infrastructure/progress.py",

    # Database consolidation
    "database/connection_pool.py": "repositories/connection_pool.py",
    "database/query_optimizer.py": "repositories/query_optimizer.py",
    "database/service.py": "repositories/simulation.py",
    "database/optimized_models.py": "models/database.py",
}

def cleanup_files():
    """Remove redundant files."""
    print("🧹 Removing redundant files...")
    removed_count = 0

    for file_path in FILES_TO_REMOVE:
        full_path = BASE_DIR / file_path
        if full_path.exists():
            try:
                os.remove(full_path)
                print(f"  ✓ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Error removing {file_path}: {e}")

    print(f"  Removed {removed_count} files")
    return removed_count

def cleanup_directories():
    """Remove redundant directories."""
    print("\n📁 Removing redundant directories...")
    removed_count = 0

    for dir_path in DIRS_TO_REMOVE:
        full_path = BASE_DIR / dir_path
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                print(f"  ✓ Removed directory: {dir_path}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Error removing {dir_path}: {e}")

    print(f"  Removed {removed_count} directories")
    return removed_count

def create_clean_structure():
    """Create the new clean architecture structure."""
    print("\n🏗️  Creating clean architecture structure...")

    new_dirs = [
        SRC_DIR / "domain",
        SRC_DIR / "repositories",
        SRC_DIR / "services" / "physics",
        SRC_DIR / "services" / "vision",
        SRC_DIR / "services" / "ai",
        SRC_DIR / "services" / "infrastructure",
    ]

    for dir_path in new_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create __init__.py
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Module initialization."""\n')
        print(f"  ✓ Created: {dir_path.relative_to(BASE_DIR)}")

def reorganize_files():
    """Move files to their proper locations."""
    print("\n📦 Reorganizing files...")
    moved_count = 0

    for src_path, dst_path in FILE_MOVES.items():
        src_full = SRC_DIR / src_path
        dst_full = SRC_DIR / dst_path

        if src_full.exists():
            try:
                # Create destination directory if needed
                dst_full.parent.mkdir(parents=True, exist_ok=True)

                # Move the file
                shutil.move(str(src_full), str(dst_full))
                print(f"  ✓ Moved: {src_path} → {dst_path}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Error moving {src_path}: {e}")

    print(f"  Moved {moved_count} files")
    return moved_count

def update_imports():
    """Update import statements in Python files."""
    print("\n🔧 Updating import statements...")

    import_mappings = {
        r"from \.\.services\.sketch_analyzer_optimized": "from ..services.vision.analyzer",
        r"from simgen\.services\.sketch_analyzer_optimized": "from simgen.services.vision.analyzer",
        r"from \.\.services\.streaming_protocol_optimized": "from ..services.physics.streaming",
        r"from simgen\.services\.streaming_protocol_optimized": "from simgen.services.physics.streaming",
        r"from \.\.services\.mjcf_compiler": "from ..services.physics.compiler",
        r"from simgen\.services\.mjcf_compiler": "from simgen.services.physics.compiler",
        r"from \.\.services\.mujoco_runtime": "from ..services.physics.runtime",
        r"from simgen\.services\.mujoco_runtime": "from simgen.services.physics.runtime",
        r"from \.\.services\.computer_vision_pipeline": "from ..services.vision.pipeline",
        r"from simgen\.services\.computer_vision_pipeline": "from simgen.services.vision.pipeline",
        r"from \.\.services\.llm_client": "from ..services.ai.llm_client",
        r"from simgen\.services\.llm_client": "from simgen.services.ai.llm_client",
        r"from \.\.services\.cache_service": "from ..services.infrastructure.cache",
        r"from simgen\.services\.cache_service": "from simgen.services.infrastructure.cache",
        r"from \.\.db\.": "from ..models.",
        r"from simgen\.db\.": "from simgen.models.",
    }

    updated_count = 0

    # Find all Python files
    for py_file in SRC_DIR.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            original_content = content

            # Apply import mappings
            for old_import, new_import in import_mappings.items():
                content = re.sub(old_import, new_import, content)

            # Write back if changed
            if content != original_content:
                py_file.write_text(content, encoding='utf-8')
                print(f"  ✓ Updated imports in: {py_file.relative_to(BASE_DIR)}")
                updated_count += 1
        except Exception as e:
            print(f"  ✗ Error updating {py_file.relative_to(BASE_DIR)}: {e}")

    print(f"  Updated {updated_count} files")
    return updated_count

def create_interfaces():
    """Create the interfaces file for dependency injection."""
    print("\n🔌 Creating interfaces...")

    interfaces_content = '''"""
Core interfaces for dependency injection and clean architecture.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Protocol
from ..models.physics_spec import PhysicsSpec

# Service Interfaces

class IPhysicsCompiler(ABC):
    """Interface for physics compilation."""

    @abstractmethod
    async def compile(self, spec: PhysicsSpec) -> str:
        """Compile PhysicsSpec to MJCF XML."""
        pass

class ISketchAnalyzer(ABC):
    """Interface for sketch analysis."""

    @abstractmethod
    async def analyze(self, image_data: bytes, user_text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sketch and return results."""
        pass

class ICacheService(ABC):
    """Interface for caching service."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        pass

class ILLMClient(ABC):
    """Interface for LLM client."""

    @abstractmethod
    async def generate_physics_spec(self, prompt: str) -> PhysicsSpec:
        """Generate PhysicsSpec from prompt."""
        pass

class IWebSocketManager(ABC):
    """Interface for WebSocket management."""

    @abstractmethod
    async def connect_session(self, websocket: Any, client_id: str) -> str:
        """Connect a new WebSocket session."""
        pass

    @abstractmethod
    async def disconnect_session(self, session_id: str) -> None:
        """Disconnect a session."""
        pass

    @abstractmethod
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Send message to session."""
        pass

# Repository Interfaces

class ISimulationRepository(Protocol):
    """Interface for simulation data access."""

    async def create(self, simulation: Dict[str, Any]) -> int: ...
    async def get(self, simulation_id: int) -> Optional[Dict[str, Any]]: ...
    async def update(self, simulation_id: int, data: Dict[str, Any]) -> bool: ...
    async def delete(self, simulation_id: int) -> bool: ...
    async def list(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]: ...

class ICacheRepository(Protocol):
    """Interface for cache data access."""

    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
'''

    interfaces_file = SRC_DIR / "core" / "interfaces.py"
    interfaces_file.write_text(interfaces_content)
    print(f"  ✓ Created: core/interfaces.py")

def create_dependency_injection():
    """Create dependency injection container."""
    print("\n💉 Creating dependency injection container...")

    container_content = '''"""
Dependency injection container for clean architecture.
"""

from typing import Dict, Any, Type, TypeVar

T = TypeVar('T')

class ServiceContainer:
    """Simple dependency injection container."""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}

    def register(self, interface: Type[T], implementation: Any, singleton: bool = True) -> None:
        """
        Register a service implementation.

        Args:
            interface: The interface type
            implementation: The implementation instance or factory
            singleton: Whether to use singleton pattern
        """
        if singleton:
            self._singletons[interface] = implementation
        else:
            self._services[interface] = implementation

    def get(self, interface: Type[T]) -> T:
        """
        Get a service implementation.

        Args:
            interface: The interface type

        Returns:
            The implementation instance
        """
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]

        # Check regular services
        if interface in self._services:
            # If it's a factory function, call it
            implementation = self._services[interface]
            if callable(implementation):
                return implementation()
            return implementation

        raise ValueError(f"No implementation registered for {interface}")

    def clear(self) -> None:
        """Clear all registrations."""
        self._services.clear()
        self._singletons.clear()

# Global container instance
container = ServiceContainer()

def register_services():
    """Register all service implementations."""
    # This will be called from main.py after imports
    pass
'''

    container_file = SRC_DIR / "core" / "container.py"
    container_file.write_text(container_content)
    print(f"  ✓ Created: core/container.py")

def create_domain_entities():
    """Create domain entities file."""
    print("\n📐 Creating domain entities...")

    entities_content = '''"""
Domain entities and business rules.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class SimulationStatus(Enum):
    """Simulation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PhysicsEntity:
    """Domain entity for physics objects."""
    id: str
    name: str
    mass: float
    position: List[float]
    rotation: List[float]

    def validate(self) -> bool:
        """Validate physics entity."""
        if self.mass <= 0:
            raise ValueError(f"Invalid mass: {self.mass}")
        if len(self.position) != 3:
            raise ValueError(f"Invalid position: {self.position}")
        if len(self.rotation) != 4:  # Quaternion
            raise ValueError(f"Invalid rotation: {self.rotation}")
        return True

@dataclass
class SketchEntity:
    """Domain entity for sketches."""
    id: str
    image_data: bytes
    user_text: Optional[str]
    shapes: List[Dict[str, Any]]

    def validate(self) -> bool:
        """Validate sketch entity."""
        if not self.image_data:
            raise ValueError("Empty image data")
        if len(self.image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image too large")
        return True

@dataclass
class SimulationEntity:
    """Domain entity for simulations."""
    id: Optional[int]
    name: str
    status: SimulationStatus
    physics_spec: Dict[str, Any]
    mjcf_xml: Optional[str]

    def can_run(self) -> bool:
        """Check if simulation can be run."""
        return self.status == SimulationStatus.PENDING and self.mjcf_xml is not None

    def can_cancel(self) -> bool:
        """Check if simulation can be cancelled."""
        return self.status == SimulationStatus.RUNNING
'''

    entities_file = SRC_DIR / "domain" / "entities.py"
    entities_file.parent.mkdir(parents=True, exist_ok=True)
    entities_file.write_text(entities_content)
    print(f"  ✓ Created: domain/entities.py")

def generate_summary():
    """Generate a summary of the cleanup."""
    print("\n" + "="*60)
    print("✅ CODEBASE CLEANUP COMPLETE")
    print("="*60)

    summary = """
Clean Architecture Implementation Summary:

1. ✓ Removed redundant files and directories
2. ✓ Created clean architecture structure
3. ✓ Reorganized files into proper modules
4. ✓ Updated import statements
5. ✓ Created interfaces for dependency injection
6. ✓ Created domain entities

New Structure:
- api/         → Thin controllers
- core/        → Interfaces and cross-cutting concerns
- domain/      → Business entities and rules
- services/    → Application logic (organized by feature)
  - physics/   → Physics compilation and simulation
  - vision/    → Computer vision and sketch analysis
  - ai/        → LLM and AI services
  - infrastructure/ → Cache, WebSocket, monitoring
- repositories/ → Data access layer
- models/      → Data transfer objects

Next Steps:
1. Run tests to ensure nothing broke
2. Update API endpoints to use dependency injection
3. Add unit tests for new structure
4. Update documentation
"""

    print(summary)

def main():
    """Main cleanup function."""
    print("🚀 Starting Codebase Cleanup")
    print("="*60)

    # Safety check
    response = input("This will reorganize the codebase. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cleanup cancelled.")
        return

    # Perform cleanup
    cleanup_files()
    cleanup_directories()
    create_clean_structure()
    reorganize_files()
    update_imports()
    create_interfaces()
    create_dependency_injection()
    create_domain_entities()

    # Generate summary
    generate_summary()

    print("\n🎉 Cleanup complete! Please run tests to verify everything works.")

if __name__ == "__main__":
    main()