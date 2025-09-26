# 🏗️ Clean Architecture & Code Organization

## Overview

This document defines the clean architecture principles and organization for the SimGen AI codebase, ensuring clear separation of concerns, maintainability, and scalability.

## 📁 Directory Structure

```
simgen/backend/src/simgen/
│
├── api/                    # API Layer (Controllers)
│   ├── __init__.py
│   ├── routes.py          # Route registration
│   ├── health.py          # Health check endpoints
│   ├── physics.py         # Physics endpoints (thin controllers)
│   ├── simulation.py      # Simulation endpoints (thin controllers)
│   └── sketch.py          # Sketch endpoints (thin controllers)
│
├── core/                   # Core Business Logic & Domain
│   ├── __init__.py
│   ├── config.py          # Centralized configuration
│   ├── exceptions.py      # Custom exceptions
│   ├── interfaces.py      # Abstract base classes / protocols
│   ├── validation.py      # Input validation
│   ├── circuit_breaker.py # Cross-cutting concerns
│   └── resource_manager.py # Resource lifecycle management
│
├── domain/                 # Domain Models & Business Rules
│   ├── __init__.py
│   ├── physics.py         # Physics domain logic
│   ├── simulation.py      # Simulation domain logic
│   ├── sketch.py          # Sketch domain logic
│   └── entities.py        # Core business entities
│
├── services/              # Application Services
│   ├── __init__.py
│   ├── physics/           # Physics Service Module
│   │   ├── __init__.py
│   │   ├── compiler.py    # MJCF compilation
│   │   ├── runtime.py     # MuJoCo runtime
│   │   └── streaming.py   # Binary streaming
│   │
│   ├── vision/            # Computer Vision Service Module
│   │   ├── __init__.py
│   │   ├── pipeline.py    # CV pipeline
│   │   ├── analyzer.py    # Sketch analysis
│   │   └── converter.py   # Sketch to physics conversion
│   │
│   ├── ai/                # AI Service Module
│   │   ├── __init__.py
│   │   ├── llm_client.py  # LLM integration
│   │   └── enhancer.py    # Multi-modal enhancement
│   │
│   └── infrastructure/    # Infrastructure Services
│       ├── __init__.py
│       ├── cache.py       # Caching service
│       ├── websocket.py   # WebSocket management
│       └── monitoring.py  # Metrics and monitoring
│
├── repositories/          # Data Access Layer
│   ├── __init__.py
│   ├── base.py           # Base repository pattern
│   ├── simulation.py      # Simulation repository
│   └── cache.py          # Cache repository
│
├── models/                # Data Models (DTOs)
│   ├── __init__.py
│   ├── physics_spec.py   # PhysicsSpec model
│   ├── schemas.py        # Pydantic schemas
│   └── database.py       # SQLAlchemy models
│
└── main.py               # Application entry point
```

## 🎯 Architecture Principles

### 1. **Separation of Concerns**

Each layer has a specific responsibility:

- **API Layer**: HTTP/WebSocket handling, no business logic
- **Domain Layer**: Business rules and entities
- **Service Layer**: Application orchestration
- **Repository Layer**: Data access abstractions
- **Models Layer**: Data transfer objects

### 2. **Dependency Inversion**

Higher-level modules don't depend on lower-level modules:

```python
# core/interfaces.py
from abc import ABC, abstractmethod

class IPhysicsCompiler(ABC):
    @abstractmethod
    async def compile(self, spec: PhysicsSpec) -> str:
        pass

class ISketchAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, image_data: bytes) -> AnalysisResult:
        pass

class ICacheService(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        pass
```

### 3. **Dependency Injection**

Services are injected, not imported directly:

```python
# services/__init__.py
from typing import Protocol

class ServiceContainer:
    """Service container for dependency injection."""

    def __init__(self):
        self._services = {}

    def register(self, interface: type, implementation: Any) -> None:
        self._services[interface] = implementation

    def get(self, interface: type) -> Any:
        return self._services.get(interface)

# Global container
container = ServiceContainer()

# Registration (in main.py)
container.register(IPhysicsCompiler, MJCFCompiler())
container.register(ISketchAnalyzer, OptimizedSketchAnalyzer())
container.register(ICacheService, CacheService())
```

### 4. **Clean API Endpoints**

API endpoints are thin controllers:

```python
# api/physics.py
from fastapi import APIRouter, Depends
from ..core.interfaces import IPhysicsCompiler
from ..services import container

router = APIRouter()

@router.post("/compile")
async def compile_physics(
    request: PhysicsSpecRequest,
    compiler: IPhysicsCompiler = Depends(lambda: container.get(IPhysicsCompiler))
):
    """Thin controller - delegates to service."""
    try:
        result = await compiler.compile(request.spec)
        return {"success": True, "mjcf": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 5. **Service Orchestration**

Services handle business logic orchestration:

```python
# services/vision/analyzer.py
class SketchAnalysisService:
    """Orchestrates sketch analysis workflow."""

    def __init__(
        self,
        cv_pipeline: IComputerVisionPipeline,
        llm_client: ILLMClient,
        cache: ICacheService,
        circuit_breaker: CircuitBreaker
    ):
        self.cv_pipeline = cv_pipeline
        self.llm_client = llm_client
        self.cache = cache
        self.circuit_breaker = circuit_breaker

    async def analyze_sketch(
        self,
        image_data: bytes,
        use_cache: bool = True
    ) -> AnalysisResult:
        """Orchestrates sketch analysis with caching and fallback."""
        # Check cache
        if use_cache:
            cached = await self.cache.get(self._get_cache_key(image_data))
            if cached:
                return cached

        # Try CV pipeline with circuit breaker
        try:
            result = await self.circuit_breaker.call(
                self.cv_pipeline.analyze,
                image_data
            )
        except Exception:
            # Fallback to LLM
            result = await self._llm_fallback(image_data)

        # Cache result
        if use_cache:
            await self.cache.set(
                self._get_cache_key(image_data),
                result,
                ttl=3600
            )

        return result
```

## 🧱 Layer Responsibilities

### API Layer (`api/`)
- **Responsibilities**:
  - HTTP request/response handling
  - Input validation (delegates to core)
  - Error formatting
  - Authentication/authorization
- **NOT Responsible For**:
  - Business logic
  - Data access
  - Service orchestration

### Core Layer (`core/`)
- **Responsibilities**:
  - Cross-cutting concerns
  - Shared interfaces
  - Configuration management
  - Common utilities
- **NOT Responsible For**:
  - Implementation details
  - External service calls

### Domain Layer (`domain/`)
- **Responsibilities**:
  - Business rules
  - Domain entities
  - Domain events
  - Business validations
- **NOT Responsible For**:
  - Infrastructure concerns
  - Data persistence

### Service Layer (`services/`)
- **Responsibilities**:
  - Use case implementation
  - Service orchestration
  - Transaction management
  - External service integration
- **NOT Responsible For**:
  - HTTP concerns
  - Direct database access

### Repository Layer (`repositories/`)
- **Responsibilities**:
  - Data access abstraction
  - Query building
  - Data mapping
  - Cache integration
- **NOT Responsible For**:
  - Business logic
  - Service orchestration

## 🔄 Data Flow

```
Request → API → Service → Domain → Repository → Database
         ↓       ↓         ↓         ↓
      Validation  Logic  Rules    Query
         ↓       ↓         ↓         ↓
Response ← API ← Service ← Domain ← Repository ← Database
```

## 📦 Module Boundaries

### Physics Module
```
services/physics/
├── compiler.py      # PhysicsSpec → MJCF
├── runtime.py       # MuJoCo simulation
└── streaming.py     # WebSocket streaming
```
- **Dependencies**: Core interfaces only
- **Exposed Interface**: IPhysicsService

### Vision Module
```
services/vision/
├── pipeline.py      # CV algorithms
├── analyzer.py      # Sketch analysis
└── converter.py     # Sketch → PhysicsSpec
```
- **Dependencies**: Core interfaces only
- **Exposed Interface**: IVisionService

### AI Module
```
services/ai/
├── llm_client.py    # LLM integration
└── enhancer.py      # Multi-modal AI
```
- **Dependencies**: Core interfaces only
- **Exposed Interface**: IAIService

## 🔌 Interface Definitions

```python
# core/interfaces.py

# Service Interfaces
class IPhysicsService(Protocol):
    async def compile_mjcf(self, spec: PhysicsSpec) -> str: ...
    async def run_simulation(self, mjcf: str) -> SimulationResult: ...

class IVisionService(Protocol):
    async def analyze_sketch(self, image: bytes) -> SketchAnalysis: ...
    async def extract_shapes(self, image: bytes) -> List[Shape]: ...

class IAIService(Protocol):
    async def enhance_prompt(self, prompt: str) -> str: ...
    async def generate_physics_spec(self, prompt: str) -> PhysicsSpec: ...

# Repository Interfaces
class ISimulationRepository(Protocol):
    async def save(self, simulation: Simulation) -> int: ...
    async def get(self, id: int) -> Optional[Simulation]: ...
    async def list(self, filters: Dict) -> List[Simulation]: ...

class ICacheRepository(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
```

## 🧪 Testing Strategy

### Unit Tests
- Test each layer in isolation
- Mock dependencies using interfaces
- Focus on business logic

### Integration Tests
- Test service orchestration
- Use in-memory implementations
- Test actual workflows

### End-to-End Tests
- Test complete user journeys
- Use test containers for dependencies
- Validate API contracts

## 🚀 Migration Plan

### Phase 1: Structure (Immediate)
1. Create new directory structure
2. Move files to appropriate locations
3. Update imports

### Phase 2: Interfaces (Week 1)
1. Define all interfaces
2. Implement dependency injection
3. Update services to use interfaces

### Phase 3: Cleanup (Week 2)
1. Remove circular dependencies
2. Extract business logic from APIs
3. Consolidate duplicate code

### Phase 4: Testing (Week 3)
1. Add unit tests for each layer
2. Add integration tests
3. Ensure 70% coverage

## 📋 Cleanup Checklist

### Files to Remove
- [x] Duplicate service files (*_optimized.py → keep optimized version)
- [x] Redundant test files (test_*_percent_*.py)
- [x] Old database directory (db/ → use database/)
- [x] Unused middleware files

### Code to Refactor
- [x] Extract business logic from API endpoints
- [x] Remove direct service dependencies
- [x] Consolidate configuration
- [x] Standardize error handling

### Patterns to Implement
- [x] Repository pattern for data access
- [x] Service interfaces
- [x] Dependency injection
- [x] Factory pattern for object creation

## 🎯 Benefits

1. **Testability**: Each layer can be tested in isolation
2. **Maintainability**: Clear boundaries make changes easier
3. **Scalability**: Services can be extracted to microservices
4. **Flexibility**: Easy to swap implementations
5. **Clarity**: Code organization matches mental model

## 📚 References

- **Clean Architecture** by Robert C. Martin
- **Domain-Driven Design** by Eric Evans
- **Dependency Injection Principles, Practices, and Patterns** by Mark Seemann

---

This clean architecture ensures the codebase remains maintainable, testable, and scalable without over-engineering.