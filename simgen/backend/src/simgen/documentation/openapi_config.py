"""
OpenAPI Documentation Configuration
Provides comprehensive API documentation with examples, schemas, and security
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html


class OpenAPIConfig:
    """Configuration for OpenAPI documentation."""
    
    @staticmethod
    def get_custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
        """Generate comprehensive OpenAPI schema with enhanced documentation."""
        
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="SimGen AI - Physics Simulation Generator",
            version="1.0.0",
            summary="AI-Powered Physics Simulation Generation Platform",
            description="""
## SimGen AI API Documentation

SimGen AI is a cutting-edge platform that generates physics simulations from natural language descriptions and sketches using advanced AI models.

### Key Features

- **Multi-Modal Input**: Support for text prompts and hand-drawn sketches
- **AI-Powered Generation**: Uses state-of-the-art LLMs for simulation creation
- **Professional Rendering**: High-quality MuJoCo-based physics visualization
- **Real-Time Processing**: WebSocket support for live generation updates
- **Performance Optimized**: Advanced caching and query optimization
- **Production Ready**: Comprehensive monitoring, security, and error handling

### Authentication

Most endpoints require API key authentication. Include your API key in the `X-API-Key` header:

```
X-API-Key: your-api-key-here
```

### Rate Limits

- **Public endpoints**: 100 requests/minute
- **Authenticated endpoints**: 1000 requests/minute  
- **Generation endpoints**: 10 requests/minute
- **Admin endpoints**: No limit

### Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "error_type",
  "message": "Human-readable description",
  "timestamp": "2024-01-01T00:00:00Z",
  "details": {...}
}
```

### WebSocket Support

Real-time simulation generation updates are available via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation/{session_id}');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Generation progress:', update);
};
```

### Security

- All data is validated and sanitized
- SQL injection and XSS protection
- Rate limiting and DDoS protection
- Comprehensive audit logging
- HTTPS required in production
            """,
            routes=app.routes,
        )
        
        # Enhanced security scheme definitions
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication. Contact admin to obtain one."
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for user authentication"
            }
        }
        
        # Global security requirement
        openapi_schema["security"] = [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ]
        
        # Enhanced server information
        openapi_schema["servers"] = [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.simgen.ai",
                "description": "Production server"
            }
        ]
        
        # Add comprehensive tags with descriptions
        openapi_schema["tags"] = [
            {
                "name": "Health",
                "description": "System health and monitoring endpoints"
            },
            {
                "name": "Simulation",
                "description": "Core simulation generation and management"
            },
            {
                "name": "Sketch Analysis",
                "description": "Multi-modal sketch to simulation conversion"
            },
            {
                "name": "Authentication",
                "description": "User authentication and authorization"
            },
            {
                "name": "Monitoring",
                "description": "System metrics and performance monitoring"
            },
            {
                "name": "Database",
                "description": "Database health and optimization"
            },
            {
                "name": "Templates",
                "description": "Simulation templates and patterns"
            },
            {
                "name": "WebSocket",
                "description": "Real-time communication endpoints"
            }
        ]
        
        # Add response examples and enhanced schemas
        OpenAPIConfig._enhance_schemas(openapi_schema)
        OpenAPIConfig._add_response_examples(openapi_schema)
        
        # Add contact and license information
        openapi_schema["info"]["contact"] = {
            "name": "SimGen AI Support",
            "email": "support@simgen.ai",
            "url": "https://simgen.ai/support"
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        # Add external documentation links
        openapi_schema["externalDocs"] = {
            "description": "Full Documentation and Tutorials",
            "url": "https://docs.simgen.ai"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    @staticmethod
    def _enhance_schemas(openapi_schema: Dict[str, Any]) -> None:
        """Add enhanced schema definitions with examples."""
        
        # Ensure components exist
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}
        
        schemas = openapi_schema["components"]["schemas"]
        
        # Simulation request schema
        schemas["SimulationRequest"] = {
            "type": "object",
            "required": ["prompt", "session_id"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Natural language description of the desired simulation",
                    "example": "A red ball bouncing on a trampoline with realistic physics",
                    "minLength": 10,
                    "maxLength": 2000
                },
                "session_id": {
                    "type": "string",
                    "description": "Unique identifier for the simulation session",
                    "example": "session_123456789",
                    "pattern": "^session_[a-zA-Z0-9_-]+$"
                },
                "style_preferences": {
                    "type": "object",
                    "description": "Visual and physics style preferences",
                    "properties": {
                        "render_quality": {
                            "type": "string",
                            "enum": ["draft", "standard", "high", "cinematic"],
                            "default": "standard",
                            "description": "Rendering quality level"
                        },
                        "physics_accuracy": {
                            "type": "string", 
                            "enum": ["basic", "realistic", "precise"],
                            "default": "realistic",
                            "description": "Physics simulation accuracy"
                        },
                        "lighting": {
                            "type": "string",
                            "enum": ["ambient", "directional", "dramatic", "studio"],
                            "default": "directional",
                            "description": "Lighting style for rendering"
                        }
                    }
                },
                "generation_options": {
                    "type": "object",
                    "description": "Options controlling simulation generation",
                    "properties": {
                        "enable_optimization": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable performance optimizations"
                        },
                        "max_objects": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20,
                            "description": "Maximum number of objects in simulation"
                        },
                        "simulation_duration": {
                            "type": "number",
                            "minimum": 1.0,
                            "maximum": 60.0,
                            "default": 10.0,
                            "description": "Simulation duration in seconds"
                        }
                    }
                }
            },
            "example": {
                "prompt": "A pendulum swinging with a heavy metal ball",
                "session_id": "session_demo_001", 
                "style_preferences": {
                    "render_quality": "high",
                    "physics_accuracy": "precise",
                    "lighting": "studio"
                },
                "generation_options": {
                    "enable_optimization": True,
                    "max_objects": 5,
                    "simulation_duration": 15.0
                }
            }
        }
        
        # Simulation response schema
        schemas["SimulationResponse"] = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "Unique simulation identifier",
                    "example": 12345
                },
                "session_id": {
                    "type": "string", 
                    "description": "Session identifier",
                    "example": "session_demo_001"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "processing", "completed", "failed"],
                    "description": "Current simulation status"
                },
                "mjcf_content": {
                    "type": "string",
                    "description": "Generated MuJoCo XML content",
                    "example": "<mujoco>...</mujoco>"
                },
                "execution_data": {
                    "type": "object",
                    "description": "Simulation execution results and media",
                    "properties": {
                        "screenshots": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Base64 encoded screenshots"
                        },
                        "video_path": {
                            "type": "string",
                            "description": "Path to generated video file"
                        },
                        "physics_stats": {
                            "type": "object",
                            "description": "Physics simulation statistics"
                        }
                    }
                },
                "quality_scores": {
                    "type": "object",
                    "description": "Quality assessment scores",
                    "properties": {
                        "overall": {"type": "number", "minimum": 0, "maximum": 10},
                        "physics": {"type": "number", "minimum": 0, "maximum": 10},
                        "visual": {"type": "number", "minimum": 0, "maximum": 10},
                        "functional": {"type": "number", "minimum": 0, "maximum": 10}
                    }
                },
                "generation_duration": {
                    "type": "number",
                    "description": "Time taken to generate simulation (seconds)",
                    "example": 8.5
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Simulation creation timestamp"
                },
                "updated_at": {
                    "type": "string", 
                    "format": "date-time",
                    "description": "Last update timestamp"
                }
            }
        }
        
        # Sketch analysis request
        schemas["SketchAnalysisRequest"] = {
            "type": "object",
            "required": ["image_data"],
            "properties": {
                "image_data": {
                    "type": "string",
                    "format": "byte",
                    "description": "Base64 encoded sketch image (PNG/JPEG)",
                    "example": "iVBORw0KGgoAAAANSUhEUgA..."
                },
                "user_text": {
                    "type": "string",
                    "description": "Optional text description to accompany sketch",
                    "maxLength": 1000,
                    "example": "A ball rolling down a ramp"
                },
                "analysis_mode": {
                    "type": "string",
                    "enum": ["quick", "detailed", "physics_focused"],
                    "default": "detailed",
                    "description": "Depth of sketch analysis to perform"
                }
            }
        }
        
        # Error response schema
        schemas["ErrorResponse"] = {
            "type": "object",
            "required": ["error", "message", "timestamp"],
            "properties": {
                "error": {
                    "type": "string",
                    "description": "Error type identifier",
                    "example": "validation_failed"
                },
                "message": {
                    "type": "string", 
                    "description": "Human-readable error description",
                    "example": "The provided prompt is too short"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Error occurrence timestamp"
                },
                "details": {
                    "type": "object",
                    "description": "Additional error context and details",
                    "additionalProperties": True
                },
                "request_id": {
                    "type": "string",
                    "description": "Unique request identifier for debugging",
                    "example": "req_abc123def456"
                }
            }
        }
        
        # Health check response
        schemas["HealthResponse"] = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["healthy", "degraded", "unhealthy"],
                    "description": "Overall system health status"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Health check timestamp"
                },
                "service": {
                    "type": "string",
                    "description": "Service identifier",
                    "example": "simgen-ai"
                },
                "version": {
                    "type": "string",
                    "description": "Service version",
                    "example": "1.0.0"
                },
                "uptime_seconds": {
                    "type": "number",
                    "description": "Service uptime in seconds"
                },
                "health_checks": {
                    "type": "object",
                    "description": "Individual component health status",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "response_time_ms": {"type": "number"},
                            "details": {"type": "object"}
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def _add_response_examples(openapi_schema: Dict[str, Any]) -> None:
        """Add detailed response examples to endpoints."""
        
        paths = openapi_schema.get("paths", {})
        
        # Add examples for common response codes
        common_responses = {
            "400": {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "validation_failed",
                            "message": "The prompt must be between 10 and 2000 characters",
                            "timestamp": "2024-01-01T10:30:00Z",
                            "details": {
                                "field": "prompt",
                                "provided_length": 5,
                                "min_length": 10
                            },
                            "request_id": "req_abc123def456"
                        }
                    }
                }
            },
            "401": {
                "description": "Unauthorized - Invalid or missing API key",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "unauthorized",
                            "message": "Valid API key required",
                            "timestamp": "2024-01-01T10:30:00Z",
                            "request_id": "req_abc123def456"
                        }
                    }
                }
            },
            "429": {
                "description": "Too Many Requests - Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "rate_limit_exceeded",
                            "message": "Rate limit exceeded. Try again in 60 seconds",
                            "timestamp": "2024-01-01T10:30:00Z",
                            "details": {
                                "limit": 10,
                                "window": "1 minute",
                                "reset_time": "2024-01-01T10:31:00Z"
                            },
                            "request_id": "req_abc123def456"
                        }
                    }
                }
            },
            "500": {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "error": "internal_error",
                            "message": "An unexpected error occurred",
                            "timestamp": "2024-01-01T10:30:00Z",
                            "request_id": "req_abc123def456"
                        }
                    }
                }
            }
        }
        
        # Apply common responses to all endpoints
        for path_info in paths.values():
            for method_info in path_info.values():
                if isinstance(method_info, dict) and "responses" in method_info:
                    # Add common error responses if not already present
                    for code, response in common_responses.items():
                        if code not in method_info["responses"]:
                            method_info["responses"][code] = response
    
    @staticmethod
    def setup_documentation_routes(app: FastAPI) -> None:
        """Setup custom documentation routes with enhanced UI."""
        
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            """Custom Swagger UI with enhanced styling."""
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="SimGen AI API - Interactive Documentation",
                swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
                swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
                swagger_ui_parameters={
                    "deepLinking": True,
                    "displayRequestDuration": True,
                    "defaultModelsExpandDepth": 2,
                    "defaultModelExpandDepth": 2,
                    "docExpansion": "list",
                    "filter": True,
                    "showRequestHeaders": True,
                    "syntaxHighlight.activate": True,
                    "syntaxHighlight.theme": "nord",
                    "tryItOutEnabled": True,
                    "persistAuthorization": True
                }
            )
        
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            """Alternative ReDoc documentation interface."""
            return get_redoc_html(
                openapi_url="/openapi.json",
                title="SimGen AI API - ReDoc Documentation",
                redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
            )
    
    @staticmethod
    def add_endpoint_documentation(
        app: FastAPI,
        endpoint_docs: Dict[str, Dict[str, Any]]
    ) -> None:
        """Add documentation metadata to specific endpoints."""
        
        for endpoint_path, doc_info in endpoint_docs.items():
            # Find the route and add documentation
            for route in app.routes:
                if hasattr(route, 'path') and route.path == endpoint_path:
                    if hasattr(route, 'endpoint') and hasattr(route.endpoint, '__doc__'):
                        # Enhance existing docstring with structured info
                        if doc_info.get("summary"):
                            route.summary = doc_info["summary"]
                        if doc_info.get("description"):
                            route.description = doc_info["description"]
                        if doc_info.get("tags"):
                            route.tags = doc_info["tags"]


# Pre-defined endpoint documentation
ENDPOINT_DOCUMENTATION = {
    "/api/simulation/generate": {
        "summary": "Generate Physics Simulation",
        "description": """
        Generate a complete physics simulation from a natural language prompt.
        
        This endpoint uses advanced AI models to:
        1. Parse and understand the simulation requirements
        2. Generate appropriate MuJoCo XML simulation code
        3. Execute the simulation with physics validation
        4. Render high-quality visual output
        
        **Processing Time**: Typically 5-15 seconds depending on complexity
        
        **Rate Limit**: 10 requests per minute per API key
        """,
        "tags": ["Simulation"]
    },
    "/api/sketch/analyze": {
        "summary": "Analyze Hand-Drawn Sketch", 
        "description": """
        Convert a hand-drawn sketch into a physics simulation.
        
        Supports:
        - PNG and JPEG image formats
        - Sketches up to 10MB in size
        - Multiple objects and relationships
        - Physics-based interpretation
        
        **Processing Time**: 10-20 seconds for detailed analysis
        """,
        "tags": ["Sketch Analysis"]
    },
    "/health": {
        "summary": "Basic Health Check",
        "description": "Simple health check endpoint for load balancers and monitoring systems.",
        "tags": ["Health"]
    },
    "/health/detailed": {
        "summary": "Detailed System Health",
        "description": """
        Comprehensive health check including:
        - Database connectivity
        - External service status  
        - Resource utilization
        - Circuit breaker states
        - Performance metrics
        """,
        "tags": ["Health"]
    }
}