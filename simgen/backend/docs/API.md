# SimGen API Documentation

## Overview

SimGen provides a RESTful API for generating physics simulations from natural language prompts.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required. API keys for LLM services are configured server-side.

## Endpoints

### Health Check
**GET** `/health`

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-07T17:41:00.000Z",
  "version": "1.0.0",
  "services": {
    "database": "ok",
    "redis": "ok", 
    "llm": "ok"
  }
}
```

### Generate Simulation
**POST** `/api/v1/simulation/generate`

Generates a physics simulation from a natural language prompt.

**Request Body:**
```json
{
  "prompt": "A robotic arm picking up colorful balls from a table",
  "session_id": "optional-session-id",
  "max_iterations": 5
}
```

**Response:**
```json
{
  "id": 123,
  "session_id": "uuid-string",
  "user_prompt": "A robotic arm picking up colorful balls from a table",
  "status": "pending",
  "created_at": "2025-09-07T17:41:00.000Z"
}
```

### Test Generation (No Database)
**POST** `/api/v1/simulation/test-generate`

Generates a simulation without storing in database. Useful for testing.

**Request Body:**
```json
{
  "prompt": "Multiple spheres bouncing with realistic physics"
}
```

**Response:**
```json
{
  "status": "success",
  "prompt": "Multiple spheres bouncing with realistic physics",
  "entities": {
    "objects": 2,
    "constraints": 0,
    "environment": "standard gravity"
  },
  "generation_method": "dynamic_composition",
  "mjcf_content": "<mujoco model=\"...\">...</mujoco>",
  "success": true
}
```

### Get Simulation
**GET** `/api/v1/simulation/{simulation_id}`

Retrieves a specific simulation by ID.

**Response:**
```json
{
  "id": 123,
  "session_id": "uuid-string",
  "user_prompt": "A robotic arm...",
  "status": "completed",
  "mjcf_content": "<mujoco model=\"...\">...</mujoco>",
  "generation_method": "dynamic_composition",
  "created_at": "2025-09-07T17:41:00.000Z",
  "updated_at": "2025-09-07T17:41:30.000Z"
}
```

### List Simulations
**GET** `/api/v1/simulation/`

Lists all simulations with pagination.

**Query Parameters:**
- `skip`: Number of records to skip (default: 0)
- `limit`: Number of records to return (default: 100)

## WebSocket

### Real-time Progress Updates
**WS** `/ws/{session_id}`

Connect to receive real-time progress updates during simulation generation.

**Messages:**
```json
{
  "type": "progress_update",
  "session_id": "uuid-string",
  "data": {
    "stage": "parsing",
    "progress": 0.2,
    "message": "Parsing simulation prompt..."
  }
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "detail": {
    "error": "Error message",
    "code": "ERROR_CODE",
    "timestamp": "2025-09-07T17:41:00.000Z"
  }
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid prompt/parameters)
- `404`: Not Found (simulation doesn't exist)
- `422`: Validation Error (malformed request)
- `500`: Internal Server Error

## Rate Limits

Currently no rate limits are enforced, but the following guidelines apply:
- Maximum prompt length: 1000 characters
- Maximum concurrent requests per session: 5
- Generation timeout: 300 seconds

## Examples

### cURL Examples

**Generate a robotic simulation:**
```bash
curl -X POST "http://localhost:8000/api/v1/simulation/test-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A robotic arm with 6 degrees of freedom manipulating objects"
  }'
```

**Generate a physics demonstration:**
```bash
curl -X POST "http://localhost:8000/api/v1/simulation/test-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Three balls with different materials bouncing on a table"
  }'
```

### Python Client Example

```python
import requests
import json

# Generate simulation
response = requests.post(
    "http://localhost:8000/api/v1/simulation/test-generate",
    json={"prompt": "A pendulum swinging with realistic physics"},
    headers={"Content-Type": "application/json"}
)

result = response.json()
if result["success"]:
    # Save the generated MJCF
    with open("simulation.xml", "w") as f:
        f.write(result["mjcf_content"])
    print("Simulation generated successfully!")
else:
    print(f"Generation failed: {result.get('error', 'Unknown error')}")
```

## MJCF Output Format

Generated simulations are returned as MuJoCo XML Format (MJCF) strings containing:

- **Professional lighting**: Multi-light setups with shadows
- **High-quality materials**: Specular, reflectance, textures
- **Physics objects**: Bodies, joints, constraints as specified
- **Visual settings**: 4096x4096 shadows, anti-aliasing
- **Environment**: Professional ground planes and skyboxes

Example output structure:
```xml
<mujoco model="ai_simulation">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
  <asset>
    <!-- Professional textures and materials -->
  </asset>
  <worldbody>
    <!-- Professional lighting -->
    <!-- Generated objects and constraints -->
  </worldbody>
  <visual>
    <!-- High-quality visual settings -->
  </visual>
</mujoco>
```