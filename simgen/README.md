# SimGen - AI Physics Simulation Generator

An advanced system that generates physics simulations from natural language prompts using state-of-the-art LLMs and MuJoCo physics engine.

## Features

- 🤖 **Natural Language to Physics**: Convert text descriptions into working physics simulations
- ⚡ **MuJoCo Integration**: Leverage MuJoCo-Warp for high-performance GPU-accelerated physics
- 🔄 **Feedback Loop System**: Iterative refinement using vision models and physics validation
- 🏗️ **Hybrid Generation**: Template-based and LLM-based generation with automatic fallback
- 📊 **Real-time Progress**: WebSocket-based progress updates during generation
- 🎯 **Quality Assessment**: Automated quality scoring using multiple metrics

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Anthropic API key (required)
- OpenAI API key (optional, for fallback)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd simgen
```

2. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# ANTHROPIC_API_KEY=your-key-here
# OPENAI_API_KEY=your-openai-key-optional
```

3. **Start the development environment:**
```bash
docker-compose up --build
```

4. **Initialize the database:**
```bash
# In a new terminal
docker-compose exec api alembic upgrade head
```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend** (when implemented): http://localhost:3000

## API Usage

### Generate a Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulation/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a simple pendulum with a 1kg mass swinging from a 1 meter string",
    "max_iterations": 5
  }'
```

### Get Simulation Status

```bash
curl "http://localhost:8000/api/v1/simulation/{simulation_id}"
```

### WebSocket Connection

Connect to `ws://localhost:8000/api/v1/simulation/ws/{session_id}` for real-time progress updates.

## Project Structure

```
simgen/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core configuration
│   │   ├── db/             # Database setup
│   │   ├── models/         # Database and Pydantic models
│   │   └── services/       # Business logic
│   ├── migrations/         # Database migrations
│   └── tests/              # Test suite
├── frontend/               # React frontend (future)
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Development

### Local Development Setup

1. **Create virtual environment:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate   # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up local database:**
```bash
# Start PostgreSQL and Redis locally or use Docker
docker-compose up postgres redis
```

4. **Run migrations:**
```bash
alembic upgrade head
```

5. **Start development server:**
```bash
uvicorn app.main:app --reload
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_simulation.py -v
```

### Database Operations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Architecture Overview

### Phase 1 Implementation (Current)

- **Hybrid Feedback System**: Combines template-based and LLM-based generation
- **Real-time Processing**: Asynchronous task processing with WebSocket updates
- **Multi-modal Analysis**: Physics validation and visual quality assessment
- **Iterative Refinement**: Automatic simulation improvement loops

### Core Components

1. **Prompt Parser**: Extracts structured entities from natural language
2. **Simulation Generator**: Creates MJCF files using templates or LLM generation
3. **Physics Engine**: MuJoCo integration for simulation execution
4. **Quality Analyzer**: Multi-modal assessment of simulation quality
5. **Feedback Loop**: Iterative refinement based on analysis results

### Technology Stack

- **Backend**: FastAPI + SQLAlchemy + AsyncIO
- **Database**: PostgreSQL + Redis
- **Physics**: MuJoCo + PyBullet (fallback)
- **AI**: Anthropic Claude + OpenAI GPT (fallback)
- **Containerization**: Docker + Docker Compose

## Configuration

Key environment variables:

- `ANTHROPIC_API_KEY`: Required for LLM operations
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `MAX_SIMULATION_DURATION`: Simulation time limit (seconds)
- `MAX_REFINEMENT_ITERATIONS`: Maximum refinement loops
- `MUJOCO_GL`: Rendering backend for headless operation

## Performance & Limits

- **Generation Time**: <30 seconds average (target)
- **Success Rate**: >90% for valid prompts (target)
- **Concurrent Users**: Up to 10 concurrent simulations
- **Simulation Duration**: Maximum 30 seconds per simulation
- **Refinement Iterations**: Up to 5 iterations per generation

## Monitoring

Health check endpoint provides status for:
- Database connectivity
- Redis connectivity
- LLM API availability
- System resources

## Troubleshooting

### Common Issues

1. **MuJoCo GL Issues**:
```bash
# For headless servers
export MUJOCO_GL=egl
```

2. **Database Connection**:
```bash
# Check PostgreSQL is running
docker-compose logs postgres
```

3. **API Key Issues**:
```bash
# Verify environment variables
docker-compose exec api env | grep API_KEY
```

### Logs

```bash
# View application logs
docker-compose logs -f api

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f redis
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the API documentation at `/docs`
- Review logs for error messages
- Ensure all environment variables are set correctly
- Verify API keys are valid and have sufficient quota