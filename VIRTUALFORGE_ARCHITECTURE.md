# 🏗️ VirtualForge - Unified Architecture

## Product Vision

**VirtualForge: From Ideas to Interactive Worlds**

A single platform with multiple creation modes - physics simulations, games, and future VR/AR experiences.

---

## Core Architecture Principles

1. **Shared Core (95%)** - Single codebase for all modes
2. **Mode-Specific Compilers (5%)** - Different output generators
3. **Unified User Experience** - Seamless mode switching
4. **Progressive Enhancement** - Start simple, add complexity

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VirtualForge Platform                 │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │ Physics │        │  Games  │        │   VR    │
   │  Mode   │        │  Mode   │        │  Mode   │
   │  🔬     │        │  🎮     │        │  🌐     │
   └────┬────┘        └────┬────┘        └────┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │         Shared Services Layer          │
        │                                        │
        │  • Sketch Analysis (CV Pipeline)       │
        │  • LLM Integration (Claude/GPT)        │
        │  • Caching (Redis)                     │
        │  • Real-time (WebSocket)               │
        │  • Auth & User Management              │
        │  • Database (PostgreSQL)               │
        └────────────────────────────────────────┘
```

---

## Directory Structure

```
virtualforge/
│
├── backend/
│   ├── core/                      # 🔄 SHARED (95%)
│   │   ├── container.py          # DI container
│   │   ├── interfaces.py         # Service contracts
│   │   ├── config.py             # Unified configuration
│   │   ├── validation.py         # Request validation
│   │   ├── retry_logic.py        # Resilience patterns
│   │   └── exceptions.py         # Error handling
│   │
│   ├── services/                  # 🔄 SHARED (95%)
│   │   ├── sketch/               # CV analysis
│   │   │   ├── analyzer.py
│   │   │   └── cv_pipeline.py
│   │   ├── ai/                   # LLM integration
│   │   │   ├── llm_client.py
│   │   │   └── prompt_templates.py
│   │   ├── infrastructure/       # Platform services
│   │   │   ├── cache.py
│   │   │   ├── websocket.py
│   │   │   └── storage.py
│   │   └── realtime/
│   │       └── feedback.py
│   │
│   ├── modes/                     # ⚡ MODE-SPECIFIC (5%)
│   │   ├── physics/              # Physics mode
│   │   │   ├── compiler.py       # MJCF generator
│   │   │   ├── simulator.py      # MuJoCo runner
│   │   │   └── templates.py      # Physics templates
│   │   │
│   │   ├── games/                # Game mode
│   │   │   ├── compiler.py       # Phaser generator
│   │   │   ├── behaviors.py      # Game components
│   │   │   ├── templates.py      # Game templates
│   │   │   └── deployer.py       # CDN deployment
│   │   │
│   │   └── common/               # Shared mode utilities
│   │       ├── spec_parser.py
│   │       └── asset_manager.py
│   │
│   ├── api/                       # 🔄 MOSTLY SHARED
│   │   ├── v1/                   # Legacy endpoints
│   │   ├── v2/
│   │   │   ├── sketch.py         # Sketch upload (shared)
│   │   │   ├── physics.py        # Physics endpoints
│   │   │   ├── games.py          # Game endpoints
│   │   │   ├── templates.py      # All templates
│   │   │   └── realtime.py       # WebSocket (shared)
│   │   └── health.py             # Health checks
│   │
│   └── main.py                    # Application entry
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Landing + mode selection
│   │   │   ├── physics/
│   │   │   │   └── page.tsx      # Physics creation
│   │   │   ├── games/
│   │   │   │   └── page.tsx      # Game creation
│   │   │   └── layout.tsx        # Shared layout
│   │   │
│   │   ├── components/
│   │   │   ├── shared/           # 🔄 SHARED (90%)
│   │   │   │   ├── sketch-canvas.tsx
│   │   │   │   ├── prompt-input.tsx
│   │   │   │   ├── mode-selector.tsx
│   │   │   │   └── realtime-feedback.tsx
│   │   │   ├── physics/          # ⚡ PHYSICS-SPECIFIC
│   │   │   │   ├── simulation-viewer.tsx
│   │   │   │   └── physics-controls.tsx
│   │   │   └── games/            # ⚡ GAME-SPECIFIC
│   │   │       ├── game-preview.tsx
│   │   │       └── game-controls.tsx
│   │   │
│   │   ├── lib/
│   │   │   ├── api/
│   │   │   │   ├── physics-api.ts
│   │   │   │   ├── games-api.ts
│   │   │   │   └── common-api.ts
│   │   │   └── types/
│   │   │       ├── physics.ts
│   │   │       └── games.ts
│   │   │
│   │   └── hooks/
│   │       ├── use-mode.ts
│   │       ├── use-realtime-feedback.ts
│   │       └── use-creation-state.ts
│   │
│   └── public/
│       ├── physics/              # Physics assets
│       └── games/                # Game assets
│
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── PHYSICS_MODE.md
    ├── GAMES_MODE.md
    └── CONTRIBUTING.md
```

---

## Mode Configuration

### Physics Mode
```yaml
id: physics
name: Physics Lab
description: Scientific simulations & education
icon: 🔬
color: blue
engines:
  - mujoco
features:
  - sketch_analysis
  - physics_compilation
  - 3d_visualization
  - educational_templates
target_users:
  - educators
  - students
  - researchers
  - engineers
```

### Games Mode
```yaml
id: games
name: Game Studio
description: 60-second game creation
icon: 🎮
color: purple
engines:
  - phaser
  - babylon
  - unity (future)
  - roblox (future)
features:
  - sketch_analysis
  - game_compilation
  - instant_preview
  - remix_system
target_users:
  - content_creators
  - streamers
  - kids
  - hobbyists
```

---

## Shared Services

### 1. Sketch Analysis Service
```python
class SketchAnalyzer:
    """Analyzes sketches for ALL modes"""

    async def analyze(self, image: bytes, mode: str) -> SketchAnalysis:
        # Same CV pipeline
        # Different interpretation per mode
        objects = await self.detect_objects(image)

        if mode == "physics":
            return self.interpret_as_physics(objects)
        elif mode == "games":
            return self.interpret_as_game(objects)
```

### 2. LLM Integration Service
```python
class LLMClient:
    """AI generation for ALL modes"""

    async def generate(self, prompt: str, mode: str, sketch: dict):
        system_prompt = self.get_mode_prompt(mode)

        response = await self.claude.complete(
            system=system_prompt,
            user=f"Sketch: {sketch}\nPrompt: {prompt}"
        )

        return response
```

### 3. Caching Service
```python
class CacheService:
    """Unified caching for all modes"""

    async def get_or_create(self, key: str, factory):
        # Same Redis for all modes
        cached = await self.redis.get(key)
        if cached:
            return cached

        result = await factory()
        await self.redis.set(key, result)
        return result
```

---

## API Structure

### Unified Endpoints

```
GET  /                          # Landing page
GET  /health                    # Health check

# Shared
POST /api/v2/sketch/upload      # Upload sketch (all modes)
GET  /api/v2/templates          # Get all templates
WS   /api/v2/realtime/{mode}    # Real-time feedback

# Physics Mode
POST /api/v2/physics/generate   # Generate simulation
POST /api/v2/physics/compile    # Compile MJCF
POST /api/v2/physics/simulate   # Run simulation

# Games Mode
POST /api/v2/games/generate     # Generate game
POST /api/v2/games/compile      # Compile to Phaser
POST /api/v2/games/deploy       # Deploy to CDN
```

---

## Data Models

### Unified Creation
```typescript
interface Creation {
  id: string
  user_id: string
  mode: 'physics' | 'games' | 'vr'

  // Input
  prompt: string
  sketch_url?: string

  // Mode-specific output
  output: PhysicsOutput | GameOutput | VROutput

  // Metadata
  created_at: Date
  updated_at: Date
  public: boolean
  remix_of?: string
}
```

### Mode-Specific Outputs
```typescript
type PhysicsOutput = {
  type: 'physics'
  mjcf: string
  simulation_url: string
  stats: PhysicsStats
}

type GameOutput = {
  type: 'game'
  game_code: string
  play_url: string
  engine: 'phaser' | 'babylon'
  stats: GameStats
}
```

---

## User Flow

```
1. User lands on VirtualForge.ai
   ↓
2. "What do you want to create?"
   ├─→ 🔬 Physics Lab
   │   └─→ Draw/describe → Generate → Simulate
   │
   ├─→ 🎮 Game Studio
   │   └─→ Draw/describe → Generate → Play
   │
   └─→ 🌐 VR Worlds (coming soon)

3. Creation saved to "My Projects"
   - All modes in one list
   - Filter by mode
   - Easy switching

4. Upgrade prompt
   - "Want to add game mechanics to this physics sim?"
   - "Want realistic physics in your game?"
```

---

## Technology Stack

### Shared Backend
- FastAPI (Python 3.11+)
- Redis (caching)
- PostgreSQL (storage)
- Anthropic Claude / OpenAI GPT-4

### Mode-Specific
- Physics: MuJoCo, NumPy, SciPy
- Games: Phaser 3, Babylon.js
- Future: Unity, Three.js, Roblox

### Frontend
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Shadcn/ui

---

## Deployment

```yaml
Development:
  - Local: npm run dev & uvicorn main:app --reload
  - Modes: Both available immediately

Production:
  - Frontend: Vercel (edge functions)
  - Backend: Railway/Fly.io
  - Database: Supabase
  - Cache: Upstash Redis
  - CDN: Cloudflare
```

---

## Metrics to Track

```yaml
Per Mode:
  - Creations started
  - Creations completed
  - Average time to create
  - User satisfaction

Cross-Mode:
  - Mode switching rate
  - Cross-pollination (physics → games)
  - Upgrade conversion
  - Remix rate
```

---

## Future Expansion

```
Q1 2025: Physics + Games
Q2 2025: Add VR/AR mode
Q3 2025: Add Collaboration mode
Q4 2025: Add Marketplace
```

---

## Success Criteria

✅ User can switch modes seamlessly
✅ 95% code sharing achieved
✅ Mode-specific features work independently
✅ Unified branding and UX
✅ Single deployment, single domain
✅ Cross-mode discovery and upgrades work

**VirtualForge = One platform, infinite possibilities** 🚀