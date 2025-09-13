# SimGen AI - Sketch to Physics Simulation

Transform your sketches into interactive physics simulations with the power of AI.

## 🎨 What is SimGen AI?

SimGen AI is a revolutionary application that combines computer vision, large language models, and physics simulation to transform hand-drawn sketches into realistic, interactive 3D physics simulations. Simply draw your physics idea, add a text description, and watch as AI brings it to life.

## ✨ Features

- **🖼️ Sketch Canvas** - Draw physics concepts directly in the browser
- **🧠 AI Vision Analysis** - Advanced computer vision understands your drawings  
- **💬 Multi-Modal Input** - Combine sketches with text descriptions
- **⚡ Real-Time Processing** - Watch AI analyze and enhance your ideas
- **🎮 3D Physics Viewer** - Interactive simulations powered by MuJoCo physics
- **🚀 Modern Tech Stack** - Built with Next.js, React, FastAPI, and Three.js

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **AI Services**: Vision analysis, multi-modal prompt enhancement
- **Physics Engine**: MuJoCo integration for realistic simulations
- **LLM Integration**: Anthropic Claude + OpenAI GPT-4o for AI processing
- **API Endpoints**: RESTful APIs with WebSocket support for real-time updates

### Frontend (Next.js + React)  
- **Interactive Canvas**: HTML5 Canvas with touch/mouse drawing support
- **3D Visualization**: Three.js powered physics simulation viewer
- **Modern UI**: Tailwind CSS with Framer Motion animations
- **Real-Time Updates**: WebSocket integration for live processing feedback

## 🚀 Quick Start

### Backend Setup
```bash
cd simgen/backend
pip install -r requirements.txt
uvicorn src.simgen.main:app --reload
```

### Frontend Setup  
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
```bash
# Backend (.env)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
DATABASE_URL=postgresql://...

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🎯 How It Works

1. **Draw Your Idea** - Sketch physics concepts like pendulums, robot arms, or bouncing balls
2. **Add Description** - Enhance with text like "make this swing faster" or "add collision detection"  
3. **AI Processing** - Watch as AI analyzes your sketch and generates physics descriptions
4. **Interactive Simulation** - Explore your creation in real-time 3D with full physics

## 🧪 Testing

### Backend Tests
```bash
cd simgen/backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🛠️ Technology Stack

- **AI/ML**: Anthropic Claude, OpenAI GPT-4o, Computer Vision
- **Physics**: MuJoCo Physics Engine  
- **Backend**: FastAPI, Python, PostgreSQL, Redis
- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **3D Graphics**: Three.js, React Three Fiber
- **Animation**: Framer Motion
- **Testing**: Pytest, Jest, React Testing Library

## 📈 Project Status

🔧 **Active Development** - Core sketch-to-physics pipeline complete, frontend interface ready, comprehensive test suite implemented.

## 🤝 Contributing

This project demonstrates the future of AI-powered physics simulation. Contributions welcome!

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ by the SimGen AI Team**

*Transform imagination into simulation*