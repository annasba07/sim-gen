# Why Vercel Won't Work for SimGen AI Backend

## ❌ Critical Limitations

### **1. 15-Second Timeout Limit**
```python
# Our simulation generation takes 10-30 seconds
async def generate_simulation(prompt: str):
    start_time = time.time()
    
    # LLM call: 5-15 seconds
    analysis = await llm_client.analyze(prompt)
    
    # MuJoCo generation: 3-8 seconds  
    mjcf = await composer.generate(analysis)
    
    # Rendering: 5-12 seconds
    video = await renderer.render(mjcf)
    
    total_time = time.time() - start_time
    # total_time = 15-35 seconds ⚠️ EXCEEDS VERCEL LIMIT
    
    return SimulationResult(video=video)
```

**Vercel Result:** ❌ `Function execution timed out after 15.00 seconds`

### **2. No Persistent Storage**
```python
# We need to save simulation videos/images
video_path = "/storage/videos/sim_12345.mp4"  # ❌ Gets deleted
await save_video(video_path, video_data)      # ❌ No persistence

# User requests video later
video = load_video(video_path)  # ❌ File gone!
```

### **3. No Long-Running Services**
```python
# We need persistent connections
redis_client = redis.Redis(...)      # ❌ Connection drops
db_pool = ConnectionPool(...)        # ❌ Pool resets
llm_client = AnthropicClient(...)    # ❌ No state

# Circuit breakers need memory
circuit_breaker.failure_count = 5   # ❌ Resets every request
```

### **4. Limited Memory/CPU**
```python
# MuJoCo physics simulation
model = mujoco.MjModel.from_xml(mjcf)  # ❌ High memory usage
data = mujoco.MjData(model)           # ❌ CPU intensive
for i in range(1000):                 # ❌ Heavy computation
    mujoco.mj_step(model, data)       # ❌ May hit limits
```

## ✅ What Vercel IS Good For

### **Frontend Only:**
```yaml
# vercel.json - Frontend deployment only
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "functions": {
    "app/api/hello.js": {
      "maxDuration": 10  # Still limited!
    }
  }
}
```

**Perfect for:**
- Static frontend hosting
- Simple API routes  
- Authentication flows
- Client-side logic

**Not good for:**
- Heavy computation
- Long-running processes
- File storage
- Database connections
- AI/ML workloads