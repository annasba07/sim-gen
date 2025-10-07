# 🚀 VirtualForge Render Deployment Guide

Deploy VirtualForge backend to Render.com in under 5 minutes.

---

## 📋 Prerequisites

- GitHub account (to connect your repo)
- Render.com account (free tier available)
- API keys:
  - Anthropic API key (for Claude)
  - OpenAI API key (for GPT-4)

---

## 🎯 Quick Deploy (5 Minutes)

### Step 1: Push Code to GitHub

```bash
# If you haven't already, initialize git and push to GitHub
git init
git add .
git commit -m "Ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin master
```

### Step 2: Deploy to Render

1. **Go to Render Dashboard**
   - Visit: https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Select "Connect a repository"
   - Authorize GitHub access
   - Select your VirtualForge repository

3. **Configure Service**
   - **Name**: `virtualforge-backend` (or any name you prefer)
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `master` (or `main`)
   - **Root Directory**: Leave empty (uses project root)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.simgen.main_clean:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   
   Click "Advanced" and add these environment variables:
   
   | Key | Value |
   |-----|-------|
   | `PYTHON_VERSION` | `3.11` |
   | `ANTHROPIC_API_KEY` | `sk-ant-your-key-here` |
   | `OPENAI_API_KEY` | `sk-your-key-here` |
   | `DEBUG` | `false` |
   | `CORS_ORIGINS` | `*` |

5. **Select Plan**
   - **Free Tier**: Good for testing (spins down after inactivity)
   - **Starter ($7/mo)**: Stays always on, faster

6. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy
   - Wait 3-5 minutes for first deployment

---

## ✅ Verify Deployment

Once deployed, you'll get a URL like: `https://virtualforge-backend-xxxx.onrender.com`

### Test Health Endpoint

```bash
curl https://YOUR_URL.onrender.com/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-10-04T...",
  "services": {
    "llm": "connected",
    "mode_registry": "ready"
  }
}
```

### Test Templates Endpoint

```bash
curl https://YOUR_URL.onrender.com/api/v2/games/templates
```

**Expected Response:**
```json
[
  {
    "id": "coin-collector",
    "name": "Coin Collector",
    "type": "platformer",
    "description": "Classic platformer where you collect coins..."
  },
  ...
]
```

### Generate Your First Game

```bash
curl -X POST https://YOUR_URL.onrender.com/api/v2/games/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A platformer where you jump on platforms and collect stars",
    "gameType": "platformer",
    "complexity": "simple"
  }' | jq -r '.html' > game.html

# Open in browser
open game.html
```

You should see a playable game! 🎮

---

## 🔧 Using render.yaml (Alternative Method)

If your repo has a `render.yaml` file (it does!), Render can auto-configure:

1. Go to https://dashboard.render.com
2. Click "New +" → "Blueprint"
3. Select your repository
4. Render reads `render.yaml` and creates services automatically
5. Just add your API keys as environment variables
6. Click "Apply"

---

## 📊 Monitoring

### View Logs

```bash
# In Render Dashboard:
# Your Service → Logs tab → Real-time logs
```

### Check Metrics

- **Dashboard**: CPU, Memory, Request count
- **Events**: Deploy history, restarts
- **Logs**: Application logs, errors

### Health Checks

Render automatically monitors `/health` endpoint and restarts if unhealthy.

---

## 🔄 Auto-Deploy on Push

Render automatically deploys when you push to your connected branch:

```bash
git add .
git commit -m "Update game templates"
git push

# Render automatically:
# 1. Detects push
# 2. Builds new image
# 3. Deploys with zero downtime
```

---

## 🚨 Troubleshooting

### Build Fails

**Error**: `ModuleNotFoundError: No module named 'X'`

**Fix**: Ensure `requirements.txt` includes all dependencies
```bash
# Check requirements.txt has:
fastapi
uvicorn
pydantic>=2.0
anthropic
openai
```

### Service Won't Start

**Error**: Application error or timeout

**Fix**: Check logs in Render dashboard
```bash
# Common issues:
# 1. Missing environment variables
# 2. PORT not set (Render sets this automatically)
# 3. Wrong start command
```

### CORS Errors

**Error**: CORS policy blocking requests

**Fix**: Update CORS_ORIGINS environment variable
```bash
# In Render Dashboard:
# Environment → Edit → CORS_ORIGINS → Add your frontend URL
```

### Free Tier Spin Down

**Issue**: Service takes 30-60s to respond after inactivity

**Fix**: Upgrade to Starter plan ($7/mo) for always-on service

---

## 💰 Pricing

### Free Tier
- ✅ 750 hours/month (enough for one service)
- ✅ Auto-scaling
- ❌ Spins down after 15 min inactivity
- ❌ Slower cold starts

### Starter - $7/month
- ✅ Always on (no spin down)
- ✅ Faster performance
- ✅ Custom domains
- ✅ Better for production

### Professional - $25/month
- ✅ Everything in Starter
- ✅ More resources
- ✅ Priority support

**Recommended**: Start with Free tier for testing, upgrade to Starter for production

---

## 🎨 Next Steps

### 1. Add Custom Domain (Optional)

```bash
# In Render Dashboard:
# Your Service → Settings → Custom Domain
# Add: virtualforge.yourdomain.com
# Update DNS: CNAME → your-service.onrender.com
```

### 2. Enable HTTPS (Automatic)

Render automatically provisions SSL certificates for all services.

### 3. Set Up Monitoring

```bash
# Consider adding:
# - Sentry for error tracking
# - Datadog/New Relic for APM
# - Uptime monitoring (UptimeRobot, Pingdom)
```

### 4. Scale Up

```bash
# In Render Dashboard:
# Your Service → Settings → Instance Type
# Upgrade for more CPU/RAM if needed
```

---

## 🧪 Test API Endpoints

Once deployed, test all game endpoints:

```bash
# Set your URL
export API_URL="https://YOUR_SERVICE.onrender.com"

# 1. Health check
curl $API_URL/health

# 2. List templates
curl $API_URL/api/v2/games/templates

# 3. Get specific template
curl $API_URL/api/v2/games/templates/coin-collector

# 4. Generate platformer
curl -X POST $API_URL/api/v2/games/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Collect coins on platforms", "gameType": "platformer"}' \
  | jq -r '.html' > platformer.html

# 5. Generate top-down game
curl -X POST $API_URL/api/v2/games/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explore a dungeon", "gameType": "topdown"}' \
  | jq -r '.html' > topdown.html

# 6. Generate shooter
curl -X POST $API_URL/api/v2/games/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Shoot asteroids in space", "gameType": "shooter"}' \
  | jq -r '.html' > shooter.html

# Open all games
open platformer.html topdown.html shooter.html
```

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Render Status**: https://status.render.com
- **Community**: https://community.render.com

---

## 🎉 You're Live!

Your VirtualForge backend is now deployed and accessible worldwide!

**Share your API:**
- Backend: `https://your-service.onrender.com`
- Health: `https://your-service.onrender.com/health`
- Games API: `https://your-service.onrender.com/api/v2/games`

**Next**: Build a frontend or integrate with existing apps! 🚀
