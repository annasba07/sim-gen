# 🚀 VirtualForge Deployment Guide

Quick guide to deploy VirtualForge to production.

---

## 🎯 Quick Deploy (Fastest Path)

### Option 1: Railway + Vercel (Recommended)

#### Backend → Railway

```bash
# 1. Install Railway CLI
brew install railway  # macOS
# or: npm install -g @railway/cli

# 2. Login
railway login

# 3. Create new project
railway init

# 4. Set environment variables
railway variables set ANTHROPIC_API_KEY=your_key_here
railway variables set OPENAI_API_KEY=your_key_here
railway variables set DATABASE_URL=postgresql://...  # if using postgres
railway variables set REDIS_URL=redis://...          # if using redis

# 5. Deploy
railway up

# 6. Get your URL
railway domain
# Returns: https://your-app.railway.app
```

#### Frontend → Vercel

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy from frontend directory
cd frontend
vercel --prod

# 3. Set environment variable
vercel env add NEXT_PUBLIC_API_URL
# Enter: https://your-app.railway.app

# 4. Redeploy with env var
vercel --prod

# Done! Returns: https://your-app.vercel.app
```

---

## Option 2: Fly.io + Vercel

### Backend → Fly.io

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Create app
fly launch --no-deploy

# 4. Set secrets
fly secrets set ANTHROPIC_API_KEY=your_key_here
fly secrets set OPENAI_API_KEY=your_key_here

# 5. Deploy
fly deploy

# Done! Returns: https://your-app.fly.dev
```

---

## 📋 Environment Variables Needed

### Backend (.env)
```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional (for full features)
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379

# App config
DEBUG=false
CORS_ORIGINS=https://your-frontend.vercel.app
HOST=0.0.0.0
PORT=8000
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

---

## 🧪 Test Deployment

### 1. Test Backend
```bash
curl https://your-backend.railway.app/health

# Should return: {"status": "healthy", ...}
```

### 2. Test Games API
```bash
curl https://your-backend.railway.app/api/v2/games/templates

# Should return: [{"id": "coin-collector", ...}, ...]
```

### 3. Test Frontend
```bash
# Open in browser
open https://your-frontend.vercel.app/virtualforge

# Should see: Mode selector with Physics/Games/VR
```

### 4. Test End-to-End
1. Visit frontend
2. Select "Game Studio"
3. Choose platformer
4. Enter: "A game where you collect stars"
5. Click "Generate Game"
6. Should see: Playable game in 3-10 seconds

---

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check logs
railway logs  # Railway
fly logs      # Fly.io

# Common issues:
# - Missing env vars → Set with railway variables set
# - Port binding → Use $PORT env var (done in Procfile)
# - Dependencies → Check requirements.txt
```

### Frontend won't connect to backend
```bash
# Check env var is set
vercel env ls

# Update if needed
vercel env add NEXT_PUBLIC_API_URL
# Value: https://your-backend.railway.app

# Redeploy
vercel --prod
```

### CORS errors
```bash
# Update backend CORS_ORIGINS
railway variables set CORS_ORIGINS=https://your-frontend.vercel.app

# Redeploy
railway up
```

### LLM generation fails
```bash
# Check API keys are set
railway variables

# Verify keys work
curl https://your-backend.railway.app/api/v2/games/templates
# Should work without API keys (uses fallback templates)
```

---

## 🎯 Quick Deploy Script

Save as `deploy.sh`:

```bash
#!/bin/bash

echo "🚀 Deploying VirtualForge..."

# Backend
echo "\n📦 Deploying backend to Railway..."
railway up
BACKEND_URL=$(railway domain | grep -o 'https://[^"]*')
echo "✅ Backend: $BACKEND_URL"

# Frontend
echo "\n🎨 Deploying frontend to Vercel..."
cd frontend
vercel env add NEXT_PUBLIC_API_URL production <<EOF
$BACKEND_URL
EOF
vercel --prod
cd ..

echo "\n🎉 Deployment complete!"
echo "Backend:  $BACKEND_URL"
echo "Frontend: https://your-app.vercel.app"
```

Make executable: `chmod +x deploy.sh`

Run: `./deploy.sh`

---

## 🌐 Custom Domain (Optional)

### Railway
```bash
railway domain add your-domain.com
# Add CNAME: your-domain.com → your-app.railway.app
```

### Vercel
```bash
vercel domains add your-frontend.com
# Add CNAME: your-frontend.com → cname.vercel-dns.com
```

---

## 📊 Monitoring

### Railway Dashboard
- Logs: `railway logs -f`
- Metrics: https://railway.app/dashboard
- Restart: `railway restart`

### Vercel Dashboard
- Logs: https://vercel.com/dashboard
- Analytics: Built-in
- Deployments: Track all versions

---

## 🔐 Security Checklist

- [ ] Environment variables set (not in code)
- [ ] CORS configured correctly
- [ ] API keys not exposed in frontend
- [ ] HTTPS enabled (automatic on Railway/Vercel)
- [ ] Rate limiting enabled (optional)

---

## 💰 Estimated Costs

### Railway
- **Hobby Plan**: $5/month (500 hours)
- **Pro Plan**: $20/month (unlimited)
- Includes: Postgres, Redis if needed

### Vercel
- **Hobby**: Free (personal projects)
- **Pro**: $20/month (commercial)
- Includes: Edge functions, analytics

### LLM APIs
- **Anthropic Claude**: Pay per token
- **OpenAI GPT-4**: Pay per token
- Estimated: $10-50/month depending on usage

**Total: ~$15-45/month** for production deployment

---

## 🎉 You're Live!

After deployment:

1. ✅ Visit your frontend URL
2. ✅ Select "Game Studio"
3. ✅ Generate a game
4. ✅ Share with the world!

**Demo URL Structure:**
- Frontend: `https://virtualforge.vercel.app`
- Backend: `https://virtualforge.railway.app`
- Game Generator: `https://virtualforge.vercel.app/virtualforge`

---

## 📞 Support

- Railway: https://railway.app/help
- Vercel: https://vercel.com/support
- VirtualForge Issues: GitHub

**Happy Deploying!** 🚀
