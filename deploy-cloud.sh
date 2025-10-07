#!/bin/bash

# VirtualForge Cloud Deployment Script
# Deploys backend to Railway and frontend to Vercel

set -e  # Exit on error

echo "🚀 VirtualForge Cloud Deployment"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found${NC}"
    echo "Install with: brew install railway"
    echo "Or: npm install -g @railway/cli"
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found${NC}"
    echo "Install with: npm install -g vercel"
    exit 1
fi

echo ""
echo -e "${YELLOW}📦 Step 1: Deploy Backend to Railway${NC}"
echo "========================================"

# Check if Railway is authenticated
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway:"
    railway login
fi

# Check if API keys are set
echo ""
echo "Checking environment variables..."
if ! railway variables | grep -q "ANTHROPIC_API_KEY"; then
    echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not set${NC}"
    read -p "Enter Anthropic API key (or press Enter to skip): " ANTHROPIC_KEY
    if [ -n "$ANTHROPIC_KEY" ]; then
        railway variables set ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
    fi
fi

if ! railway variables | grep -q "OPENAI_API_KEY"; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set${NC}"
    read -p "Enter OpenAI API key (or press Enter to skip): " OPENAI_KEY
    if [ -n "$OPENAI_KEY" ]; then
        railway variables set OPENAI_API_KEY="$OPENAI_KEY"
    fi
fi

# Deploy backend
echo ""
echo "Deploying backend..."
railway up

# Get backend URL
echo ""
echo "Getting backend URL..."
BACKEND_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)

if [ -z "$BACKEND_URL" ]; then
    echo -e "${YELLOW}⚠️  No domain found. Creating one...${NC}"
    railway domain
    BACKEND_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)
fi

echo -e "${GREEN}✅ Backend deployed: $BACKEND_URL${NC}"

# Wait for backend to be ready
echo ""
echo "Waiting for backend to be ready..."
sleep 10

# Test backend health
echo "Testing backend health..."
if curl -s -f "${BACKEND_URL}/health" > /dev/null; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Backend health check failed (might be normal for first deploy)${NC}"
fi

echo ""
echo -e "${YELLOW}🎨 Step 2: Deploy Frontend to Vercel${NC}"
echo "========================================"

cd frontend

# Check if Vercel is authenticated
if ! vercel whoami &> /dev/null; then
    echo "Please login to Vercel:"
    vercel login
fi

# Set environment variable for production
echo ""
echo "Setting NEXT_PUBLIC_API_URL to: $BACKEND_URL"
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_API_URL production

# Deploy to production
echo ""
echo "Deploying frontend..."
FRONTEND_URL=$(vercel --prod 2>&1 | grep -o 'https://[^[:space:]]*vercel.app' | head -1)

cd ..

echo ""
echo -e "${GREEN}✅ Frontend deployed: $FRONTEND_URL${NC}"

# Update CORS on backend
echo ""
echo -e "${YELLOW}📝 Step 3: Configure CORS${NC}"
echo "========================="
echo "Setting CORS_ORIGINS to: $FRONTEND_URL"
railway variables set CORS_ORIGINS="$FRONTEND_URL"

echo ""
echo "Redeploying backend with CORS config..."
railway up

echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "======================="
echo ""
echo -e "${GREEN}Backend:${NC}  $BACKEND_URL"
echo -e "${GREEN}Frontend:${NC} $FRONTEND_URL"
echo ""
echo "Next steps:"
echo "1. Visit: $FRONTEND_URL/virtualforge"
echo "2. Select 'Game Studio'"
echo "3. Try generating a game!"
echo ""
echo "Useful commands:"
echo "  railway logs -f    # View backend logs"
echo "  vercel logs        # View frontend logs"
echo "  railway restart    # Restart backend"
echo ""
