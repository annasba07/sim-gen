#!/bin/bash

# VirtualForge Games API Test Script
# Tests all game generation endpoints

set -e

# Check if API_URL is provided
if [ -z "$1" ]; then
    echo "Usage: ./test_games_api.sh <API_URL>"
    echo "Example: ./test_games_api.sh https://your-service.onrender.com"
    exit 1
fi

API_URL="$1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🧪 Testing VirtualForge Games API"
echo "=================================="
echo "API URL: $API_URL"
echo ""

# Create output directory
mkdir -p test_games

# Test 1: Health check
echo -e "${YELLOW}Test 1: Health Check${NC}"
if curl -s -f "$API_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: List templates
echo -e "${YELLOW}Test 2: List Templates${NC}"
TEMPLATES=$(curl -s "$API_URL/api/v2/games/templates")
TEMPLATE_COUNT=$(echo "$TEMPLATES" | jq '. | length')
echo "Found $TEMPLATE_COUNT templates"
echo "$TEMPLATES" | jq -r '.[] | "  - \(.id): \(.name)"'
echo -e "${GREEN}✅ Templates loaded${NC}"
echo ""

# Test 3: Get specific template
echo -e "${YELLOW}Test 3: Get Template (coin-collector)${NC}"
TEMPLATE=$(curl -s "$API_URL/api/v2/games/templates/coin-collector")
TEMPLATE_NAME=$(echo "$TEMPLATE" | jq -r '.spec.title')
echo "Template: $TEMPLATE_NAME"
echo -e "${GREEN}✅ Template retrieved${NC}"
echo ""

# Test 4: Generate platformer game
echo -e "${YELLOW}Test 4: Generate Platformer Game${NC}"
echo "Prompt: 'A game where you collect stars on platforms'"
GAME_RESPONSE=$(curl -s -X POST "$API_URL/api/v2/games/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A game where you collect stars on platforms",
    "gameType": "platformer",
    "complexity": "simple"
  }')

SUCCESS=$(echo "$GAME_RESPONSE" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
    echo "$GAME_RESPONSE" | jq -r '.html' > test_games/platformer.html
    echo -e "${GREEN}✅ Platformer generated → test_games/platformer.html${NC}"
else
    echo -e "${RED}❌ Generation failed${NC}"
    echo "$GAME_RESPONSE" | jq '.errors'
fi
echo ""

# Test 5: Generate top-down game
echo -e "${YELLOW}Test 5: Generate Top-Down Game${NC}"
echo "Prompt: 'Explore a dungeon and collect treasures'"
GAME_RESPONSE=$(curl -s -X POST "$API_URL/api/v2/games/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explore a dungeon and collect treasures",
    "gameType": "topdown",
    "complexity": "simple"
  }')

SUCCESS=$(echo "$GAME_RESPONSE" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
    echo "$GAME_RESPONSE" | jq -r '.html' > test_games/topdown.html
    echo -e "${GREEN}✅ Top-down game generated → test_games/topdown.html${NC}"
else
    echo -e "${RED}❌ Generation failed${NC}"
    echo "$GAME_RESPONSE" | jq '.errors'
fi
echo ""

# Test 6: Generate shooter game
echo -e "${YELLOW}Test 6: Generate Shooter Game${NC}"
echo "Prompt: 'Shoot asteroids in space'"
GAME_RESPONSE=$(curl -s -X POST "$API_URL/api/v2/games/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Shoot asteroids in space",
    "gameType": "shooter",
    "complexity": "simple"
  }')

SUCCESS=$(echo "$GAME_RESPONSE" | jq -r '.success')
if [ "$SUCCESS" = "true" ]; then
    echo "$GAME_RESPONSE" | jq -r '.html' > test_games/shooter.html
    echo -e "${GREEN}✅ Shooter generated → test_games/shooter.html${NC}"
else
    echo -e "${RED}❌ Generation failed${NC}"
    echo "$GAME_RESPONSE" | jq '.errors'
fi
echo ""

echo "=================================="
echo -e "${GREEN}🎉 All tests complete!${NC}"
echo ""
echo "Generated games saved to: test_games/"
echo "Open them in your browser to play:"
echo "  open test_games/platformer.html"
echo "  open test_games/topdown.html"
echo "  open test_games/shooter.html"
