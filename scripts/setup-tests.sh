#!/bin/bash

# SimGen AI - Test Environment Setup Script
# This script sets up the complete testing environment for the project

set -e  # Exit on error

echo "🧪 SimGen AI Test Environment Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "README.md" ] || [ ! -d "simgen" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo "📦 Installing Backend Test Dependencies..."
echo "----------------------------------------"
cd simgen/backend

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate || . .venv/Scripts/activate

# Upgrade pip
pip install --upgrade pip

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout pytest-watch

# Install main dependencies
pip install -r requirements.txt

echo -e "${GREEN}✓ Backend test dependencies installed${NC}"
echo ""

# Return to project root
cd ../..

echo "📦 Installing Frontend Test Dependencies..."
echo "-----------------------------------------"
cd frontend

# Install npm packages including test dependencies
npm install

# Install additional test packages if not in package.json
npm install --save-dev \
    @testing-library/jest-dom \
    @testing-library/react \
    @testing-library/user-event \
    @types/jest \
    jest \
    jest-environment-jsdom \
    jest-watch-typeahead \
    identity-obj-proxy

echo -e "${GREEN}✓ Frontend test dependencies installed${NC}"
echo ""

# Return to project root
cd ..

echo "📦 Installing E2E Test Dependencies..."
echo "-------------------------------------"

# Install Playwright
npm install --save-dev @playwright/test

# Install Playwright browsers
npx playwright install

echo -e "${GREEN}✓ E2E test dependencies installed${NC}"
echo ""

echo "🔧 Creating Test Configuration Files..."
echo "--------------------------------------"

# Create .env.test for backend if it doesn't exist
if [ ! -f "simgen/backend/config/.env.test" ]; then
    cat > simgen/backend/config/.env.test <<EOF
# Test Environment Variables
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/simgen_test
REDIS_URL=redis://localhost:6379/1
ANTHROPIC_API_KEY=test_anthropic_key
OPENAI_API_KEY=test_openai_key
SECRET_KEY=test_secret_key_for_testing_only
DEBUG=True
TESTING=True
EOF
    echo -e "${GREEN}✓ Created backend test environment file${NC}"
fi

echo ""
echo "📊 Setting Up Coverage Directories..."
echo "------------------------------------"

# Create coverage directories
mkdir -p simgen/backend/htmlcov
mkdir -p simgen/backend/test-results
mkdir -p frontend/coverage
mkdir -p playwright-report
mkdir -p test-results

echo -e "${GREEN}✓ Coverage directories created${NC}"
echo ""

echo "🚀 Creating Test Runner Scripts..."
echo "---------------------------------"

# Create backend test runner
cat > run-backend-tests.sh <<'EOF'
#!/bin/bash
cd simgen/backend
source .venv/bin/activate || . .venv/Scripts/activate
echo "Running backend tests..."
pytest tests/ -v --cov=simgen --cov-report=html --cov-report=term
echo "Coverage report available at: simgen/backend/htmlcov/index.html"
EOF

# Create frontend test runner
cat > run-frontend-tests.sh <<'EOF'
#!/bin/bash
cd frontend
echo "Running frontend tests..."
npm test -- --coverage --watchAll=false
echo "Coverage report available at: frontend/coverage/lcov-report/index.html"
EOF

# Create E2E test runner
cat > run-e2e-tests.sh <<'EOF'
#!/bin/bash
echo "Running E2E tests..."
npx playwright test
echo "Test report available at: playwright-report/index.html"
EOF

# Create all tests runner
cat > run-all-tests.sh <<'EOF'
#!/bin/bash
echo "🧪 Running All Tests..."
echo "====================="
echo ""

# Run backend tests
echo "📦 Backend Tests"
echo "---------------"
./run-backend-tests.sh
echo ""

# Run frontend tests
echo "📦 Frontend Tests"
echo "----------------"
./run-frontend-tests.sh
echo ""

# Run E2E tests
echo "📦 E2E Tests"
echo "-----------"
./run-e2e-tests.sh
echo ""

echo "✅ All tests completed!"
echo ""
echo "📊 Coverage Reports:"
echo "- Backend: simgen/backend/htmlcov/index.html"
echo "- Frontend: frontend/coverage/lcov-report/index.html"
echo "- E2E: playwright-report/index.html"
EOF

# Make scripts executable
chmod +x run-backend-tests.sh
chmod +x run-frontend-tests.sh
chmod +x run-e2e-tests.sh
chmod +x run-all-tests.sh

echo -e "${GREEN}✓ Test runner scripts created${NC}"
echo ""

echo "=================================="
echo -e "${GREEN}✅ Test Environment Setup Complete!${NC}"
echo "=================================="
echo ""
echo "📚 Quick Start Guide:"
echo "-------------------"
echo ""
echo "Run all tests:"
echo "  ${YELLOW}./run-all-tests.sh${NC}"
echo ""
echo "Run specific test suites:"
echo "  ${YELLOW}./run-backend-tests.sh${NC}  - Backend unit & integration tests"
echo "  ${YELLOW}./run-frontend-tests.sh${NC} - Frontend component tests"
echo "  ${YELLOW}./run-e2e-tests.sh${NC}      - End-to-end tests"
echo ""
echo "Run tests in watch mode:"
echo "  Backend:  ${YELLOW}cd simgen/backend && pytest-watch${NC}"
echo "  Frontend: ${YELLOW}cd frontend && npm test${NC}"
echo ""
echo "View coverage reports:"
echo "  Backend:  ${YELLOW}open simgen/backend/htmlcov/index.html${NC}"
echo "  Frontend: ${YELLOW}open frontend/coverage/lcov-report/index.html${NC}"
echo ""
echo "Happy Testing! 🚀"