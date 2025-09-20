# SimGen AI - Testing Documentation

## 📊 Test Coverage Overview

Comprehensive test suite covering backend services, frontend components, and end-to-end user flows with **70%+ coverage target**.

## 🧪 Test Structure

```
simulation-mujoco/
├── simgen/backend/tests/
│   ├── unit/                    # Unit tests for services
│   │   ├── test_dynamic_scene_composer.py
│   │   ├── test_prompt_parser.py
│   │   └── test_simulation_generator.py
│   ├── integration/              # API integration tests
│   │   └── test_api_endpoints.py
│   └── fixtures/                 # Test data and mocks
├── frontend/
│   ├── src/__tests__/           # Frontend component tests
│   │   └── components/
│   │       ├── sketch-canvas.test.tsx
│   │       └── simulation-viewer.test.tsx
│   ├── jest.config.js           # Jest configuration
│   └── jest.setup.js            # Test environment setup
├── e2e/                         # End-to-end tests
│   └── sketch-to-simulation.spec.ts
└── playwright.config.ts         # E2E test configuration
```

## 🚀 Running Tests

### Backend Tests

```bash
cd simgen/backend

# Run all tests
pytest

# Run with coverage
pytest --cov=simgen --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration        # Integration tests only
pytest -m "not slow"        # Skip slow tests

# Run in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Watch mode for development
pytest-watch
```

### Frontend Tests

```bash
cd frontend

# Install dependencies first
npm install

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch

# CI mode (no watch, with coverage)
npm run test:ci
```

### E2E Tests

```bash
# Install Playwright
npx playwright install

# Run all E2E tests
npx playwright test

# Run specific test file
npx playwright test e2e/sketch-to-simulation.spec.ts

# Run in headed mode (see browser)
npx playwright test --headed

# Run specific browser
npx playwright test --project=chromium

# Debug mode
npx playwright test --debug

# Generate test report
npx playwright show-report
```

## 📝 Test Categories

### Unit Tests
- **Purpose**: Test individual functions and classes in isolation
- **Coverage**: Services, utilities, parsers, generators
- **Mocking**: External dependencies are mocked
- **Speed**: Fast (<1s per test)

### Integration Tests
- **Purpose**: Test component interactions and API endpoints
- **Coverage**: API routes, database operations, service integration
- **Mocking**: External APIs may be mocked
- **Speed**: Medium (1-5s per test)

### E2E Tests
- **Purpose**: Test complete user workflows
- **Coverage**: Critical user paths from UI to backend
- **Mocking**: Minimal, tests against real services
- **Speed**: Slow (5-30s per test)

## 🎯 Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| Backend Services | 80% | - |
| API Endpoints | 90% | - |
| Frontend Components | 75% | - |
| Critical Paths | 100% | - |
| Overall | 70% | - |

## 🔧 Test Configuration

### Backend (pytest.ini)
```ini
[pytest]
minversion = 7.0
testpaths = tests
addopts =
    -v
    --cov=simgen
    --cov-report=html
    --cov-report=term-missing
    --asyncio-mode=auto
```

### Frontend (jest.config.js)
```javascript
module.exports = {
  testEnvironment: 'jsdom',
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
}
```

### E2E (playwright.config.ts)
```typescript
export default {
  testDir: './e2e',
  fullyParallel: true,
  reporter: [['html'], ['json'], ['junit']],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
}
```

## 🏷️ Test Markers & Tags

### Backend Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests > 5 seconds
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.llm` - Tests requiring LLM APIs
- `@pytest.mark.critical` - Must-pass tests

### Frontend Tags
- `@group:unit` - Component unit tests
- `@group:integration` - Component integration
- `@group:visual` - Visual regression tests
- `@group:a11y` - Accessibility tests

## 📊 Coverage Reports

### Viewing Coverage

**Backend HTML Report:**
```bash
cd simgen/backend
pytest --cov=simgen --cov-report=html
open htmlcov/index.html
```

**Frontend HTML Report:**
```bash
cd frontend
npm run test:coverage
open coverage/lcov-report/index.html
```

### CI/CD Integration

Coverage reports are automatically:
- Generated on every push
- Uploaded to Codecov
- Posted as PR comments
- Used for merge checks

## 🐛 Debugging Tests

### Backend Debugging
```bash
# Run with debugger
pytest --pdb

# Run specific test with print statements
pytest tests/unit/test_simulation_generator.py::test_generate -s

# Increase verbosity
pytest -vvv

# Show local variables on failure
pytest --showlocals
```

### Frontend Debugging
```bash
# Run single test file
npm test -- sketch-canvas.test.tsx

# Debug in VS Code
# Add breakpoint and run: Debug > Jest Current File

# Update snapshots
npm test -- -u
```

### E2E Debugging
```bash
# Debug mode with inspector
npx playwright test --debug

# Headed mode to see browser
npx playwright test --headed

# Slow motion
npx playwright test --headed --slow-mo=1000

# Generate trace
npx playwright test --trace on
```

## 🔄 Continuous Integration

Tests run automatically on:
- Every push to main/master/develop
- Every pull request
- Nightly scheduled runs
- Manual workflow dispatch

### GitHub Actions Workflow
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    # Runs unit + integration tests

  frontend-tests:
    runs-on: ubuntu-latest
    # Runs Jest tests

  e2e-tests:
    runs-on: ubuntu-latest
    # Runs Playwright tests
```

## 📈 Test Metrics

### Key Metrics Tracked
- Line coverage percentage
- Branch coverage percentage
- Test execution time
- Flaky test detection
- Test failure rate

### Performance Benchmarks
- Backend unit tests: < 30 seconds
- Frontend tests: < 1 minute
- E2E tests: < 5 minutes
- Total CI pipeline: < 10 minutes

## 🎭 Mocking Strategies

### Backend Mocks
```python
# LLM API mocking
from unittest.mock import AsyncMock
mock_llm = AsyncMock(return_value="response")

# Database mocking
from unittest.mock import MagicMock
mock_db = MagicMock()
```

### Frontend Mocks
```javascript
// API mocking
jest.mock('@/lib/api', () => ({
  generateSimulation: jest.fn()
}))

// Canvas mocking
HTMLCanvasElement.prototype.getContext = jest.fn()
```

## 🔍 Test Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names describe what is being tested
3. **Speed**: Prefer unit tests over integration tests
4. **Coverage**: Aim for behavior coverage, not line coverage
5. **Maintenance**: Update tests when requirements change
6. **Documentation**: Comment complex test setups

## 🚨 Common Issues & Solutions

### Issue: Tests failing in CI but passing locally
**Solution**: Check environment variables and service dependencies

### Issue: Flaky E2E tests
**Solution**: Add proper waits and increase timeouts

### Issue: Slow test execution
**Solution**: Use parallel execution and mock expensive operations

### Issue: Coverage not updating
**Solution**: Clear cache and regenerate reports

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Jest Documentation](https://jestjs.io/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://testingjavascript.com/)

---

**Test Coverage Status**: 🟢 Active Development

For questions or issues with testing, please open an issue on GitHub.