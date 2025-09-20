@echo off
REM SimGen AI - Test Environment Setup Script for Windows
REM This script sets up the complete testing environment for the project

echo ==================================
echo SimGen AI Test Environment Setup
echo ==================================
echo.

REM Check if running from project root
if not exist "README.md" (
    echo Error: Please run this script from the project root directory
    exit /b 1
)

echo Installing Backend Test Dependencies...
echo ----------------------------------------
cd simgen\backend

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout pytest-watch

REM Install main dependencies
pip install -r requirements.txt

echo Backend test dependencies installed
echo.

REM Return to project root
cd ..\..

echo Installing Frontend Test Dependencies...
echo -----------------------------------------
cd frontend

REM Install npm packages including test dependencies
call npm install

REM Install additional test packages
call npm install --save-dev @testing-library/jest-dom @testing-library/react @testing-library/user-event @types/jest jest jest-environment-jsdom jest-watch-typeahead identity-obj-proxy

echo Frontend test dependencies installed
echo.

REM Return to project root
cd ..

echo Installing E2E Test Dependencies...
echo -------------------------------------

REM Install Playwright
call npm install --save-dev @playwright/test

REM Install Playwright browsers
call npx playwright install

echo E2E test dependencies installed
echo.

echo Creating Test Configuration Files...
echo --------------------------------------

REM Create .env.test for backend if it doesn't exist
if not exist "simgen\backend\config\.env.test" (
    (
    echo # Test Environment Variables
    echo DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/simgen_test
    echo REDIS_URL=redis://localhost:6379/1
    echo ANTHROPIC_API_KEY=test_anthropic_key
    echo OPENAI_API_KEY=test_openai_key
    echo SECRET_KEY=test_secret_key_for_testing_only
    echo DEBUG=True
    echo TESTING=True
    ) > simgen\backend\config\.env.test
    echo Created backend test environment file
)

echo.
echo Setting Up Coverage Directories...
echo ------------------------------------

REM Create coverage directories
if not exist "simgen\backend\htmlcov" mkdir simgen\backend\htmlcov
if not exist "simgen\backend\test-results" mkdir simgen\backend\test-results
if not exist "frontend\coverage" mkdir frontend\coverage
if not exist "playwright-report" mkdir playwright-report
if not exist "test-results" mkdir test-results

echo Coverage directories created
echo.

echo Creating Test Runner Scripts...
echo ---------------------------------

REM Create backend test runner
(
echo @echo off
echo cd simgen\backend
echo call .venv\Scripts\activate
echo echo Running backend tests...
echo pytest tests\ -v --cov=simgen --cov-report=html --cov-report=term
echo echo Coverage report available at: simgen\backend\htmlcov\index.html
) > run-backend-tests.bat

REM Create frontend test runner
(
echo @echo off
echo cd frontend
echo echo Running frontend tests...
echo call npm test -- --coverage --watchAll=false
echo echo Coverage report available at: frontend\coverage\lcov-report\index.html
) > run-frontend-tests.bat

REM Create E2E test runner
(
echo @echo off
echo echo Running E2E tests...
echo call npx playwright test
echo echo Test report available at: playwright-report\index.html
) > run-e2e-tests.bat

REM Create all tests runner
(
echo @echo off
echo echo =======================
echo echo Running All Tests...
echo echo =======================
echo echo.
echo.
echo echo Backend Tests
echo echo ---------------
echo call run-backend-tests.bat
echo echo.
echo.
echo echo Frontend Tests
echo echo ----------------
echo call run-frontend-tests.bat
echo echo.
echo.
echo echo E2E Tests
echo echo -----------
echo call run-e2e-tests.bat
echo echo.
echo.
echo echo All tests completed!
echo echo.
echo echo Coverage Reports:
echo echo - Backend: simgen\backend\htmlcov\index.html
echo echo - Frontend: frontend\coverage\lcov-report\index.html
echo echo - E2E: playwright-report\index.html
) > run-all-tests.bat

echo Test runner scripts created
echo.

echo ==================================
echo Test Environment Setup Complete!
echo ==================================
echo.
echo Quick Start Guide:
echo -------------------
echo.
echo Run all tests:
echo   run-all-tests.bat
echo.
echo Run specific test suites:
echo   run-backend-tests.bat  - Backend unit and integration tests
echo   run-frontend-tests.bat - Frontend component tests
echo   run-e2e-tests.bat      - End-to-end tests
echo.
echo Run tests in watch mode:
echo   Backend:  cd simgen\backend ^&^& pytest-watch
echo   Frontend: cd frontend ^&^& npm test
echo.
echo View coverage reports:
echo   Backend:  start simgen\backend\htmlcov\index.html
echo   Frontend: start frontend\coverage\lcov-report\index.html
echo.
echo Happy Testing!

pause