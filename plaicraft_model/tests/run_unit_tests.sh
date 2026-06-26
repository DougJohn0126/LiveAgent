#!/bin/bash
# Run all unit tests with useful output and reporting

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Unit Tests${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Get the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default pytest arguments
PYTEST_ARGS="-v"
PYTEST_ARGS="$PYTEST_ARGS --tb=short"
PYTEST_ARGS="$PYTEST_ARGS tests/unit"

# Add coverage if available
if command -v pytest-cov &> /dev/null; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=term-missing"
    echo -e "${YELLOW}Running with coverage report${NC}\n"
fi

# Run the tests
echo -e "${BLUE}Command: pytest $PYTEST_ARGS${NC}\n"
pytest $PYTEST_ARGS

PYTEST_EXIT=$?

echo -e "\n${BLUE}========================================${NC}"
if [ $PYTEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ All unit tests passed!${NC}"
else
    echo -e "${YELLOW}✗ Some tests failed (exit code: $PYTEST_EXIT)${NC}"
fi
echo -e "${BLUE}========================================${NC}"

exit $PYTEST_EXIT
