#!/bin/bash
# Wrapper script to run pytest with required environment variables set.

# Set the environment variable to silence the jupyter_client warning
export JUPYTER_PLATFORM_DIRS=1

# Disable asyncio debug mode to prevent hanging
export PYTHONASYNCIODBG=0
export PYTHONASYNCIODEBUG=0

# Set a shorter default timeout for asyncio operations
export ASYNCIO_TASK_TIMEOUT_SEC=5

# Check if pytest-timeout is installed
TIMEOUT_INSTALLED=$(python -c "import importlib.util; print(importlib.util.find_spec('pytest_timeout') is not None)" 2>/dev/null)

# Function to run tests
run_tests() {
    if [ "$TIMEOUT_INSTALLED" = "True" ]; then
        # Run with a 30-second timeout per test if pytest-timeout is installed
        python -m pytest "$@" --timeout=30 -v
    else
        # Run without timeout if pytest-timeout is not installed
        python -m pytest "$@" -v
    fi
}

# Run pytest with the appropriate settings
if [ "$#" -eq 0 ]; then
    # If no arguments provided, run tests in small batches to prevent hanging
    echo "Running tests in batches to prevent hanging..."
    
    # Create a directory for coverage data
    mkdir -p .coverage_data
    
    # Remove any existing coverage data
    rm -f .coverage .coverage.*
    
    # Run each test file separately with coverage
    for test_file in tests/test_*.py; do
        echo "Running tests in $test_file"
        if [ "$TIMEOUT_INSTALLED" = "True" ]; then
            python -m pytest "$test_file" --timeout=30 -v --cov=cursor_notebook_mcp --cov-append || echo "Warning: Some tests in $test_file failed or timed out"
        else
            python -m pytest "$test_file" -v --cov=cursor_notebook_mcp --cov-append || echo "Warning: Some tests in $test_file failed or timed out"
        fi
    done
    
    echo "All test batches completed."
    
    # Generate a combined coverage report
    echo "Generating combined coverage report..."
    python -m coverage report
    python -m coverage html
    
    echo "HTML coverage report generated in htmlcov/index.html"
else
    # Run pytest with the specified arguments
    run_tests "$@"
fi 