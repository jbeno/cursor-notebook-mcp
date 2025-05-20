# PowerShell script to run pytest with required environment variables set

# Set environment variables
$env:JUPYTER_PLATFORM_DIRS = 1
$env:PYTHONASYNCIODBG = 0
$env:PYTHONASYNCIODEBUG = 0
$env:ASYNCIO_TASK_TIMEOUT_SEC = 5

# Check if pytest-timeout is installed
$timeoutInstalled = python -c "import importlib.util; print(importlib.util.find_spec('pytest_timeout') is not None)" 2>$null

# Function to run tests
function Run-Tests {
    param (
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )
    
    if ($timeoutInstalled -eq "True") {
        # Run with a 30-second timeout per test if pytest-timeout is installed
        python -m pytest $Arguments --timeout=30 -v
    }
    else {
        # Run without timeout if pytest-timeout is not installed
        python -m pytest $Arguments -v
    }
}

# Main execution
if ($args.Count -eq 0) {
    # If no arguments provided, run tests in small batches to prevent hanging
    Write-Host "Running tests in batches to prevent hanging..."
    
    # Create a directory for coverage data
    New-Item -ItemType Directory -Force -Path ".coverage_data" | Out-Null
    
    # Remove any existing coverage data
    Remove-Item -Path ".coverage" -ErrorAction SilentlyContinue
    Remove-Item -Path ".coverage.*" -ErrorAction SilentlyContinue
    
    # Run each test file separately with coverage
    Get-ChildItem -Path "tests/test_*.py" | ForEach-Object {
        Write-Host "Running tests in $($_.Name)"
        try {
            if ($timeoutInstalled -eq "True") {
                python -m pytest $_.FullName --timeout=30 -v --cov=cursor_notebook_mcp --cov-append
            }
            else {
                python -m pytest $_.FullName -v --cov=cursor_notebook_mcp --cov-append
            }
        }
        catch {
            Write-Host "Warning: Some tests in $($_.Name) failed or timed out"
        }
    }
    
    Write-Host "All test batches completed."
    
    # Generate a combined coverage report
    Write-Host "Generating combined coverage report..."
    python -m coverage report
    python -m coverage html
    
    Write-Host "HTML coverage report generated in htmlcov/index.html"
}
else {
    # Run pytest with the specified arguments
    Run-Tests $args
} 