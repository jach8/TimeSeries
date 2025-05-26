@echo off
echo Setting up TimeSeriesTools package...

REM Get script directory
set SCRIPT_DIR=%~dp0
cd %SCRIPT_DIR%

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install package dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create test data directory if it doesn't exist
echo Setting up test data...
if not exist timeseriestools\test_data mkdir timeseriestools\test_data

REM Generate test data
echo Generating test data...
python generate_data.py

REM Install package in editable mode
echo Installing package...
pip install -e .

REM Verify installation
echo Verifying installation...
python install_test.py

REM Print success message
echo.
echo Setup complete! ✨
echo.
echo You can now:
echo 1. Run the package tests:
echo    pytest tests/
echo.
echo 2. Try the examples:
echo    cd examples
echo    python package_test.py
echo.
echo 3. Check the documentation in README.md
echo.

REM Keep window open
echo Press any key to exit...
pause > nul

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat