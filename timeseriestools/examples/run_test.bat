@echo off
REM Ensure we're in the examples directory
cd %~dp0
cd ..

echo Setting up test environment...

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install package in editable mode
echo Installing package in editable mode...
pip install -e .

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create test data directory if it doesn't exist
if not exist test_data mkdir test_data

REM Run the test script
echo.
echo Running package test...
echo.
python examples/package_test.py

REM Keep window open to see results
echo.
echo Press any key to close...
pause > nul

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat