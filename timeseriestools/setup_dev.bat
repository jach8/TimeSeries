@echo off
REM Setup development environment for Windows

REM Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install development dependencies
pip install -r requirements-dev.txt

REM Install package in editable mode
pip install -e .

REM Create test data directory if it doesn't exist
if not exist test_data mkdir test_data

REM Run tests
pytest tests/ -v

REM Keep window open to see results
echo.
echo Setup complete. Press any key to close...
pause > nul