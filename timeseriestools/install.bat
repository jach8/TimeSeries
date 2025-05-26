@echo off
echo Installing TimeSeriesTools package...

REM Get script directory
set SCRIPT_DIR=%~dp0
cd %SCRIPT_DIR%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo Setting up package structure...
if not exist timeseriestools\test_data mkdir timeseriestools\test_data

REM Generate test data
echo Generating test data...
python generate_data.py

REM Install package
echo Installing package...
pip install -e .

REM Verify installation
echo Verifying installation...
python verify_install.py

if %ERRORLEVEL% EQU 0 (
    echo Running minimal test...
    python minimal_test.py
)

REM Print completion message
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation successful! ✨
    echo.
    echo You can now:
    echo 1. Import the package:
    echo    import timeseriestools as ts
    echo.
    echo 2. Run the test suite:
    echo    pytest tests/
    echo.
    echo 3. Try the examples in examples/
    echo.
    echo For help, see README.md
) else (
    echo Installation failed. Check the error messages above.
)

REM Keep window open to see results
echo.
echo Press any key to exit...
pause > nul

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat