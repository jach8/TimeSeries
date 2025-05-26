#!/bin/bash

# Exit on any error
set -e

echo "Installing TimeSeriesTools package..."

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "Setting up package structure..."
mkdir -p timeseriestools/test_data

# Generate test data
echo "Generating test data..."
python generate_data.py

# Install package
echo "Installing package..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python verify_install.py

if [ $? -eq 0 ]; then
    echo "Running minimal test..."
    python minimal_test.py
fi

# Print completion message
if [ $? -eq 0 ]; then
    echo """
Installation successful! ✨

You can now:
1. Import the package:
   import timeseriestools as ts

2. Run the test suite:
   pytest tests/

3. Try the examples in examples/

For help, see README.md
"""
else
    echo "Installation failed. Check the error messages above."
fi

# Deactivate virtual environment
deactivate