#!/bin/bash

# Exit on any error
set -e

echo "Setting up TimeSeriesTools package..."

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install package dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create test data directory if it doesn't exist
echo "Setting up test data..."
mkdir -p timeseriestools/test_data

# Generate test data
echo "Generating test data..."
python generate_data.py

# Install package in editable mode
echo "Installing package..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python install_test.py

# Print success message
echo "
Setup complete! ✨

You can now:
1. Run the package tests:
   pytest tests/

2. Try the examples:
   cd examples
   python package_test.py

3. Check the documentation in README.md
"

# Deactivate virtual environment
deactivate