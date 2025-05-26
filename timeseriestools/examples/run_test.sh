#!/bin/bash

# Ensure we're in the examples directory
cd "$(dirname "$0")"

# Go up one level to package root
cd ..

echo "Setting up test environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the test script
echo -e "\nRunning package test...\n"
python examples/package_test.py

# Deactivate virtual environment
deactivate