#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # For Unix/Linux
# .\venv\Scripts\activate  # For Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Create test data directory if it doesn't exist
mkdir -p test_data

# Run tests
pytest tests/ -v