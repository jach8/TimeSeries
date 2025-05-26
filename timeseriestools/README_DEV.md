# TimeSeriesTools Development Guide

This guide explains how to set up your development environment and contribute to the TimeSeriesTools package.

## Setting Up Development Environment

### Unix/Linux/Mac
```bash
# Clone the repository
git clone https://github.com/username/timeseriestools.git
cd timeseriestools

# Make setup script executable and run it
chmod +x setup_dev.sh
./setup_dev.sh
```

### Windows
```batch
# Clone the repository
git clone https://github.com/username/timeseriestools.git
cd timeseriestools

# Run setup script
setup_dev.bat
```

## Project Structure
```
timeseriestools/
├── timeseriestools/        # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── analyze.py         # High-level interface
│   ├── causality.py       # Causality analysis
│   ├── correlation.py     # Correlation analysis
│   ├── data.py           # Data loading utilities
│   ├── stationarity.py   # Stationarity tests
│   └── utils/            # Utility functions
├── tests/                 # Test directory
├── examples/              # Example notebooks
└── docs/                 # Documentation
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_analyze.py

# Run with coverage report
pytest --cov=timeseriestools tests/
```

## Code Style Guidelines

1. Type Hints:
   - Use type hints for all function parameters and return values
   - Use typing module for complex types
   ```python
   from typing import Dict, List, Optional
   
   def process_data(data: pd.DataFrame,
                   columns: Optional[List[str]] = None) -> Dict[str, float]:
       ...
   ```

2. Docstrings:
   - Use Google-style docstrings
   - Include examples in docstrings
   ```python
   def function_name(arg1: type1, arg2: type2) -> return_type:
       """Short description.
       
       Detailed description.
       
       Args:
           arg1: Description of arg1
           arg2: Description of arg2
           
       Returns:
           Description of return value
           
       Example:
           >>> function_name(1, "test")
           expected_output
       """
   ```

3. Code Formatting:
   - Use black for code formatting
   - Maximum line length: 88 characters
   - Use isort for import sorting

## Running Code Quality Checks

```bash
# Format code
black timeseriestools/

# Sort imports
isort timeseriestools/

# Run type checker
mypy timeseriestools/

# Run linter
flake8 timeseriestools/
```

## Adding New Features

1. Create a new branch
```bash
git checkout -b feature/your-feature-name
```

2. Write tests first
```bash
# Create test file
touch tests/test_your_feature.py
```

3. Implement feature
```bash
# Create feature file
touch timeseriestools/your_feature.py
```

4. Update documentation
- Add docstrings
- Update README.md if needed
- Add example notebook if applicable

5. Submit pull request
- Ensure all tests pass
- Update CHANGELOG.md
- Fill out PR template

## Debugging Tips

1. Use logging for debugging:
```python
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

2. Use pytest's built-in debugger:
```bash
pytest --pdb tests/test_file.py
```

3. Use VSCode's debugger with the provided launch configurations

## Common Issues

1. Import errors:
   - Ensure you're in the virtual environment
   - Check PYTHONPATH includes project root
   
2. Test data missing:
   - Run setup script to create test_data directory
   - Download test data if needed

3. Type checking errors:
   - Check type hints
   - Use cast() for complex type situations

## Getting Help

- Check existing issues
- Read the documentation
- Ask in discussions
- Contact maintainers

## Release Process

1. Update version in:
   - setup.py
   - __init__.py
   - CHANGELOG.md

2. Run tests and checks
```bash
pytest tests/
black timeseriestools/
isort timeseriestools/
mypy timeseriestools/
flake8 timeseriestools/
```

3. Create release branch
```bash
git checkout -b release/v0.1.0
```

4. Tag release
```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
```

5. Build and publish
```bash
python -m build
twine upload dist/*