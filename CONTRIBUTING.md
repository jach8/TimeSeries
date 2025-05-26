# Contributing to TimeSeriesTools

Thank you for your interest in contributing to TimeSeriesTools! This document provides guidelines and instructions for contributing.

## Code Style

We follow these coding standards:

1. **Google Style Python Docstrings**
   ```python
   def function_name(arg1: type1, arg2: type2) -> return_type:
       """Short description of function.
       
       Detailed description of function behavior.
       
       Args:
           arg1 (type1): Description of arg1
           arg2 (type2): Description of arg2
           
       Returns:
           return_type: Description of return value
           
       Raises:
           ExceptionType: When and why this exception occurs
           
       Example:
           >>> function_name(1, "test")
           expected_output
       """
   ```

2. **Type Hints**
   - Use type hints for all function parameters and return values
   - Use typing module for complex types
   - Example:
     ```python
     from typing import Dict, List, Optional
     
     def process_data(data: pd.DataFrame,
                     columns: Optional[List[str]] = None) -> Dict[str, float]:
     ```

3. **Code Formatting**
   - Use black for code formatting
   - Maximum line length: 88 characters
   - Use isort for import sorting

## Development Process

1. **Setting Up Development Environment**
   ```bash
   # Clone repository
   git clone https://github.com/username/timeseriestools.git
   cd timeseriestools
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

2. **Running Tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/test_causality.py
   
   # Run with coverage
   pytest --cov=timeseriestools tests/
   ```

3. **Code Quality Checks**
   ```bash
   # Run black
   black timeseriestools/
   
   # Run isort
   isort timeseriestools/
   
   # Run mypy
   mypy timeseriestools/
   
   # Run flake8
   flake8 timeseriestools/
   ```

## Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write tests for new functionality
   - Update documentation
   - Follow code style guidelines
   - Keep commits focused and atomic

3. **Submit Pull Request**
   - Update CHANGELOG.md
   - Fill out PR template
   - Reference any related issues
   - Wait for CI checks to pass

4. **Code Review**
   - Address reviewer comments
   - Update PR as needed
   - Maintain a civil and professional discourse

## Documentation

When contributing new features, please include:

1. **Docstrings** for all public functions/methods
2. **Type hints** for parameters and return values
3. **Examples** showing usage
4. **Tests** demonstrating functionality
5. **API documentation** updates if needed

## Testing Guidelines

1. **Unit Tests**
   - Write tests for all new functionality
   - Use pytest fixtures for test setup
   - Mock external dependencies
   - Aim for high test coverage

2. **Integration Tests**
   - Test interactions between components
   - Use real data examples
   - Test edge cases and error conditions

## Getting Help

- Create an issue for bug reports or feature requests
- Join our discussions for questions
- Read our documentation thoroughly
- Check existing issues and PRs before creating new ones

Thank you for contributing to TimeSeriesTools!