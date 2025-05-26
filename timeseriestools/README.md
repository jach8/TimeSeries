# TimeSeriesTools

A comprehensive Python package for time series analysis, featuring stationarity testing, causality analysis, and correlation analysis.

## Features

- Stationarity analysis with multiple tests
- Granger causality testing
- Correlation analysis with automatic stationarity checks
- PCA decomposition for feature reduction
- Built-in test data and examples

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/timeseriestools.git
cd timeseriestools

# Install with setup script
# On Unix/Linux/Mac:
./install.sh

# On Windows:
install.bat
```

### Basic Usage

```python
import timeseriestools as ts

# Generate sample data
X, y = ts.random_test_data(n=100, return_xy=True)

# Create analyzer
analyzer = ts.Analyze(verbose=True)

# Run analysis
results = analyzer.analyze_correlation(X, y)

# Check results
print("Stationarity Report:")
print(results['stationarity_report'])

print("\nCausality Results:")
for rel in results['causality']['granger']:
    print(f"{rel[0][1]} Granger causes {rel[0][0]} at lags {rel[1]}")
```

## Installation Options

### 1. Using Setup Scripts (Recommended)

The setup scripts will:
- Create a virtual environment
- Install dependencies
- Generate test data
- Install the package
- Run verification tests

```bash
# Unix/Linux/Mac
./install.sh

# Windows
install.bat
```

### 2. Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# Install package and dependencies
pip install -r requirements.txt
python generate_data.py
pip install -e .

# Verify installation
python verify_install.py
```

### 3. Development Installation

For development work, install with additional tools:

```bash
pip install -e ".[dev]"
```

## Dependencies

Core dependencies:
- Python >= 3.8
- pandas >= 1.0.0
- numpy >= 1.18.0
- statsmodels >= 0.13.0
- scikit-learn >= 0.24.0

## Documentation

- [Development Guide](README_DEV.md)
- [Examples](examples/)
- [API Documentation](api_documentation.md)
- [Contributing](CONTRIBUTING.md)

## Examples

Basic examples are in the `examples` directory:
```bash
cd examples
python package_test.py
```

For more detailed examples, check out the Jupyter notebooks in `examples/notebooks/`.

## Testing

Run the test suite:
```bash
pytest tests/
```

Quick functionality test:
```bash
python minimal_test.py
```

## Common Issues

### Import Errors
```
ModuleNotFoundError: No module named 'timeseriestools'
```
Solution: Make sure you're in the package root directory and install in editable mode:
```bash
pip install -e .
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'pandas'
```
Solution: Install all dependencies:
```bash
pip install -r requirements.txt
```

### Test Data Not Found
```
FileNotFoundError: test_data/data.pkl not found
```
Solution: Generate test data:
```bash
python generate_data.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.