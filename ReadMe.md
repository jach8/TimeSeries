# TimeSeriesTools

A comprehensive Python package for time series analysis, focusing on causality testing, correlation analysis, and stationarity checks.

## Features

- **Causality Analysis**: Implement various causality tests including Granger causality and instantaneous causality
- **Correlation Analysis**: Perform correlation analysis with built-in stationarity checks
- **Stationarity Testing**: Multiple stationarity tests including ADF, KPSS, Phillips-Perron, and more
- **Visualization**: Generate insightful visualizations of time series relationships
- **Type Safety**: Full type hint coverage for better code reliability
- **Documentation**: Comprehensive API documentation with examples

## Installation

```bash
pip install timeseriestools
```

## Quick Start

```python
import timeseriestools as ts
import pandas as pd

# Load example data
x, y = ts.test_data1(return_xy=True)

# Initialize analyzer
analyzer = ts.Analyze(verbose=True)

# Run correlation analysis
results = analyzer.analyze_correlation(x, y)

# Access results
print("\nStationarity Report:")
print(results['stationarity_report'])

print("\nCausality Results:")
print(results['causality'])
```

## Advanced Usage

### Custom Configuration

```python
# Configure stationarity tests
stationarity_config = {
    'adf': {'max_diff': 5, 'significance': 0.05},
    'kpss': {'significance': 0.05},
    'pp': {'significance': 0.05},
    'structural_break': True,
    'gls': True,
    'nonlinear': True
}

# Configure causality tests
causality_config = {
    'significance_level': 0.05,
    'max_lag': 3
}

# Initialize with custom config
analyzer = ts.Analyze(
    verbose=True,
    stationarity_config=stationarity_config,
    causality_config=causality_config
)
```

### Working with Custom Data

```python
# Load your own time series data
data = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')

# Split into features and target
x = data.drop('target', axis=1)
y = data['target']

# Run analysis
analyzer = ts.Analyze()
results = analyzer.analyze_correlation(x, y, decompose=True)
```

## Documentation

For detailed API documentation and examples, see:
- [Package Structure](package_structure.md)
- [API Documentation](api_documentation.md)
- [Example Notebooks](examples/notebooks/)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/username/timeseriestools.git
cd timeseriestools
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run tests:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TimeSeriesTools in your research, please cite:

```bibtex
@software{timeseriestools2025,
  title = {TimeSeriesTools: A Python Package for Time Series Analysis},
  author = {Authors},
  year = {2025},
  version = {0.1.0}
}
```

## Acknowledgments

- Built with inspiration from statsmodels and pandas
- Includes implementations based on academic papers in time series analysis
- Thanks to all contributors and the open source community
