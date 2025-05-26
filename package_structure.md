# TimeSeriesTools Package Documentation

## Overview

TimeSeriesTools is a Python package for time series analysis, focusing on causality testing, correlation analysis, and stationarity checks.

## Package Structure

```
timeseriestools/
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
├── timeseriestools/
│   ├── __init__.py
│   ├── analyze.py
│   ├── causality.py
│   ├── correlation.py
│   ├── stationarity.py
│   ├── data.py
│   └── utils/
│       ├── __init__.py
│       └── granger.py
├── tests/
│   ├── __init__.py
│   ├── test_causality.py
│   ├── test_correlation.py
│   └── test_stationarity.py
├── examples/
│   └── notebooks/
└── docs/
    ├── api/
    └── guides/
```

## Module Documentation

### analyze.py

```python
class Analyze:
    """Main analysis class for time series correlation and causality testing.
    
    This class provides a high-level interface for performing time series analysis,
    including correlation analysis, causality testing, and stationarity checks.
    
    Args:
        verbose (bool, optional): Whether to print detailed output. Defaults to False.
        stationarity_config (Dict[str, Any], optional): Configuration for stationarity tests.
            Example:
            {
                'adf': {'max_diff': 5, 'significance': 0.05},
                'kpss': {'significance': 0.05},
                'pp': {'significance': 0.05},
                'structural_break': True,
                'gls': True,
                'nonlinear': True
            }
        causality_config (Dict[str, Any], optional): Configuration for causality tests.
            Example:
            {
                'significance_level': 0.05,
                'max_lag': 3
            }
            
    Attributes:
        verbose (bool): Verbosity flag
        stationary_config (Dict): Configuration for stationarity tests
        causality_config (Dict): Configuration for causality tests
        AC (AnalyzeCorrelation): Instance of correlation analyzer
        results (Dict): Analysis results
        
    Example:
        >>> import timeseriestools as ts
        >>> x, y = ts.test_data1(return_xy=True)
        >>> analyzer = ts.Analyze(verbose=True)
        >>> results = analyzer.analyze_correlation(x, y)
    """

    def analyze_correlation(self, x: pd.Series, y: pd.Series, decompose: bool = False) -> Dict[str, Any]:
        """Analyze correlation between two time series.
        
        Performs comprehensive correlation analysis including:
        - Stationarity checks
        - Granger causality tests
        - Impulse response analysis
        
        Args:
            x (pd.Series): First time series
            y (pd.Series): Second time series
            decompose (bool, optional): Whether to use PCA decomposition. Defaults to False.
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - stationarity_report: Results of stationarity tests
                - var_model: Vector autoregression model
                - causality: Causality test results
                - new_data: Processed time series data
                
        Example:
            >>> x = pd.Series([1, 2, 3, 4])
            >>> y = pd.Series([2, 4, 6, 8])
            >>> analyzer = Analyze()
            >>> results = analyzer.analyze_correlation(x, y)
        """
```

### Installation

```bash
pip install timeseriestools
```

### Basic Usage

```python
import timeseriestools as ts

# Load test data
x, y = ts.test_data1(return_xy=True)

# Initialize analyzer
analyzer = ts.Analyze(verbose=True)

# Run analysis
results = analyzer.analyze_correlation(x, y)

# Access results
print(results['stationarity_report'])
print(results['causality'])
```

## Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/`

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.