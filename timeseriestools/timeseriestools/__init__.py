"""TimeSeriesTools: A comprehensive Python package for time series analysis."""

from typing import List

# Core functionality
from .analyze import Analyze
from .correlation import AnalyzeCorrelation
from .causality import CausalityAnalyzer
from .stationarity import StationaryTests
from .data import (
    test_data1,
    test_data2,
    test_data3,
    random_test_data
)

# Version information
__version__ = "0.1.0"
__author__ = "TimeSeriesTools Contributors"
__author_email__ = "contributors@timeseriestools.org"
__license__ = "MIT"

# Export public API
__all__: List[str] = [
    # Main classes
    "Analyze",
    "AnalyzeCorrelation",
    "CausalityAnalyzer",
    "StationaryTests",
    
    # Data functions
    "test_data1",
    "test_data2",
    "test_data3",
    "random_test_data",
    
    # Version info
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
]

# Default configuration
default_config = {
    'stationarity': {
        'adf': {'max_diff': 3, 'significance': 0.05},
        'kpss': {'significance': 0.05},
        'structural_break': True,
        'gls': True,
        'nonlinear': True
    },
    'causality': {
        'significance_level': 0.05,
        'max_lag': 10
    }
}

# Verify all exports are available
for name in __all__:
    if not name.startswith('__'):
        if name not in globals():
            raise ImportError(f"Required export {name} not found in package")