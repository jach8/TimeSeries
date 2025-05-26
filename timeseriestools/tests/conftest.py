"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest
from typing import Dict


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time series data for testing.
    
    Returns:
        pd.DataFrame: DataFrame with test time series
    """
    np.random.seed(42)
    n = 100
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create test data
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.normal(0, 1, n),
        'x4': np.random.normal(0, 1, n),
        'target': np.random.normal(0, 1, n)
    }, index=dates)
    
    return data


@pytest.fixture
def stationarity_config() -> Dict:
    """Test configuration for stationarity tests.
    
    Returns:
        Dict: Configuration dictionary
    """
    return {
        'adf': {'max_diff': 3, 'significance': 0.05},
        'kpss': {'significance': 0.05},
        'structural_break': True,
        'gls': True,
        'nonlinear': True
    }


@pytest.fixture
def causality_config() -> Dict:
    """Test configuration for causality tests.
    
    Returns:
        Dict: Configuration dictionary
    """
    return {
        'significance_level': 0.05,
        'max_lag': 2
    }