"""Tests for the stationarity module."""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, cast
from pandas.testing import assert_frame_equal, assert_series_equal

from timeseriestools.stationarity import StationaryTests


def create_non_stationary_data(n: int = 100) -> Series:
    """Create non-stationary time series with trend."""
    np.random.seed(42)
    t = np.linspace(0, 1, n)
    trend = 5 * t
    noise = np.random.normal(0, 0.1, n)
    return pd.Series(trend + noise, name='non_stationary')


def create_stationary_data(n: int = 100) -> Series:
    """Create stationary time series."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0, 1, n), name='stationary')


def test_stationarity_initialization(stationarity_config: Dict) -> None:
    """Test StationaryTests initialization."""
    # Test with default config
    tester = StationaryTests()
    assert hasattr(tester, 'test_config')
    assert hasattr(tester, 'verbose')
    
    # Test with custom config
    tester = StationaryTests(test_config=stationarity_config)
    assert tester.test_config == stationarity_config


def test_adf_test() -> None:
    """Test Augmented Dickey-Fuller test."""
    tester = StationaryTests()
    
    # Test non-stationary data
    non_stationary = create_non_stationary_data()
    result = tester.adf_test(non_stationary)
    assert isinstance(result, dict)
    assert not result['stationary']
    
    # Test stationary data
    stationary = create_stationary_data()
    result = tester.adf_test(stationary)
    assert isinstance(result, dict)
    assert result['stationary']


def test_kpss_test() -> None:
    """Test KPSS test."""
    tester = StationaryTests()
    
    # Test non-stationary data
    non_stationary = create_non_stationary_data()
    result = tester.kpss_test(non_stationary)
    assert isinstance(result, dict)
    assert not result['stationary']
    
    # Test stationary data
    stationary = create_stationary_data()
    result = tester.kpss_test(stationary)
    assert isinstance(result, dict)
    assert result['stationary']


def test_check_stationarity() -> None:
    """Test comprehensive stationarity check."""
    tester = StationaryTests()
    
    # Create test data
    data = pd.DataFrame({
        'non_stationary': create_non_stationary_data(),
        'stationary': create_stationary_data()
    })
    
    # Run stationarity check
    stationary_df, report, full_results = tester.check_stationarity(data)
    
    # Check outputs
    assert isinstance(stationary_df, DataFrame)
    assert isinstance(report, dict)
    assert isinstance(full_results, dict)
    
    # Check report structure
    assert 'non_stationary' in report
    assert 'stationary' in report
    assert 'diffs_applied' in report['non_stationary']
    assert 'final_status' in report['non_stationary']
    
    # Check that non-stationary series required differencing
    diffs_applied = cast(int, report['non_stationary']['diffs_applied'])
    assert diffs_applied > 0
    
    # Check that stationary series did not require differencing
    diffs_stationary = cast(int, report['stationary']['diffs_applied'])
    assert diffs_stationary == 0


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    tester = StationaryTests()
    
    # Test empty DataFrame
    try:
        tester.check_stationarity(pd.DataFrame())
        assert False, "Should raise ValueError for empty DataFrame"
    except ValueError:
        pass
    
    # Test invalid input type
    try:
        tester.check_stationarity(cast(DataFrame, [1, 2, 3]))  # type: ignore
        assert False, "Should raise ValueError for invalid input type"
    except ValueError:
        pass
    
    # Test DataFrame with NaN values
    data = pd.DataFrame({
        'with_nan': [1, np.nan, 3],
        'without_nan': [1, 2, 3]
    })
    stationary_df, report, _ = tester.check_stationarity(data)
    assert not stationary_df.isna().any().any()


def test_test_battery() -> None:
    """Test individual test battery execution."""
    tester = StationaryTests()
    series = create_stationary_data()
    
    # Run test battery
    results = tester._run_test_battery(series)
    
    # Check results structure
    assert isinstance(results, dict)
    if 'adf' in results:
        assert 'stationary' in results['adf']
        assert 'p' in results['adf']
    
    if 'kpss' in results:
        assert 'stationary' in results['kpss']
        assert 'p' in results['kpss']


def test_kss_test() -> None:
    """Test Kapetanios-Snell-Shin test."""
    tester = StationaryTests()
    
    # Test non-stationary data
    non_stationary = create_non_stationary_data()
    result = tester._kss_test(non_stationary)
    assert isinstance(result, dict)
    assert 'stationary' in result
    assert 'p' in result
    
    # Test stationary data
    stationary = create_stationary_data()
    result = tester._kss_test(stationary)
    assert isinstance(result, dict)
    assert 'stationary' in result
    assert 'p' in result