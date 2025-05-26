"""Tests for the analyze module."""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, Any, cast, Tuple, Union

from timeseriestools.analyze import Analyze
from timeseriestools.data import test_data1


def create_test_data(n: int = 100) -> Tuple[DataFrame, Series]:
    """Create test data for analysis.
    
    Returns:
        Tuple[DataFrame, Series]: Features and target with known relationships
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create target series
    y = Series(np.random.normal(0, 1, n).cumsum(), name='target', index=dates)
    
    # Create features with varying relationships to target
    x1 = 0.7 * y + np.random.normal(0, 0.3, n)  # Strong relationship
    x2 = 0.3 * y + np.random.normal(0, 0.7, n)  # Moderate relationship
    x3 = np.random.normal(0, 1, n)  # No relationship
    
    X = DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    }, index=dates)
    
    return X, y


def test_analyze_initialization(stationarity_config: Dict, causality_config: Dict) -> None:
    """Test Analyze class initialization."""
    # Test default initialization
    analyzer = Analyze()
    assert hasattr(analyzer, 'verbose')
    assert hasattr(analyzer, 'stationary_config')
    assert hasattr(analyzer, 'causality_config')
    
    # Test with custom configs
    analyzer = Analyze(
        verbose=True,
        stationarity_config=stationarity_config,
        causality_config=causality_config
    )
    assert analyzer.verbose
    assert analyzer.stationary_config == stationarity_config
    assert analyzer.causality_config == causality_config


def test_analyze_correlation() -> None:
    """Test correlation analysis method."""
    analyzer = Analyze(verbose=False)
    features, target = create_test_data()
    
    # Ensure proper types
    features_df = cast(DataFrame, features)
    target_series = cast(Series, target)
    
    # Test without decomposition
    results = analyzer.analyze_correlation(
        x=features_df,
        y=target_series
    )
    assert isinstance(results, dict)
    assert 'stationarity_report' in results
    assert 'var_model' in results
    assert 'causality' in results
    assert 'new_data' in results
    
    # Test with decomposition
    results = analyzer.analyze_correlation(
        x=features_df,
        y=target_series,
        decompose=True
    )
    assert isinstance(results, dict)
    assert 'new_data' in results
    assert all('PC' in col for col in results['new_data'].columns if col != 'target')


def test_results_access() -> None:
    """Test results access and storage."""
    analyzer = Analyze()
    features_df, target_series = create_test_data()
    
    # Test without running analysis
    try:
        analyzer.get_last_results()
        assert False, "Should raise ValueError if no analysis has been run"
    except ValueError:
        pass
    
    # Test after running analysis
    first_results = analyzer.analyze_correlation(features_df, target_series)
    stored_results = analyzer.get_last_results()
    assert first_results == stored_results


def test_with_test_data() -> None:
    """Test analyze with built-in test data."""
    analyzer = Analyze()
    
    try:
        result = test_data1(return_xy=True)
        if isinstance(result, tuple):
            features, target = result
            features_df = cast(DataFrame, features)
            target_series = cast(Series, target)
            results = analyzer.analyze_correlation(features_df, target_series)
            assert isinstance(results, dict)
            assert all(key in results for key in [
                'stationarity_report',
                'var_model',
                'causality',
                'new_data'
            ])
    except Exception as e:
        # Skip if test data not available
        if "No such file or directory" not in str(e):
            raise


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    analyzer = Analyze()
    features_df, target_series = create_test_data()
    
    # Test with single feature
    features_single = features_df[['x1']]
    results = analyzer.analyze_correlation(features_single, target_series)
    assert isinstance(results, dict)
    
    # Test with short time series
    try:
        analyzer.analyze_correlation(features_df.iloc[:2], target_series.iloc[:2])
        assert False, "Should raise error for insufficient data"
    except Exception:
        pass
    
    # Test with NaN values
    features_with_nan = features_df.copy()
    features_with_nan.iloc[0, 0] = np.nan
    try:
        results = analyzer.analyze_correlation(features_with_nan, target_series)
        assert not results['new_data'].isna().any().any()
    except Exception as e:
        assert False, f"Should handle NaN values, got {str(e)}"


def test_configuration_validation() -> None:
    """Test configuration validation."""
    # Test invalid stationarity config
    try:
        Analyze(stationarity_config={'invalid_key': 'value'})
    except ValueError:
        pass
    
    # Test invalid causality config
    try:
        Analyze(causality_config={'significance_level': 2.0})
        assert False, "Should raise ValueError for invalid significance level"
    except ValueError:
        pass
    
    # Test valid configurations
    valid_stationarity_config = {
        'adf': {'max_diff': 3, 'significance': 0.05},
        'kpss': {'significance': 0.05}
    }
    valid_causality_config = {
        'significance_level': 0.05,
        'max_lag': 3
    }
    analyzer = Analyze(
        stationarity_config=valid_stationarity_config,
        causality_config=valid_causality_config
    )
    assert analyzer.stationary_config == valid_stationarity_config
    assert analyzer.causality_config == valid_causality_config