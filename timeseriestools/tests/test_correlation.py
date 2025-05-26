"""Tests for the correlation module."""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, cast
from statsmodels.tsa.vector_ar.var_model import VARResults

from timeseriestools.correlation import (
    AnalyzeCorrelation,
    CorrelationAnalysisError,
    DataValidationError,
    StationarityError,
    VARModelError
)


def create_correlated_data(n: int = 100) -> tuple[DataFrame, Series]:
    """Create time series with known correlation structure.
    
    Returns:
        tuple[DataFrame, Series]: Features and target with known relationships
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create base series
    base = np.random.normal(0, 1, n).cumsum()
    
    # Create correlated features
    x1 = 0.7 * base + np.random.normal(0, 0.3, n)  # Strong correlation
    x2 = 0.3 * base + np.random.normal(0, 0.7, n)  # Moderate correlation
    x3 = np.random.normal(0, 1, n)  # No correlation
    
    # Create target
    y = base + np.random.normal(0, 0.1, n)
    
    X = DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    }, index=dates)
    
    y = Series(y, index=dates, name='target')
    
    return X, y


def test_correlation_initialization(stationarity_config: Dict, causality_config: Dict) -> None:
    """Test AnalyzeCorrelation initialization."""
    X, y = create_correlated_data()
    
    # Test default initialization
    analyzer = AnalyzeCorrelation(X, y)
    assert analyzer.cause == 'target'
    assert not analyzer.decompose
    assert isinstance(analyzer.df, DataFrame)
    
    # Test with configurations
    analyzer = AnalyzeCorrelation(
        X, y,
        decompose=True,
        verbose=True,
        stationarity_config=stationarity_config,
        causality_config=causality_config
    )
    assert analyzer.decompose
    assert analyzer.verbose


def test_input_validation() -> None:
    """Test input data validation."""
    X, y = create_correlated_data()
    
    # Test invalid X type
    try:
        AnalyzeCorrelation(X.values, y)  # type: ignore
        assert False, "Should raise DataValidationError for non-DataFrame X"
    except DataValidationError:
        pass
    
    # Test invalid y type
    try:
        AnalyzeCorrelation(X, y.values)  # type: ignore
        assert False, "Should raise DataValidationError for non-Series y"
    except DataValidationError:
        pass
    
    # Test unnamed target
    try:
        y_unnamed = y.copy()
        y_unnamed.name = None
        AnalyzeCorrelation(X, y_unnamed)
        assert False, "Should raise DataValidationError for unnamed target"
    except DataValidationError:
        pass


def test_data_processing() -> None:
    """Test data processing methods."""
    X, y = create_correlated_data()
    
    # Test without decomposition
    analyzer = AnalyzeCorrelation(X, y)
    assert analyzer.df.shape[1] == X.shape[1] + 1
    assert analyzer.features == X.columns.tolist()
    
    # Test with decomposition
    analyzer = AnalyzeCorrelation(X, y, decompose=True)
    assert analyzer.df.shape[1] <= X.shape[1] + 1
    assert all('PC' in col for col in analyzer.features)


def test_analyze_relationships() -> None:
    """Test complete analysis pipeline."""
    X, y = create_correlated_data()
    analyzer = AnalyzeCorrelation(X, y)
    
    results = analyzer.analyze_relationships()
    assert isinstance(results, dict)
    
    # Check required keys in results
    assert 'stationarity_report' in results
    assert 'var_model' in results
    assert 'causality' in results
    assert 'new_data' in results
    
    # Check stationarity report
    assert isinstance(results['stationarity_report'], dict)
    
    # Check VAR model
    assert isinstance(results['var_model'], VARResults)
    
    # Check causality results
    assert isinstance(results['causality'], dict)
    assert 'granger' in results['causality']
    
    # Check transformed data
    assert isinstance(results['new_data'], DataFrame)


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    X, y = create_correlated_data()
    
    # Test with single column
    single_X = X[['x1']]
    analyzer = AnalyzeCorrelation(single_X, y)
    results = analyzer.analyze_relationships()
    assert isinstance(results, dict)
    
    # Test with NaN values
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan
    try:
        analyzer = AnalyzeCorrelation(X_with_nan, y)
        results = analyzer.analyze_relationships()
        assert not results['new_data'].isna().any().any()
    except Exception as e:
        assert False, f"Should handle NaN values, got {str(e)}"
    
    # Test with constant column
    X_with_const = X.copy()
    X_with_const['const'] = 1.0
    analyzer = AnalyzeCorrelation(X_with_const, y)
    results = analyzer.analyze_relationships()
    assert isinstance(results, dict)


def test_error_handling() -> None:
    """Test error handling and custom exceptions."""
    X, y = create_correlated_data()
    
    # Test with mismatched indices
    try:
        y_shifted = y.shift(1)
        AnalyzeCorrelation(X, y_shifted)
        assert False, "Should raise DataValidationError for mismatched indices"
    except DataValidationError:
        pass
    
    # Test with empty data
    try:
        AnalyzeCorrelation(DataFrame(), y)
        assert False, "Should raise DataValidationError for empty data"
    except DataValidationError:
        pass
    
    # Test with too few observations
    try:
        analyzer = AnalyzeCorrelation(X.iloc[:2], y.iloc[:2])
        analyzer.analyze_relationships()
        assert False, "Should raise VARModelError for insufficient data"
    except VARModelError:
        pass