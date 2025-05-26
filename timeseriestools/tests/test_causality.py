"""Tests for the causality module."""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, cast
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

from timeseriestools.causality import CausalityAnalyzer


def create_causal_data(n: int = 100) -> DataFrame:
    """Create time series with known causal relationship.
    
    x1 -> x2 (x1 Granger causes x2)
    x3 is independent
    """
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create x1 as random walk
    x1 = np.random.normal(0, 1, n).cumsum()
    
    # Create x2 that depends on lagged x1
    x2 = np.zeros(n)
    for i in range(1, n):
        x2[i] = 0.5 * x1[i-1] + np.random.normal(0, 0.1)
    
    # Create independent x3
    x3 = np.random.normal(0, 1, n)
    
    return DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    }, index=dates)


def test_causality_initialization(causality_config: Dict) -> None:
    """Test CausalityAnalyzer initialization."""
    # Test with default config
    analyzer = CausalityAnalyzer()
    assert hasattr(analyzer, 'significance_level')
    assert hasattr(analyzer, 'max_lag')
    
    # Test with custom config
    analyzer = CausalityAnalyzer(causality_config=causality_config)
    assert analyzer.significance_level == causality_config['significance_level']
    assert analyzer.max_lag == causality_config['max_lag']
    
    # Test invalid config
    try:
        CausalityAnalyzer(causality_config={'significance_level': 1.5})
        assert False, "Should raise ValueError for invalid significance level"
    except ValueError:
        pass


def test_granger_test() -> None:
    """Test Granger causality testing."""
    analyzer = CausalityAnalyzer()
    data = create_causal_data()
    
    # Test known causal relationship x1 -> x2
    results = analyzer.granger_test(data, target='x2')
    assert isinstance(results, dict)
    
    # Verify x1 causes x2
    key = ('x2', 'x1')
    assert key in results
    assert any(results[key]['ssr_ftest'] < analyzer.significance_level)
    
    # Test non-causal relationship x3 -> x1
    results = analyzer.granger_test(data, target='x1')
    key = ('x1', 'x3')
    assert key in results
    assert all(results[key]['ssr_ftest'] > analyzer.significance_level)
    
    # Check format of results
    for k, v in results.items():
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert isinstance(v, pd.DataFrame)
        assert 'ssr_ftest' in v.columns


def test_causality_tests() -> None:
    """Test comprehensive causality testing."""
    analyzer = CausalityAnalyzer()
    data = create_causal_data()
    
    # Run tests without VAR model
    results = analyzer.causality_tests(data, target='x2')
    assert isinstance(results, dict)
    assert 'granger' in results
    assert isinstance(results['granger'], list)
    
    # Run tests with VAR model
    from statsmodels.tsa.vector_ar.var_model import VAR
    model_wrapper = VAR(data).fit()
    # Cast VARResultsWrapper to VARResults
    model = cast(VARResults, model_wrapper)
    results = analyzer.causality_tests(data, target='x2', model=model)
    assert 'instantaneous' in results
    assert 'impulse_response' in results


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    analyzer = CausalityAnalyzer()
    
    # Test empty DataFrame
    try:
        analyzer.granger_test(pd.DataFrame(), target='x1')
        assert False, "Should raise ValueError for empty DataFrame"
    except ValueError:
        pass
    
    # Test missing target
    data = create_causal_data()
    try:
        analyzer.granger_test(data, target='nonexistent')
        assert False, "Should raise ValueError for missing target"
    except ValueError:
        pass
    
    # Test insufficient data
    try:
        analyzer.granger_test(data.iloc[:2], target='x1')
        assert False, "Should raise ValueError for insufficient data"
    except ValueError:
        pass


def test_column_pairs() -> None:
    """Test column pair generation."""
    analyzer = CausalityAnalyzer()
    data = create_causal_data()
    
    pairs = analyzer._get_column_pairs(data, 'x1')
    assert isinstance(pairs, list)
    assert len(pairs) == 2  # x1->x2, x1->x3
    assert all(isinstance(p, list) for p in pairs)
    assert all(len(p) == 2 for p in pairs)
    assert all(p[0] == 'x1' for p in pairs)