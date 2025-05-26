"""Tests for the data module."""

import os
from typing import cast, Tuple, Union

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from timeseriestools.data import (
    test_data1,
    test_data2,
    test_data3,
    random_test_data
)


def test_random_test_data() -> None:
    """Test random data generation."""
    # Test default parameters
    result = random_test_data()
    df = cast(DataFrame, result if not isinstance(result, tuple) else result[0])
    assert isinstance(df, DataFrame)
    assert df.shape[1] == 5  # 4 features + 1 target
    assert len(df) == 500  # default size
    assert isinstance(df.index, pd.DatetimeIndex)
    
    # Test return_xy=True
    result = random_test_data(return_xy=True)
    if isinstance(result, tuple):
        X, y = result
        assert isinstance(X, DataFrame)
        assert isinstance(y, Series)
        assert X.shape[1] == 4
        assert len(y) == len(X)
    
    # Test custom size
    result = random_test_data(n=100)
    df = cast(DataFrame, result if not isinstance(result, tuple) else result[0])
    assert len(df) == 100
    
    # Test data quality
    assert not df.isna().any().any()
    assert df.index.is_monotonic_increasing


def test_test_data2() -> None:
    """Test macroeconomic data loading."""
    # Test combined data
    result = test_data2()
    df = cast(DataFrame, result if not isinstance(result, tuple) else result[0])
    assert isinstance(df, DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'realgdp' in df.columns
    
    # Test split data
    result = test_data2(return_xy=True)
    if isinstance(result, tuple):
        X, y = result
        assert isinstance(X, DataFrame)
        assert isinstance(y, Series)
        assert y.name == 'realgdp'
        assert 'realgdp' not in X.columns
        assert len(y) == len(X)
    
    # Test data quality
    assert not df.isna().any().any()
    assert df.index.is_monotonic_increasing


def test_test_data3() -> None:
    """Test stock returns data loading."""
    # Test combined data
    result = test_data3()
    df = cast(DataFrame, result if not isinstance(result, tuple) else result[0])
    assert isinstance(df, DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'SPY' in df.columns
    
    # Test split data with default target
    result = test_data3(return_xy=True)
    if isinstance(result, tuple):
        X, y = result
        assert isinstance(X, DataFrame)
        assert isinstance(y, Series)
        assert y.name == 'SPY'
        assert 'SPY' not in X.columns
    
    # Test custom target
    result = test_data3(return_xy=True, target='AAPL')
    if isinstance(result, tuple):
        X, y = result
        assert isinstance(X, DataFrame)
        assert isinstance(y, Series)
        assert y.name == 'AAPL'
        assert 'AAPL' not in X.columns
    
    # Test data quality
    assert df.index.is_monotonic_increasing


def test_test_data1(tmp_path) -> None:
    """Test pickle data loading."""
    # Skip if test data file not found
    data_path = os.path.join(os.path.dirname(__file__), '..', 'test_data/data.pkl')
    if not os.path.exists(data_path):
        return
    
    # Test combined data
    result = test_data1()
    df = cast(DataFrame, result if not isinstance(result, tuple) else result[0])
    assert isinstance(df, DataFrame)
    assert 'target' in df.columns
    
    # Test split data
    result = test_data1(return_xy=True)
    if isinstance(result, tuple):
        X, y = result
        assert isinstance(X, DataFrame)
        assert isinstance(y, Series)
        assert y.name == 'target'
        assert 'target' not in X.columns
    
    # Test data consistency
    result1 = test_data1()
    result2 = test_data1()
    df1 = cast(DataFrame, result1 if not isinstance(result1, tuple) else result1[0])
    df2 = cast(DataFrame, result2 if not isinstance(result2, tuple) else result2[0])
    assert_frame_equal(df1, df2)