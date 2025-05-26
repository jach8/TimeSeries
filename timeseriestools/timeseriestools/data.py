"""Data loading utilities for TimeSeriesTools."""

import os
from typing import Tuple, Optional, Union, Dict, Any

import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def test_data1(
    path_to_src: Optional[str] = None,
    return_xy: bool = False
) -> Union[DataFrame, Tuple[DataFrame, Series]]:
    """Load pre-generated test data from pickle file."""
    if path_to_src is None:
        path_to_src = os.path.dirname(__file__)
    
    if not path_to_src.endswith('/'):
        path_to_src = path_to_src + '/'
        
    try:
        data = pd.read_pickle(os.path.join(path_to_src, 'test_data', 'data.pkl'))
        x = data['xvar']
        y = data['yvar']
        y.name = 'target'
    except FileNotFoundError:
        raise FileNotFoundError(
            "Test data not found. Run generate_data.py to create test data."
        )
    
    if return_xy:
        return x, y
    return pd.concat([x, y], axis=1)


def random_test_data(
    n: int = 500,
    start_date: str = '2020-01-01',
    return_xy: bool = False
) -> Union[DataFrame, Tuple[DataFrame, Series]]:
    """Generate random test data with date index."""
    end_date = pd.Timedelta(days=n) + pd.to_datetime(start_date)
    date_index = pd.date_range(start_date, end_date, freq='D')
    n = len(date_index)

    # Create target
    y = pd.Series(
        np.random.normal(0, 1, n).cumsum(),
        name='target',
        index=date_index
    )

    # Create features with varying relationships
    x = DataFrame({
        'x1': 0.7 * y + np.random.normal(0, 0.3, n),  # Strong relationship
        'x2': 0.3 * y + np.random.normal(0, 0.7, n),  # Moderate relationship
        'x3': np.random.normal(0, 1, n),  # No relationship
        'x4': np.sin(np.linspace(0, 8*np.pi, n))  # Periodic pattern
    }, index=date_index)
    
    if return_xy:
        return x, y
    return pd.concat([x, y], axis=1)


def test_data2(
    return_xy: bool = False
) -> Union[DataFrame, Tuple[DataFrame, Series]]:
    """Load macroeconomic test data."""
    try:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'test_data', 'stock_returns.csv'),
            parse_dates=True,
            index_col=0
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stock returns data not found. Run generate_data.py to create test data."
        )
    
    if return_xy:
        return (
            data.drop(columns=['SPY']),
            data['SPY']
        )
    return data


def test_data3(
    start_date: str = '2020-01-01',
    return_xy: bool = False,
    target: str = "SPY"
) -> Union[DataFrame, Tuple[DataFrame, Series]]:
    """Load stock returns test data."""
    try:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'test_data', 'stock_returns.csv'),
            parse_dates=['Date'],
            index_col='Date'
        )
        
        # Filter by date
        data = data[start_date:]
        
        if return_xy:
            return data.drop(columns=[target]), data[target]
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stock returns data not found. Run generate_data.py to create test data."
        )


__all__ = [
    'test_data1',
    'test_data2',
    'test_data3',
    'random_test_data'
]