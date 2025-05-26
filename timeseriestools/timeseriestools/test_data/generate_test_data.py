"""Generate test data for TimeSeriesTools package."""

import os
import pickle
import pandas as pd
import numpy as np

def generate_sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create target variable
    target = pd.Series(
        np.random.normal(0, 1, n).cumsum(),
        name='target',
        index=dates
    )
    
    # Create features
    features = pd.DataFrame({
        'x1': 0.7 * target + np.random.normal(0, 0.3, n),
        'x2': 0.3 * target + np.random.normal(0, 0.7, n),
        'x3': np.random.normal(0, 1, n),
        'x4': np.sin(np.linspace(0, 8*np.pi, n))
    }, index=dates)
    
    return {
        'xvar': features,
        'yvar': target
    }

def generate_stock_returns():
    """Generate simulated stock return data."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create market factor
    market = np.random.normal(0.0005, 0.01, n).cumsum()
    
    # Create correlated stock returns
    stocks = pd.DataFrame({
        'SPY': market + np.random.normal(0, 0.002, n),
        'AAPL': 1.2 * market + np.random.normal(0, 0.003, n),
        'MSFT': 1.1 * market + np.random.normal(0, 0.003, n),
        'GOOGL': 1.3 * market + np.random.normal(0, 0.004, n),
        'AMZN': 1.4 * market + np.random.normal(0, 0.005, n)
    }, index=dates)
    
    return stocks

def main():
    """Generate and save test data files."""
    # Create directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Generate and save sample data
    data = generate_sample_data()
    with open('test_data/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✓ Generated sample data")
    
    # Generate and save stock returns
    stocks = generate_stock_returns()
    stocks.to_csv('test_data/stock_returns.csv')
    print("✓ Generated stock returns data")
    
if __name__ == "__main__":
    main()
    print("\nTest data generation complete!")