"""Generate test data for TimeSeriesTools package."""

import os
import pickle
import pandas as pd
import numpy as np

# Ensure we're in the package root directory
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(PACKAGE_ROOT, 'timeseriestools', 'test_data')

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    print(f"✓ Created directory: {TEST_DATA_DIR}")

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
    
    data = {
        'xvar': features,
        'yvar': target
    }
    
    # Save data
    output_path = os.path.join(TEST_DATA_DIR, 'data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Generated sample data: {output_path}")
    
    return data

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
    
    # Save data
    output_path = os.path.join(TEST_DATA_DIR, 'stock_returns.csv')
    stocks.to_csv(output_path)
    print(f"✓ Generated stock returns data: {output_path}")
    
    return stocks

def main():
    """Generate all test data files."""
    print("Generating test data...")
    
    # Create directories
    create_directories()
    
    # Generate data
    generate_sample_data()
    generate_stock_returns()
    
    print("\nData generation complete! ✨")

if __name__ == "__main__":
    main()