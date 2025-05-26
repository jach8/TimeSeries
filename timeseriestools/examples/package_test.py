"""Basic test script to verify TimeSeriesTools package functionality."""

import pandas as pd
import numpy as np
from typing import cast, Tuple

import timeseriestools as ts


def create_test_data(n: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
    """Create sample data with known relationships."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create target
    target = pd.Series(
        np.random.normal(0, 1, n).cumsum(),
        name='target',
        index=dates
    )
    
    # Create features
    features = pd.DataFrame({
        'strong': 0.7 * target + np.random.normal(0, 0.3, n),
        'weak': 0.3 * target + np.random.normal(0, 0.7, n),
        'none': np.random.normal(0, 1, n)
    }, index=dates)
    
    return features, target


def test_package_interface():
    """Test all main components of the package interface."""
    print("Testing package interface components...")
    
    # Test version info
    print(f"\nPackage version: {ts.__version__}")
    
    # Test data loading functions
    print("\nTesting data loading functions:")
    try:
        test_data = ts.random_test_data()
        print("✓ random_test_data")
    except Exception as e:
        print(f"✗ random_test_data: {str(e)}")


def test_analysis_workflow():
    """Test complete analysis workflow."""
    print("\nTesting analysis workflow...")
    
    # Create test data
    print("\nGenerating test data...")
    features, target = create_test_data()
    print(f"Created features shape: {features.shape}")
    print(f"Created target shape: {target.shape}")
    
    # Test basic analysis
    print("\nTesting basic analysis:")
    try:
        # Initialize analyzer
        analyzer = ts.Analyze(verbose=True)
        
        # Configure tests
        stationarity_config = {
            'adf': {'max_diff': 3, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'structural_break': True,
            'gls': True
        }
        
        causality_config = {
            'significance_level': 0.05,
            'max_lag': 3
        }
        
        # Create analyzer with config
        analyzer = ts.Analyze(
            verbose=True,
            stationarity_config=stationarity_config,
            causality_config=causality_config
        )
        
        # Run analysis
        results = analyzer.analyze_correlation(features, target)
        print("✓ Basic analysis completed")
        
        # Check results structure
        assert 'stationarity_report' in results, "Missing stationarity report"
        assert 'causality' in results, "Missing causality results"
        assert 'var_model' in results, "Missing VAR model"
        assert 'new_data' in results, "Missing processed data"
        print("✓ Results structure verified")
        
        # Print summary
        print("\nAnalysis Summary:")
        print("\nStationarity Report:")
        print(pd.DataFrame(results['stationarity_report']))
        
        print("\nCausality Results:")
        for relationship in results['causality']['granger']:
            print(f"{relationship[0][1]} Granger causes {relationship[0][0]} at lags {relationship[1]}")
            
    except Exception as e:
        print(f"✗ Analysis failed: {str(e)}")
        raise


def test_advanced_features():
    """Test advanced package features."""
    print("\nTesting advanced features...")
    
    features, target = create_test_data()
    analyzer = ts.Analyze(verbose=False)
    
    try:
        # Test PCA decomposition
        results_pca = analyzer.analyze_correlation(features, target, decompose=True)
        pca_features = [col for col in results_pca['new_data'].columns if col != target.name]
        print("✓ PCA decomposition")
        print(f"PCA features: {pca_features}")
        
    except Exception as e:
        print(f"✗ Advanced features failed: {str(e)}")
        raise


def main():
    """Run all package tests."""
    print("Starting TimeSeriesTools package tests...\n")
    
    try:
        # Test package interface
        test_package_interface()
        
        # Test analysis workflow
        test_analysis_workflow()
        
        # Test advanced features
        test_advanced_features()
        
        print("\nAll tests completed successfully! ✨")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()