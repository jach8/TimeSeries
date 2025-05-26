"""Minimal test script for TimeSeriesTools package."""

def test_import():
    """Test importing the package."""
    print("Testing TimeSeriesTools imports...")
    
    try:
        import timeseriestools as ts
        print("✓ Basic import successful")
        
        # Test version info
        print(f"✓ Version: {ts.__version__}")
        
        # Test imports
        from timeseriestools import (
            Analyze,
            AnalyzeCorrelation,
            CausalityAnalyzer,
            StationaryTests,
            random_test_data
        )
        print("✓ All components imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {str(e)}")
        print("\nTry installing the package:")
        print("python setup_fresh.py")
        return False
        
def test_functionality():
    """Test basic package functionality."""
    try:
        import timeseriestools as ts
        
        # Generate test data
        print("\nGenerating test data...")
        X, y = ts.random_test_data(n=10, return_xy=True)
        print("✓ Test data generated")
        
        # Create analyzer
        print("\nCreating analyzer...")
        analyzer = ts.Analyze(verbose=False)
        print("✓ Analyzer created")
        
        # Run basic analysis
        print("\nRunning analysis...")
        results = analyzer.analyze_correlation(X, y)
        print("✓ Analysis completed")
        
        # Check results
        required_keys = ['stationarity_report', 'var_model', 'causality', 'new_data']
        missing_keys = [key for key in required_keys if key not in results]
        
        if not missing_keys:
            print("✓ All expected results present")
            return True
        else:
            print(f"✗ Missing results: {missing_keys}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def main():
    """Run minimal tests."""
    print("Running minimal package tests...\n")
    
    if not test_import():
        return
        
    if not test_functionality():
        return
        
    print("\n✨ All minimal tests passed!")

if __name__ == "__main__":
    main()