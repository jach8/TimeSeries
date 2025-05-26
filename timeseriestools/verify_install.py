"""Installation verification script for TimeSeriesTools."""

import os
import sys
from pathlib import Path


def check_package_structure():
    """Verify the package directory structure."""
    base_dir = Path(__file__).parent
    
    required_files = [
        'setup.py',
        'pyproject.toml',
        'requirements.txt',
        'timeseriestools/__init__.py',
        'timeseriestools/analyze.py',
        'timeseriestools/causality.py',
        'timeseriestools/correlation.py',
        'timeseriestools/data.py',
        'timeseriestools/stationarity.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not (base_dir / file_path).exists():
            missing.append(file_path)
            
    return missing


def verify_environment():
    """Verify Python environment and dependencies."""
    # Check Python version
    py_version = sys.version_info
    if py_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print("✓ Python version OK")
    
    # Check dependencies installation
    try:
        import pandas
        import numpy
        import statsmodels
        import sklearn
        print("✓ Core dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    return True


def verify_test_data():
    """Verify test data is present."""
    base_dir = Path(__file__).parent
    test_data_dir = base_dir / 'timeseriestools' / 'test_data'
    
    required_files = [
        'data.pkl',
        'stock_returns.csv'
    ]
    
    if not test_data_dir.exists():
        print("❌ Test data directory missing")
        return False
        
    missing = []
    for file_path in required_files:
        if not (test_data_dir / file_path).exists():
            missing.append(file_path)
            
    return not missing


def main():
    """Run verification checks."""
    print("Verifying TimeSeriesTools package installation...\n")
    
    # Check package structure
    print("Checking package structure...")
    missing_files = check_package_structure()
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nTry running setup_fresh.py to fix installation.")
        return
    print("✓ Package structure OK")
    
    # Check environment
    print("\nChecking environment...")
    if not verify_environment():
        print("\nTry reinstalling dependencies:")
        print("pip install -r requirements.txt")
        return
    print("✓ Environment OK")
    
    # Check test data
    print("\nChecking test data...")
    if not verify_test_data():
        print("\nTry regenerating test data:")
        print("python generate_data.py")
        return
    print("✓ Test data OK")
    
    print("\n✨ Installation verified successfully!")
    print("\nYou can now:")
    print("1. Run the minimal test:")
    print("   python minimal_test.py")
    print("2. Run the full test suite:")
    print("   pytest tests/")


if __name__ == "__main__":
    main()