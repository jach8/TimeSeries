"""Test script to verify TimeSeriesTools package installation."""

import sys
import importlib
from typing import Any, Dict, List, Optional


def check_import(module: str) -> Optional[str]:
    """Try importing a module and return error message if it fails."""
    try:
        importlib.import_module(module)
        return None
    except ImportError as e:
        return str(e)


def check_attribute(module: Any, attr: str) -> bool:
    """Check if a module has a given attribute."""
    return hasattr(module, attr)


def main() -> None:
    """Run installation verification tests."""
    print("Verifying TimeSeriesTools installation...\n")

    # Check Python version
    py_version = sys.version.split()[0]
    print(f"Python version: {py_version}")
    if tuple(map(int, py_version.split("."))) < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print("✓ Python version OK")

    # Check core dependencies
    dependencies = ["numpy", "pandas", "statsmodels", "scikit-learn"]
    print("\nChecking dependencies:")
    for dep in dependencies:
        error = check_import(dep)
        if error:
            print(f"❌ {dep}: {error}")
            sys.exit(1)
        print(f"✓ {dep}")

    # Import timeseriestools
    print("\nImporting timeseriestools...")
    try:
        import timeseriestools as ts
        print("✓ Package imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        print("\nTry reinstalling the package:")
        print("pip install -e .")
        sys.exit(1)

    # Check version information
    print("\nChecking version information:")
    version_attrs = ["__version__", "__author__", "__license__"]
    for attr in version_attrs:
        if check_attribute(ts, attr):
            print(f"✓ {attr}: {getattr(ts, attr)}")
        else:
            print(f"❌ Missing {attr}")
            sys.exit(1)

    # Check public API
    print("\nVerifying public API:")
    required_classes = ["Analyze", "AnalyzeCorrelation", "CausalityAnalyzer", "StationaryTests"]
    required_functions = ["test_data1", "test_data2", "test_data3", "random_test_data"]

    for cls in required_classes:
        if check_attribute(ts, cls):
            print(f"✓ {cls}")
        else:
            print(f"❌ Missing {cls}")
            sys.exit(1)

    for func in required_functions:
        if check_attribute(ts, func):
            print(f"✓ {func}")
        else:
            print(f"❌ Missing {func}")
            sys.exit(1)

    # Try basic functionality
    print("\nTesting basic functionality:")
    try:
        # Create test data
        X, y = ts.random_test_data(n=10, return_xy=True)
        print("✓ Generated test data")

        # Create analyzer
        analyzer = ts.Analyze()
        print("✓ Created analyzer instance")

        # Run analysis
        results = analyzer.analyze_correlation(X, y)
        print("✓ Ran correlation analysis")

        expected_keys = ["stationarity_report", "var_model", "causality", "new_data"]
        if all(key in results for key in expected_keys):
            print("✓ Analysis results complete")
        else:
            print("❌ Missing results keys")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        sys.exit(1)

    print("\n✨ Installation verified successfully! ✨")


if __name__ == "__main__":
    main()