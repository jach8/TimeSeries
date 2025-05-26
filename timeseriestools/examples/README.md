# TimeSeriesTools Examples and Tests

This directory contains example code and test scripts for the TimeSeriesTools package.

## Quick Test

To run a quick test of the package functionality:

### Unix/Linux/Mac
```bash
./run_test.sh
```

### Windows
```batch
run_test.bat
```

This will:
1. Create a virtual environment
2. Install the package and dependencies
3. Run basic functionality tests
4. Display test results

## Test Script (package_test.py)

The test script demonstrates:
- Package import and version check
- Data generation and loading
- Basic analysis workflow
- Advanced features (PCA decomposition)
- Error handling

You can also run it directly after installing the package:
```bash
python package_test.py
```

## Common Issues

1. Import Errors:
   ```
   ModuleNotFoundError: No module named 'timeseriestools'
   ```
   Solution: Install the package in editable mode:
   ```bash
   pip install -e .
   ```

2. Missing Dependencies:
   ```
   ModuleNotFoundError: No module named 'pandas'
   ```
   Solution: Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Test Data Not Found:
   ```
   FileNotFoundError: test_data/data.pkl not found
   ```
   Solution: Create test_data directory:
   ```bash
   mkdir test_data
   ```

## Test Output

The test script will show:
- Package version
- Data loading test results
- Analysis workflow results
- Advanced features test results

Expected output format:
```
Starting TimeSeriesTools package tests...

Package version: 0.1.0

Testing data loading functions:
✓ random_test_data

Testing analysis workflow...
✓ Basic analysis completed
✓ Results structure verified

Analysis Summary:
[Analysis results will be displayed here]

Testing advanced features...
✓ PCA decomposition

All tests completed successfully! ✨
```

## Next Steps

After confirming the package works:
1. Check out the Jupyter notebooks for detailed examples
2. Review the API documentation
3. Try with your own data
4. Read the development guide if you want to contribute