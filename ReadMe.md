# TimeSeriesModels

This project contains Python scripts for analyzing and modeling time series data. It includes classes and methods for performing various time series analyses, such as stationarity tests, PCA decomposition, VAR modeling, and Granger causality tests.

## Files

- `main.py`: Main entry point for running the analysis.
- `correlation.py`: Contains the `AnalyzeCorrelation` class for performing correlation analysis, including stationarity tests, PCA decomposition, VAR modeling, and Granger causality tests.
- `stationary_checks.py`: Contains the `StationaryTests` class for performing various stationarity tests.
- `causality_logic.py`: Contains the `CausalityAnalyzer` class for performing causality tests.
- `data.py`: Contains functions for loading and generating test data.

## Usage

### Analyze

The `Analyze` class in `main.py` is used for analyzing the correlation between two time series. Below is an example of how to use it:

```python
import pandas as pd
from src.data import random_test_data
from main import Analyze

# Load your data into x (features) and y (target)
x, y = random_test_data(n=500, return_xy=True)

# Initialize the analysis
analyzer = Analyze(verbose=True)

# Run the analysis
results = analyzer.results(x, y)

# Print the results
print(results)
```

### AnalyzeCorrelation

The `AnalyzeCorrelation` class in `correlation.py` is used for correlation analysis of time series data. Below is an example of how to use it:

```python
import pandas as pd
from src.correlation import AnalyzeCorrelation

# Load your data into x (features) and y (target)
x = pd.DataFrame(...)  # Replace with your features DataFrame
y = pd.Series(...)  # Replace with your target Series

# Example configuration
stationarity_cfg = {
    'adf': {'max_diff': 5, 'significance': 0.05},
    'kpss': {'significance': 0.05},
    'pp': {'significance': 0.05},
    'structural_break': True,
    'gls': False,
    'nonlinear': True
}

causality_cfg = {
    'significance_level': 0.05,
    'max_lag': 4
}

# Initialize the analyzer
analyzer = AnalyzeCorrelation(
    x=x, y=y,
    stationarity_config=stationarity_cfg,
    causality_config=causality_cfg,
    verbose=True,
    decompose=True
)

# Run the analysis
results = analyzer.analyze_relationships()

# Print the results
print(results)
```

### StationaryTests

The `StationaryTests` class in `stationary_checks.py` is used for performing various stationarity tests. Below is an example of how to use it:

For the configuration you can change the parameters as needed. 
  1. `adf`: Unit Root Test for Stationarity
  2. `kpss`: Test for Trend Stationarity
  3. `pp`: Phillips-Perron Test for Unit root 
  4. `structural_break`: Zivot Andrews Test stationarity with a Structural Break (shock)
  5. `gls`: DFGLS test for unit Root.  
  6. `nonlinear`: KSS Test for Non-Linear Stationarity


```python
import pandas as pd
from src.stationary_checks import StationaryTests

# Load your data into a DataFrame
data = pd.DataFrame(...)  # Replace with your DataFrame

# Initialize the StationaryTests class
st = StationaryTests(verbose=True)

# Check stationarity
stationary_df, report, results = st.check_stationarity(data)

# Print the results
print(report)
```

### CausalityAnalyzer

The `CausalityAnalyzer` class in `causality_logic.py` is used for performing causality tests. Below is an example of how to use it:

```python
import pandas as pd
from src.causality_logic import CausalityAnalyzer

# Load your data into a DataFrame
data = pd.DataFrame(...)  # Replace with your DataFrame

# Initialize the CausalityAnalyzer class
ca = CausalityAnalyzer(significance_level=0.05, max_lag=4)

# Perform causality tests
causality_results = ca.causality_tests(data, target='target_column')

# Print the results
print(causality_results)
```

## Dependencies

- pandas
- numpy
- statsmodels
- scikit-learn
- arch
- matplotlib
- pmdarima
- tqdm

## Installation

To install the required dependencies, run:

```bash
pip install pandas numpy statsmodels scikit-learn arch matplotlib pmdarima tqdm
```

## License

This project is licensed under the MIT License.
