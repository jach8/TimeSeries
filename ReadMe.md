# Time Series Analysis Toolkit

This package provides a comprehensive suite of tools for time series analysis, focusing on stationarity checks, causality analysis, and correlation study. It's designed to assist data scientists and analysts in understanding complex temporal relationships data through integrated and modular analysis techniques.

## Introduction

This toolkit is designed for anyone working with time series data, from forecasting stock prices to analyzing climate change patterns. It simplifies complex statistical tests into an easy-to-use Python framework.


## Overview

- **StationaryTests**: A class for performing various stationarity tests on time series data.
- **CausalityAnalyzer**: Implements causality tests like Granger causality to determine if one time series can predict another.
- **AnalyzeCorrelation**: Combines stationarity checks, data preprocessing, and causality analysis into a single analysis pipeline.
- **Analyze**: A high-level interface class that simplifies the usage of the above tools with default configurations.


## Analyze

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

## AnalyzeCorrelation

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

## StationaryTests

The `StationaryTests` class in `stationary_checks.py` is used for performing various stationarity tests. Below is an example of how to use it:
- For the configuration you can change the parameters as needed. 
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


# Example configuration
stationarity_cfg = {
    'adf': {'max_diff': 5, 'significance': 0.05},
    'kpss': {'significance': 0.05},
    'pp': {'significance': 0.05},
    'structural_break': True,
    'seasonal': {'period': 12}, 
    'gls': False,
    'nonlinear': True
}

# Initialize the StationaryTests class (Default Config shown above)
st = StationaryTests(verbose=True)

# Check stationarity
stationary_df, report, results = st.check_stationarity(data)

# Print the results
print(report)
```

## CausalityAnalyzer

The `CausalityAnalyzer` class in `causality_logic.py` is used for performing causality tests. Below is an example of how to use it:

If you would like to use a different significance level or maximum lag, you can change the parameters as needed, in the configuration dictionary. 

```python
import pandas as pd
from src.causality_logic import CausalityAnalyzer

# Load your data into a DataFrame
data = pd.DataFrame(...)  # Replace with your DataFrame

# Example Config
causality_cfg = {
    'significance_level': 0.05,
    'max_lag': 4
}

# Initialize the CausalityAnalyzer class Default Config shown above. 
ca = CausalityAnalyzer(verbose=True)

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

## Contributing
We welcome contributions! Please create a pull request to get involved.

## Support
For questions or support, please open an issue on our GitHub Issues page 

## License

This project is licensed under the MIT License.
