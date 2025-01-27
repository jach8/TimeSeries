# TimeSeriesModels

This project contains Python scripts for analyzing and modeling time series data. It includes classes and methods for performing various time series analyses, such as stationarity tests, PCA decomposition, VAR modeling, and Granger causality tests.

## Files

- `timeseries.py`: Contains the `TimeSeriesAnalyzer` class for performing time series analysis, including stationarity tests, VAR modeling, and Granger causality tests.
- `arima.py`: Contains the `arima_trend` class for handling time series data operations, including conversion, stationarity testing, and ARIMA/ARMA modeling.
- `correlation.py`: Contains the `AnalyzeCorrelation` class for performing correlation analysis, including stationarity tests, PCA decomposition, VAR modeling, and Granger causality tests.
- `stationary_checks.py`: Contains the `StationaryTests` class for performing various stationarity tests.
- `causality_logic.py`: Contains the `CausalityAnalyzer` class for performing causality tests.

## Usage

### TimeSeriesAnalyzer

The `TimeSeriesAnalyzer` class in `timeseries.py` is used for analyzing time series data. Below is an example of how to use it:

```python
import pandas as pd
from timeseries import TimeSeriesAnalyzer

# Load your data into x (features) and y (target)
x = pd.DataFrame(...)  # Replace with your features DataFrame
y = pd.Series(...)  # Replace with your target Series

# Initialize the analyzer
analyzer = TimeSeriesAnalyzer(x, y, verbose=True)

# Run the analysis
results = analyzer.analyze()

# Print the results
print(results)
```

### arima_trend

The `arima_trend` class in `arima.py` is used for ARIMA/ARMA modeling of time series data. Below is an example of how to use it:

```python
import pandas as pd
from arima import arima_trend

# Load your data into a DataFrame
data = pd.DataFrame(...)  # Replace with your DataFrame

# Initialize the arima_trend class
arima = arima_trend()

# Perform ARIMA modeling
arima_results = arima.arima_model(data)

# Print the results
print(arima_results['results'])

# Train the model
train_results = arima.train_model(data, arima_results)

# Print the training results
print(train_results)
```

### AnalyzeCorrelation

The `AnalyzeCorrelation` class in `correlation.py` is used for correlation analysis of time series data. Below is an example of how to use it:

```python
import pandas as pd
from correlation import AnalyzeCorrelation

# Load your data into x (features) and y (target)
x = pd.DataFrame(...)  # Replace with your features DataFrame
y = pd.Series(...)  # Replace with your target Series

# Example configuration
stationarity_cfg = {
    'adf': {'max_diff': 4, 'significance': 0.05},
    'kpss': {'significance': 0.05},
    'structural_break': False
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

```python
import pandas as pd
from stationary_checks import StationaryTests

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
from causality_logic import CausalityAnalyzer

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
