# TimeSeriesModels

This project contains Python scripts for analyzing and modeling time series data. It includes classes and methods for performing various time series analyses, such as stationarity tests, ARIMA/ARMA modeling, and Granger causality tests.

## Files

- `timeseries.py`: Contains the `TimeSeriesAnalyzer` class for performing time series analysis, including stationarity tests, VAR modeling, and Granger causality tests.
- `arima.py`: Contains the `arima_trend` class for handling time series data operations, including conversion, stationarity testing, and ARIMA/ARMA modeling.
- `correlation.py`: Contains the `analyze_correlation` class for performing correlation analysis, including stationarity tests, VAR modeling, and Granger causality tests.

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

### analyze_correlation

The `analyze_correlation` class in `correlation.py` is used for correlation analysis of time series data. Below is an example of how to use it:

```python
import pandas as pd
from correlation import analyze_correlation

# Load your data into x (features) and y (target)
x = pd.DataFrame(...)  # Replace with your features DataFrame
y = pd.Series(...)  # Replace with your target Series

# Initialize the analyzer
analyzer = analyze_correlation(x, y, verbose=True)

# Run the analysis
results = analyzer.analyze()

# Print the results
print(results)
```

## Dependencies

- pandas
- numpy
- statsmodels
- scikit-learn
- arch
- matplotlib (for plotting)

## Installation

To install the required dependencies, run:

```bash
pip install pandas numpy statsmodels scikit-learn arch matplotlib
```

## License

This project is licensed under the MIT License.
