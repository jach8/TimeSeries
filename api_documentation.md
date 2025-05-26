# TimeSeriesTools API Documentation

## Core Modules

### causality.py

```python
class CausalityAnalyzer:
    """Handles various types of causality tests between time series.
    
    This class implements multiple causality testing methods including:
    - Granger causality tests
    - Instantaneous causality tests
    - Impulse response analysis
    
    Args:
        causality_config (Dict[str, float], optional): Configuration parameters.
            - significance_level (float): P-value threshold (default: 0.05)
            - max_lag (int): Maximum number of lags to test (default: 3)
        verbose (bool, optional): Whether to print detailed output.
        
    Attributes:
        significance_level (float): Significance level for statistical tests
        max_lag (int): Maximum lag order for tests
        verbose (bool): Verbosity flag
        
    Example:
        >>> analyzer = CausalityAnalyzer(significance_level=0.05)
        >>> results = analyzer.granger_test(data, target='SPY')
    """

    def granger_test(self, data: pd.DataFrame, target: str) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Run Granger causality tests on all columns in relation to target.
        
        Args:
            data (pd.DataFrame): Input time series data
            target (str): Name of target variable to test causality against
            
        Returns:
            Dict[Tuple[str, str], pd.DataFrame]: Results of Granger causality tests
                Keys are tuples of (target, cause) variable names
                Values are DataFrames containing test statistics
        """
```

### correlation.py

```python
class AnalyzeCorrelation:
    """Comprehensive correlation analysis with stationarity checks.
    
    This class combines stationarity testing, correlation analysis, and 
    causality testing into a unified workflow.
    
    Args:
        x (pd.DataFrame): Input features dataframe
        y (pd.Series): Target variable series
        decompose (bool, optional): Whether to perform PCA decomposition
        verbose (bool, optional): Enable verbose output
        stationarity_config (Dict, optional): Configuration for stationarity tests
        causality_config (Dict, optional): Configuration for causality analysis
        
    Attributes:
        cause (str): Name of target variable
        features (List[str]): List of feature names
        df (pd.DataFrame): Combined analysis dataframe
        var_model (VARResults): Vector autoregression model results
        
    Example:
        >>> x, y = ts.test_data1(return_xy=True)
        >>> analysis = AnalyzeCorrelation(x, y, decompose=True)
        >>> results = analysis.analyze_relationships()
    """
```

### stationarity.py

```python
class StationaryTests:
    """Comprehensive stationarity testing with multiple diagnostic methods.
    
    Implements various stationarity tests including:
    - Augmented Dickey-Fuller (ADF)
    - KPSS test
    - Phillips-Perron test
    - Zivot-Andrews structural break test
    - Elliott-Rothenberg-Stock GLS test
    - Kapetanios-Snell-Shin nonlinear test
    
    Args:
        test_config (Dict, optional): Configuration for tests.
            Example:
            {
                'adf': {'max_diff': 5, 'significance': 0.05},
                'kpss': {'significance': 0.05},
                'pp': {'significance': 0.05},
                'structural_break': True,
                'gls': True,
                'nonlinear': True
            }
        verbose (bool, optional): Whether to print detailed test results
        
    Example:
        >>> tester = StationaryTests(test_config={'adf': {'max_diff': 3}})
        >>> df, report, results = tester.check_stationarity(data)
    """

    def check_stationarity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run comprehensive stationarity checks on all columns.
        
        Args:
            df (pd.DataFrame): Input time series data
            
        Returns:
            Tuple containing:
                pd.DataFrame: Transformed stationary data
                Dict: Differencing report
                Dict: Full test results
        """
```

### utils/granger.py

```python
def grangercausalitytests(x: np.ndarray, maxlag: int, addconst: bool = True) -> Dict:
    """Run Granger causality tests with multiple lag orders.
    
    Implements four tests for Granger non-causality:
    1. F-test based on sum of squared residuals
    2. Chi-squared test
    3. Likelihood ratio test  
    4. Parameter F-test
    
    Args:
        x (np.ndarray): 2-D array with time series in columns
        maxlag (int): Maximum lag order to test
        addconst (bool, optional): Include constant term
        
    Returns:
        Dict: Test results for each lag order
    
    Example:
        >>> x = np.random.randn(100, 2)
        >>> results = grangercausalitytests(x, maxlag=3)
    """
```

## Testing Guide

Each module has corresponding unit tests in the tests/ directory that demonstrate usage and verify functionality:

- test_causality.py: Tests for CausalityAnalyzer
- test_correlation.py: Tests for AnalyzeCorrelation
- test_stationarity.py: Tests for StationaryTests

Example test:
```python
def test_granger_causality():
    """Test Granger causality analysis with known relationship."""
    x = np.random.randn(100)
    y = np.roll(x, 1) + np.random.randn(100) * 0.1
    data = pd.DataFrame({'x': x, 'y': y})
    
    analyzer = CausalityAnalyzer()
    results = analyzer.granger_test(data, 'y')
    
    assert ('y', 'x') in results
    assert results[('y', 'x')]['ssr_ftest'][1] < 0.05