# Time Series Stationarity Tests

This document describes key statistical tests used to evaluate stationarity in time series data. Understanding these tests is crucial for proper time series analysis and modeling.

## Overview of Stationarity

A time series is considered stationary when its statistical properties (mean, variance) remain constant over time. This is a fundamental assumption for many time series models.

## Available Tests

### 1. Augmented Dickey-Fuller (ADF) Test

Tests for the presence of a unit root in a time series sample.

**Hypotheses:**
- Null Hypothesis (H0): The time series is non-stationary (has a unit root)
- Alternative Hypothesis (H1): The time series is stationary

**Interpretation:**
- If p-value < significance level (typically 0.05): Reject H0, series is stationary
- If p-value ≥ significance level: Fail to reject H0, series is non-stationary

### 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test

Tests whether a time series is stationary around a deterministic trend.

**Key Features:**
- Complements the ADF test by testing the null hypothesis of stationarity
- Useful for confirming results from other unit root tests
- Tests trend-stationarity vs. difference-stationarity

**Hypotheses:**
- Null Hypothesis (H0): The process is trend-stationary
- Alternative Hypothesis (H1): The process has a unit root (non-stationary)

**Properties:**
- Can detect if a series is trend-stationary or difference-stationary
- Mean can grow/decrease over time in a trend-stationary process
- Trend-stationary processes are mean-reverting after shocks

### 3. Zivot-Andrews Structural Break Test

Tests for a unit root in the presence of a potential structural break.

**Implementation Details:**
- Based on Baum (2004/2015) approximation
- Uses an optimized approach:
  1. Runs single autolag regression on base model (constant + trend)
  2. Determines optimal lag length
  3. Applies this lag length to all break-period regressions
- More computationally efficient than original method
- Slightly more conservative test statistics

**Hypotheses:**
- Null Hypothesis (H0): Process contains a unit root with a single structural break
- Alternative Hypothesis (H1): Process is trend and break stationary

### 4. Phillips-Perron (PP) Test

A non-parametric unit root test that controls for serial correlation.

**Key Features:**
- Handles serial correlation without adding lagged difference terms
- More robust to unspecified autocorrelation and heteroscedasticity

**Hypotheses:**
- Null Hypothesis (H0): The time series has a unit root (non-stationary)
- Alternative Hypothesis (H1): The time series is stationary

**Decision Rule:**
- Reject H0 if the test statistic is more negative than the critical value
- Consider significance level when evaluating p-value

### 5. Elliott-Rothenberg-Stock GLS Detrended Test

A more powerful variant of the ADF test using generalized least squares detrending.

**Key Features:**
- Higher power than standard ADF test
- Particularly useful for series with unknown deterministic components

**Hypotheses:**
- Null Hypothesis (H0): The time series has a unit root (non-stationary)
- Alternative Hypothesis (H1): The time series is stationary (no unit root)

**Decision Rule:**
- If p-value > critical size: Cannot reject null hypothesis
- Series likely contains a unit root

## Best Practices

1. Use multiple tests to confirm stationarity
2. Consider the nature of your data when selecting tests
3. Pay attention to:
   - Trend components
   - Seasonal patterns
   - Potential structural breaks

## References

1. Dickey, D.A. and Fuller, W.A. (1979) "Distribution of the Estimators for Autoregressive Time Series with a Unit Root"
2. Kwiatkowski et al. (1992) "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root"
3. Zivot, E. and Andrews, D.W.K. (1992) "Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis"
4. Phillips, P.C.B. and Perron, P. (1988) "Testing for a Unit Root in Time Series Regression"
5. Elliott, G., Rothenberg, T.J., and Stock, J.H. (1996) "Efficient Tests for an Autoregressive Unit Root"