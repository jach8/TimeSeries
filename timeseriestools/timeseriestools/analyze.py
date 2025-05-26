"""High-level interface for time series analysis."""

from typing import Dict, Optional, Any

import pandas as pd
from pandas import DataFrame, Series

from .correlation import AnalyzeCorrelation
from .causality import CausalityAnalyzer
from .stationarity import StationaryTests


class Analyze:
    """High-level interface for time series analysis.
    
    This class provides a unified interface to access all major functionality:
    - Correlation analysis
    - Causality testing
    - Stationarity checks
    
    Args:
        verbose: Print detailed output
        stationarity_config: Configuration for stationarity tests
        causality_config: Configuration for causality tests
    
    Example:
        >>> analyzer = Analyze(verbose=True)
        >>> results = analyzer.analyze_correlation(X, y)
        >>> print(results['stationarity_report'])
    """
    
    def __init__(
        self,
        verbose: bool = False,
        stationarity_config: Optional[Dict[str, Any]] = None,
        causality_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize analyzer with optional configurations."""
        self.verbose = verbose
        self.stationary_config = stationarity_config or {}
        self.causality_config = causality_config or {}

    def analyze_correlation(
        self,
        x: DataFrame,
        y: Series,
        decompose: bool = False
    ) -> Dict[str, Any]:
        """Run complete correlation analysis pipeline.
        
        Args:
            x: Feature DataFrame
            y: Target Series
            decompose: Whether to use PCA decomposition
            
        Returns:
            Dictionary containing:
            - stationarity_report: Stationarity test results
            - var_model: Fitted VAR model
            - causality: Causality test results
            - new_data: Processed stationary data
        """
        analyzer = AnalyzeCorrelation(
            x=x,
            y=y,
            decompose=decompose,
            verbose=self.verbose,
            stationarity_config=self.stationary_config,
            causality_config=self.causality_config
        )
        return analyzer.analyze_relationships()

    def check_stationarity(
        self,
        data: DataFrame
    ) -> Dict[str, Any]:
        """Run stationarity tests on data.
        
        Args:
            data: Time series data to test
            
        Returns:
            Dictionary with test results for each series
        """
        tester = StationaryTests(
            test_config=self.stationary_config,
            verbose=self.verbose
        )
        df, report, results = tester.check_stationarity(data)
        return {
            'stationary_data': df,
            'report': report,
            'test_results': results
        }

    def test_causality(
        self,
        data: DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """Run causality tests.
        
        Args:
            data: Time series data
            target: Target column name
            
        Returns:
            Dictionary with causality test results
        """
        analyzer = CausalityAnalyzer(
            causality_config=self.causality_config,
            verbose=self.verbose
        )
        return analyzer.causality_tests(data, target)