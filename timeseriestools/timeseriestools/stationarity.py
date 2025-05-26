"""Module for comprehensive stationarity testing of time series data."""

import logging
import warnings
from typing import Dict, Optional, List, Union, Tuple, Any, cast

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS
from pmdarima.arima import CHTest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.regression.linear_model import OLS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StationaryTests:
    """Comprehensive stationarity testing with multiple diagnostic methods."""
    
    def __init__(self, test_config: Optional[Dict[str, Any]] = None, verbose: bool = False) -> None:
        """Initialize StationaryTests with configuration.
        
        Args:
            test_config: Configuration dictionary for tests
            verbose: Whether to print detailed output
        """
        self.default_config = {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'structural_break': False,
            'gls': True,
            'nonlinear': True
        }
        self.test_config = test_config or self.default_config
        self.verbose = verbose
        self._test_history: List[Dict[str, Any]] = []
        warnings.filterwarnings("ignore")

    def check_stationarity(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Union[int, str]]], Dict[str, List[Dict[str, Any]]]]:
        """Run comprehensive stationarity checks on all columns.
        
        Args:
            df: Input DataFrame with time series data
            
        Returns:
            Tuple containing:
                - Transformed stationary DataFrame
                - Differencing report
                - Full test results
                
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        logger.info(f"Starting stationarity check for DataFrame with columns: {df.columns.tolist()}")
        
        stationary_df = df.copy()
        report: Dict[str, Dict[str, Union[int, str]]] = {}
        full_results: Dict[str, List[Dict[str, Any]]] = {}
        
        pbar = tqdm(df.columns, desc="Stationarity Check")
        max_diff = cast(int, self.test_config['adf']['max_diff'])
        
        for col in pbar:
            pbar.set_description(f"Checking {col}")
            current_series = df[col].copy()
            diff_count = 0
            col_results: List[Dict[str, Any]] = []
            
            for _ in range(max_diff + 1):
                test_results = self._run_test_battery(current_series)
                col_results.append(test_results)
                
                if self._is_stationary(test_results):
                    break
                    
                current_series = current_series.diff().dropna()
                diff_count += 1
            
            stationary_df[col] = current_series
            full_results[col] = col_results
            report[col] = {
                'diffs_applied': diff_count,
                'final_status': 'stationary' if diff_count < max_diff else 'non-stationary'
            }
            
        # Align all series by dropping NaN values from differencing
        max_diff_applied = max(v['diffs_applied'] for v in report.values())
        stationary_df = stationary_df.iloc[max_diff_applied:].copy()
        
        logger.info("Stationarity check completed")
        return stationary_df, report, full_results

    def _run_test_battery(self, series: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Execute configured stationarity tests.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary containing results of all configured tests
        """
        try:
            results: Dict[str, Dict[str, Any]] = {}
            
            if 'adf' in self.test_config:
                results['adf'] = self.adf_test(series, self.test_config['adf']['significance'])
            
            if 'kpss' in self.test_config:
                results['kpss'] = self.kpss_test(series, self.test_config['kpss']['significance'])

            if self.test_config.get('structural_break'):
                results['zivot_andrews'] = self.zivot_andrews_test(series)
                
            if self.test_config.get('gls'):
                results['dfgls'] = self.dfgls_test(series)
                
            if self.test_config.get('nonlinear'):
                results['kss'] = self._kss_test(series)
                
            return results
        except Exception as e:
            logger.error(f"Test battery failed: {str(e)}")
            raise

    def _kss_test(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Implement Kapetanios-Snell-Shin nonlinear test."""
        try:
            # Convert to numpy array and ensure 1D
            y = np.asarray(series.dropna())
            
            # Create lag matrix and get first lag
            y_lag = np.asarray(pd.Series(y).shift(1).dropna())
            delta_y = np.asarray(pd.Series(y).diff().dropna())
            
            # KSS test equation: Δy_t = ρ * y_{t-1}^3 + error
            X = y_lag**3
            X = X - X.mean()  # Demean for stability
            model = OLS(delta_y, X[:, np.newaxis])
            results = model.fit()
            
            # Calculate test statistic
            t_stat = results.tvalues[0]
            
            # Critical values from KSS (2003) Table 1
            cv = {
                '1%': -3.48,
                '5%': -2.93,
                '10%': -2.66
            }
            
            # Determine p-value
            if t_stat < cv['1%']:
                pval = 0.01
            elif t_stat < cv['5%']:
                pval = 0.05
            elif t_stat < cv['10%']:
                pval = 0.10
            else:
                pval = 0.50
                
            return {
                'test': 'Non-Linear Stationarity test',
                'statistic': float(t_stat),
                'stationary': bool(t_stat < cv['5%']),
                'p': float(pval),
                'alpha': float(alpha)
            }
        except Exception as e:
            logger.error(f"KSS test failed: {str(e)}")
            raise

    def _is_stationary(self, test_results: Dict[str, Dict[str, Any]]) -> bool:
        """Combine results from multiple tests to determine stationarity."""
        try:
            tests = {
                'adf': test_results.get('adf', {}).get('stationary', False),
                'kpss': test_results.get('kpss', {}).get('stationary', False),
                'zivot': test_results.get('zivot_andrews', {}).get('stationary', False),
                'dfgls': test_results.get('dfgls', {}).get('stationary', False),
                'kss': test_results.get('kss', {}).get('stationary', False)
            }

            majority = sum(bool(v) for v in tests.values()) >= 3
            
            if not majority:
                test_results['summary'] = {'conflicting_results': True}
                if self.verbose:
                    logger.warning("Conflicting test results - applying conservative differencing")
            return majority
        except Exception as e:
            logger.error(f"Error in _is_stationary: {str(e)}")
            raise

    @staticmethod
    def adf_test(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Augmented Dickey-Fuller test."""
        try:
            result = adfuller(series.dropna())
            return {
                'test': 'ADF',
                'statistic': float(result[0]),
                'stationary': bool(result[1] < alpha),
                'p': float(result[1]),
                'alpha': float(alpha)
            }
        except Exception as e:
            logger.error(f"ADF test failed: {str(e)}")
            raise

    @staticmethod
    def kpss_test(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform KPSS test."""
        try:
            result = kpss(series.dropna())
            return {
                'test': 'KPSS',
                'statistic': float(result[0]),
                'stationary': bool(result[1] > alpha),
                'p': float(result[1]),
                'alpha': float(alpha)
            }
        except Exception as e:
            logger.error(f"KPSS test failed: {str(e)}")
            raise

    def zivot_andrews_test(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Zivot-Andrews structural break test."""
        try:
            result = ZivotAndrews(series.dropna())
            return {
                'test': 'Zivot-Andrews',
                'statistic': float(result.stat),
                'stationary': bool(float(result.pvalue) < alpha),
                'p': float(result.pvalue),
                'alpha': float(alpha)
            }
        except Exception as e:
            logger.error(f"Zivot-Andrews test failed: {str(e)}")
            return {
                'test': 'Zivot-Andrews',
                'statistic': 0.0,
                'stationary': False,
                'p': 1.0,
                'alpha': float(alpha)
            }

    def dfgls_test(self, series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Elliott-Rothenberg-Stock GLS test."""
        try:
            result = DFGLS(series.dropna())
            return {
                'test': 'DFGLS',
                'statistic': float(result.stat),
                'p': float(result.pvalue),
                'stationary': bool(result.pvalue < alpha),
                'alpha': float(alpha)
            }
        except Exception as e:
            logger.error(f"DFGLS test failed: {str(e)}")
            raise