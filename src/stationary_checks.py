


import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS
from pmdarima.arima import CHTest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat
import warnings
from typing import Dict, Optional, List, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StationaryTests:
    """
    Comprehensive stationarity testing with multiple diagnostic methods.
    
    Parameters:
    -----------
    test_config : dict
        Configuration for tests to perform. Example:
        {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'pp': {'significance': 0.05},
            'structural_break': True,
            'seasonal': {'period': 12}, 
            'gls': False,
            'nonlinear': True
        }
    verbose : bool
        Whether to print detailed test results
    """
    
    def __init__(self, test_config: Optional[Dict] = None, verbose: bool = False):
        self.default_config = {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'structural_break': False,
            'gls': True,
            'nonlinear': True
        }
        self.test_config = test_config or self.default_config
        self.verbose = verbose
        self._test_history = []
        warnings.filterwarnings("ignore")
        
    def _kss_test(self, series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Custom implementation of Kapetanios-Snell-Shin nonlinear stationarity test
        From Kapetanios et al. (2003):
           "a simple testing procedure to detect the presence of 
            nonstationarity against nonlinear but globally stationary exponential 
            smooth transition autoregressive processes."
           
        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series has no unit root (Stationary).
        
        References:
        - Kapetanios, G., Shin, Y., & Snell, A. (2003). Testing for a unit root 
          in the nonlinear STAR framework. Journal of Econometrics, 112(2), 359-379.
          
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'KSS (Custom)',
                    'alpha': 0.05
                }
        """
        y = series.dropna().values
        n = len(y)
        y_lag = lagmat(y, maxlag=1, trim='both')[:,0]
        delta_y = np.diff(y)
        
        # KSS test equation: Δy_t = ρ * y_{t-1}^3 + error
        X = y_lag**3
        X = X - X.mean()  # Demean for stability
        model = OLS(delta_y, X)
        results = model.fit()
        
        # Calculate test statistic
        t_stat = results.tvalues[0]
        
        # Critical values from KSS (2003) Table 1 (T=100)
        # These are approximate - for precise values use response surface
        cv = {
            '1%': -3.48,
            '5%': -2.93,
            '10%': -2.66
        }
        
        # Determine p-value using cubic interpolation between critical values
        if t_stat < cv['1%']:
            pval = 0.01
        elif t_stat < cv['5%']:
            pval = 0.05
        elif t_stat < cv['10%']:
            pval = 0.10
        else:
            pval = 0.50  # Conservative upper bound
            
        return {
            'test': 'Non-Linear Stationarity test',
            'stationary': t_stat < cv['5%'],
            'p': pval,
            'alpha': alpha
        }

    def _run_test_battery(self, series: pd.Series) -> Dict[str, Dict]:
        """
        Execute all configured stationarity tests.
        
        Parameters:
            - series: pd.Series: The time series to test.
            
        Returns:
            - Dict[str, Dict]: Results of all tests. Example:
                {
                    'adf': {'p': 0.05, 'stationary': False, 'test': 'ADF', 'alpha': 0.05},
                    'kpss': {'p': 0.05, 'stationary': False, 'test': 'KPSS', 'alpha': 0.05}
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")

            results: Dict[str, Dict] = {}
            
            # Core tests
            if 'adf' in self.test_config:
                adf_config = self.test_config['adf']
                results['adf'] = self.adf_test(series, adf_config['significance'])
            
            if 'kpss' in self.test_config:
                kpss_config = self.test_config['kpss']
                results['kpss'] = self.kpss_test(series, kpss_config['significance'])

            # Structural break test
            if self.test_config.get('structural_break'):
                results['zivot_andrews'] = self.zivot_andrews_test(series)
            
            if self.test_config.get('pp'):
                pp_config = self.test_config['pp']
                results['pp'] = self.phillips_perron_test(series, pp_config['significance'])
                
            # Advanced tests
            if self.test_config.get('gls'):
                results['dfgls'] = self.dfgls_test(series)
                
            if self.test_config.get('nonlinear'):
                results['kss'] = self._kss_test(series)

            logger.info("Test battery executed for series: %s", series.name)
            return results
        except Exception as e:
            logger.error("Error in _run_test_battery: %s", e)
            raise

    @staticmethod
    def adf_test(series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series is stationary.

        Parameters:
            - series: pd.Series: The time series to test.
            - alpha: float: The significance level.
        
        Returns:
            - Dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'ADF',
                    'alpha': 0.05
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")

            result = adfuller(series.dropna())
            test_result = {
                'test': 'Stationarity (ADF)',
                'stationary': result[1] < alpha,
                'p': result[1],
                'alpha': alpha
            }
            logger.info("ADF test completed: %s", test_result)
            return test_result
        except Exception as e:
            logger.error("Error in ADF test: %s", e)
            raise

    @staticmethod
    def kpss_test(series: pd.Series, alpha: float = 0.05, regression: str = 'c') -> Dict:
        """
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

        Null Hypothesis (H0): The process is trend-stationary.
        Alternative Hypothesis (H1): The process has a unit root (non-stationary).

        Parameters:
            - series: pd.Series: The time series to test.
            - alpha: float: The significance level.
            - regression: str: The type of trend component to include in the test.
                - 'c': Constant term only.
                - 'ct': Constant and trend.
                - 'ctt': Constant, trend, and quadratic trend.
        
        Returns:
            - Dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'KPSS',
                    'alpha': 0.05
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")
            if regression not in ['c', 'ct', 'ctt']:
                raise ValueError("regression must be one of ['c', 'ct', 'ctt'].")

            result = kpss(series, regression=regression)
            test_result = {
                'test': 'Trend Stationarity (KPSS)',
                'stationary': result[1] > alpha,  # KPSS has inverse logic
                'p': result[1],
                'alpha': alpha
            }
            logger.info("KPSS test completed: %s", test_result)
            return test_result
        except Exception as e:
            logger.error("Error in KPSS test: %s", e)
            raise

    def phillips_perron_test(self, series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Phillips-Perron test.

        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series has no unit root (Stationary).
        
        Parameters:
            - series: pd.Series: The time series to test.
            - alpha: float: The significance level.
        
        Returns:
            - Dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'Phillips-Perron',
                    'alpha': 0.05
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")

            result = PhillipsPerron(series.dropna())
            test_result = {
                'test': 'Unit Root (Phillips-Perron)',
                'stationary': result.pvalue < alpha,
                'p': result.pvalue,
                'alpha': alpha
            }
            logger.info("Phillips-Perron test completed: %s", test_result)
            return test_result
        except Exception as e:
            logger.error("Error in Phillips-Perron test: %s", e)
            raise

    def zivot_andrews_test(self, series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Wrapper for arch.unitroot.ZivotAndrews:
        https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.ZivotAndrews.html
        
        Zivot-Andrews structural break test.
        
        Null Hypothesis (H0): The process contains a unit root with a single structural break.
        Alternate Hypothesis (H1): The process is trend and break stationary.
        
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - Dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'Zivot-Andrews',
                    'alpha': 0.05
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")

            result = ZivotAndrews(series)
            out = {
                'test': 'Structural Break',
                'stationary': float(result.pvalue) < alpha,
                'p': float(result.pvalue),
                'alpha': alpha
            }
            logger.info("Zivot-Andrews test completed: %s", out)
            return out
        except Exception as e:
            logger.error("Error in Zivot-Andrews test: %s", e)
            return {
                'test': 'Structural Break',
                'stationary': False,
                'p': 1.0,
                'alpha': alpha
            }

    def dfgls_test(self, series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Elliott-Rothenberg-Stock GLS detrended test.
        
        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series has no unit root (Weakly Stationary).
        
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - Dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'DFGLS',
                    'alpha': 0.05
                }
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")

            result = DFGLS(series.dropna())
            out = {
                'test': 'Unit Root (DFGLS)',
                'p': float(result.pvalue),
                'stationary': result.pvalue < alpha,
                'alpha': alpha
            }
            logger.info("DFGLS test completed: %s", out)
            return out
        except Exception as e:
            logger.error("Error in DFGLS test: %s", e)
            raise

    def kapetanios_test(self, series: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Kapetanios-Snell-Shin nonlinear test.
        
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - Dict: Test results.
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
                raise ValueError("alpha must be a float between 0 and 1.")

            result = self._kss_test(series.dropna(), alpha)
            out = {
                'p': result['p'],
                'stationary': result['stationary'],
                'test': 'KSS',
                'alpha': alpha
            }
            logger.info("Kapetanios test completed: %s", out)
            return out
        except Exception as e:
            logger.error("Error in Kapetanios test: %s", e)
            raise

    def seasonal_tests(self, series: pd.Series, period: int = 12) -> Dict:
        """
        Seasonal diagnostics package.
        
        Parameters:
            - series: pd.Series: The time series to test
            - period: int: The period for seasonal decomposition
        
        Returns:
            - Dict: Seasonal test results.
        """
        try:
            if not isinstance(series, pd.Series):
                raise ValueError("Input 'series' must be a pandas Series.")
            if series.empty:
                raise ValueError("Input series is empty.")
            if not isinstance(period, int) or period <= 0:
                raise ValueError("period must be a positive integer.")

            results: Dict = {}
            
            # Canova-Hansen test
            ch_result = CHTest().is_stationary(series)
            results['canova_hansen'] = {
                'stationary': ch_result,
                'test': 'Canova-Hansen',
                'period': period
            }

            # Seasonal decomposition
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomp = seasonal_decompose(series.dropna(), period=period)
                seasonal_stationary = self.adf_test(decomp.seasonal)
                
            results['seasonal_decomp'] = {
                'seasonal_stationary': seasonal_stationary['stationary'],
                'residual_stationary': self.adf_test(decomp.resid)['stationary'],
                'test': 'Seasonal Decomposition'
            }
            
            logger.info("Seasonal tests completed for series: %s", series.name)
            return results
        except Exception as e:
            logger.error("Error in seasonal_tests: %s", e)
            raise

    def _is_stationary(self, test_results: Dict) -> bool:
        """
        Decision logic combining multiple test results.
        
        Parameters:
            - test_results: Dict: Results from stationarity tests.
        
        Returns:
            - bool: Whether the series is considered stationary.
        """
        try:
            adf = test_results.get('adf', {}).get('stationary', False)
            kpss = test_results.get('kpss', {}).get('stationary', False)
            zivot = test_results.get('zivot_andrews', {}).get('stationary', False)
            dfgls = test_results.get('dfgls', {}).get('stationary', False)
            kss = test_results.get('kss', {}).get('stationary', False)

            majority = sum([adf, kpss, zivot, dfgls, kss]) >= 3
            
            if majority:
                return True
            else:
                test_results['Conflicting'] = True
                if self.verbose:
                    logger.warning("Conflicting test results - applying conservative differencing")
                return False
        except Exception as e:
            logger.error("Error in _is_stationary: %s", e)
            raise

    def check_stationarity(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Enhanced stationarity check with multiple diagnostics.
        
        Parameters:
            - df: pd.DataFrame: Input DataFrame with time series columns.
        
        Returns:
            - Tuple[pd.DataFrame, Dict, Dict]: 
                - Stationary DataFrame
                - Differencing report
                - Full test results
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input 'df' must be a pandas DataFrame.")
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            if not all(isinstance(col, str) for col in df.columns):
                raise ValueError("All column names in the DataFrame must be strings.")
            if 'adf' not in self.test_config or 'max_diff' not in self.test_config['adf']:
                raise KeyError("test_config must contain 'adf' with 'max_diff' specified.")

            logger.info("Starting stationarity check for DataFrame with columns: %s", df.columns.tolist())

            stationary_df: pd.DataFrame = df.copy()
            report: Dict[str, Dict[str, Union[int, str]]] = {}
            full_results: Dict[str, List[Dict]] = {}
            pbar = tqdm(df.columns, desc="Stationarity Check")

            for col in pbar:
                pbar.set_description(f"Stationarity {col}")
                if col not in df.columns:
                    logger.warning("Column %s not found in DataFrame. Skipping.", col)
                    continue

                current_series: pd.Series = df[col]
                if current_series.empty:
                    logger.warning("Column %s is empty. Skipping.", col)
                    continue

                diff_count: int = 0
                col_results: List[Dict] = []

                for _ in range(self.test_config['adf']['max_diff'] + 1):
                    test_results = self._run_test_battery(current_series.dropna())
                    col_results.append(test_results)
                    
                    if self._is_stationary(test_results):
                        break
                    
                    # Apply differencing
                    current_series = current_series.diff()
                    diff_count += 1

                stationary_df[col] = current_series
                full_results[col] = col_results
                report[col] = {
                    'diffs_applied': diff_count,
                    'final_status': 'stationary' if diff_count < self.test_config['adf']['max_diff'] else 'non-stationary'
                }
                
                if self.verbose:
                    logger.info(f"{col}: {diff_count} differences applied")
                    logger.info("Last test results: %s", {k: v for k, v in test_results.items()})

            if not report:
                raise RuntimeError("No columns were processed. Check input data and configuration.")

            max_diff = max(v['diffs_applied'] for v in report.values())
            stationary_df = stationary_df.iloc[max_diff:]
            logger.info("Stationarity check completed. Max differences applied: %d", max_diff)
            return stationary_df, report, full_results
        except Exception as e:
            logger.error("Error in check_stationarity: %s", e)
            raise


if __name__ == "__main__":
    print(""" 10.11: Out of compassion for them, I situated within the heart, certainly destroy the darkness born of ignorance with the radiant light of knowledge. \n""")
    ###########################################################
    from data import *
    x, y = test_data1(return_xy=True, path_to_src='src/')
    print(x.head()) 
    ###########################################################
    config =   {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'pp': {'significance': 0.05},
            'structural_break': True,
            'gls': True,
            'nonlinear': True
        }
    
    sc = StationaryTests(config, verbose=False)
    stationary_df, report, full_results = sc.check_stationarity(x)
    
    for k, v in full_results.items():
        print(k)
        for j in v: 
            for i, m in j.items():
                print(i, m)
            print("\n")
    
    print(report)