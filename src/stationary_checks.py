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
    
    def __init__(self, test_config=None, verbose=False):
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
        
    def _kss_test(self, series, alpha=0.05):
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

    def _run_test_battery(self, series):
        """
        
        Execute all configured stationarity tests
        
        Parameters:
            - series: pd.Series: The time series to test
            
        Returns:
            - dict: Results of all tests. Example:
                {
                    'adf': {'p': 0.05, 'stationary': False, 'test': 'ADF', 'alpha': 0.05},
                    'kpss': {'p': 0.05, 'stationary': False, 'test': 'KPSS', 'alpha': 0.05}
                }
        
        """
        results = {}
        
        # Core tests
        if 'adf' in self.test_config:
            results['adf'] = self.adf_test(
                series, 
                self.test_config['adf']['significance']
            )
            
        if 'kpss' in self.test_config:
            results['kpss'] = self.kpss_test(
                series,
                self.test_config['kpss']['significance']
            )

        # Structural break test
        if self.test_config.get('structural_break'):
            results['zivot_andrews'] = self.zivot_andrews_test(series)
            
        # Advanced tests
        if self.test_config.get('gls'):
            results['dfgls'] = self.dfgls_test(series)
            
        if self.test_config.get('nonlinear'):
            results['kss'] = self._kss_test(series)

        return results

    @staticmethod
    def adf_test(series, alpha=0.05):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Null Hypothesis (H0): The time series is Trend Stationary
        Alternate Hypothesis (H1): The time series is Non-Stationary (has a unit root)

        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'ADF',
                    'alpha': 0.05
                }
        """
        result = adfuller(series.dropna())
        return {
            'test': 'Stationarity (ADF)',
            'stationary': result[1] < alpha,
            'p': result[1],
            'alpha': alpha
        }

    @staticmethod
    def kpss_test(series, alpha=0.05, regression='c'):
        """
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    
        - Used for testing if an observable time series is stationary around a determinstic trend. 
        - The absense of a unit root in a time series indicates a trend-stationary process.
        - This means that the mean of the series can be growing or decreasing over time. 
        - In a presnece of a shock, trend stationary process are mean-reverting. 

        Null Hypothesis (H0): The process is trend-stationary.
        Alternative Hypothesis (H1): The process has a unit root (non-stationary).

        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
            - regression: str: The type of trend component to include in the test
                - 'c': Constant term only
                - 'ct': Constant and trend
                - 'ctt': Constant, trend, and quadratic trend
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'KPSS',
                    'alpha': 0.05
                }
    
        """
        result = kpss(series, regression=regression)
        return {
            'test': 'Trend Stationarity',
            'stationary': result[1] > alpha,  # KPSS has inverse logic
            'p': result[1],
            'alpha': alpha
        }

    def phillips_perron_test(self, series, alpha=0.05):
        """
        Phillips-Perron test
        A unit root test used to test the the null hypothesis that 
        
        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series has no unit root (Stationary).
        
        Fail to Reject the null hypothesis if the p-value is less than the significance level.
        
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'Phillips-Perron',
                    'alpha': 0.05
        }
    
        """
        result = PhillipsPerron(series.dropna())
        return {
            'test': 'Unit Root',
            'stationary': result.pvalue < alpha,
            'p': result.pvalue,
            'alpha': alpha
        }

    def zivot_andrews_test(self, series, alpha=0.05):
        """
        Wrapper for arch.unitroot.ZivotAndrews: 
        https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.ZivotAndrews.html
        
        Zivot-Andrews structural break test
        Algorithm follows Baum (2004/2015) approximation to original Zivot-Andrews method. 
        Rather than performing an autolag regression at each candidate break period (as per the original paper), 
        a single autolag regression is run up-front on the base model (constant + trend with no dummies) 
        to determine the best lag length. This lag length is then used for all subsequent break-period regressions. 
        This results in significant run time reduction but also slightly more 
        pessimistic test statistics than the original Zivot-Andrews method.
    
        Null Hypothesis (H0): The process contains a unit root with a single structural break.
        Alternate Hypothesis (H1): The process is trend and break stationary.
        
        Accept the null hypothesis if the p-value is less than the significance level.
        
        Parameters:
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'Zivot-Andrews',
                    'alpha': 0.05
                }
        """
        result = ZivotAndrews(series)
        return {
            'test': 'Structural Break',
            'stationary': float(result.pvalue) < alpha,
            'p': result.pvalue,
            'alpha': alpha
        }

    def dfgls_test(self, series, alpha=0.05):
        """
        Elliott-Rothenberg-Stock GLS detrended test
        The null hypothesis of the Dickey-Fuller GLS is that there is a unit root, 
        with the alternative that there is no unit root. If the pvalue is above a critical size, 
        then the null cannot be rejected and the series appears to be a unit root.
        
        Null Hypothesis (H0): The time series has a unit root (Non-Stationary).
        Alternate Hypothesis (H1): The time series has no unit root (Weakly Stationary).
        
        Parameters: 
            - series: pd.Series: The time series to test
            - alpha: float: The significance level
        
        Returns:
            - dict: Test results. Example:
                {
                    'p': 0.05,
                    'stationary': False,
                    'test': 'DFGLS',
                    'alpha': 0.05
                }
        """
        result = DFGLS(series.dropna())
        return {
            'test': 'Unit Root',
            'p': result.pvalue,
            'stationary': result.pvalue < alpha,
            'alpha': alpha
        }

    def kapetanios_test(self, series, alpha=0.05):
        """Kapetanios-Snell-Shin nonlinear test"""
        result = _kss_test(series.dropna())
        return {
            'p': result.pvalue,
            'stationary': result.pvalue < alpha,
            'test': 'KSS',
            'alpha': alpha
        }
     
    def seasonal_tests(self, series, period=12):
        """Seasonal diagnostics package"""
        results = {}
        
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
        
        return results

    def _is_stationary(self, test_results):
        """Decision logic combining multiple test results"""
        # Implement your decision logic here
        # Example: Require both ADF and KPSS agreement
        adf = test_results.get('adf', {}).get('stationary', False)
        kpss = test_results.get('kpss', {}).get('stationary', False)
        zivot = test_results.get('zivot_andrews', {}).get('stationary', False)
        dfgls = test_results.get('dfgls', {}).get('stationary', False)
        kss = test_results.get('kss', {}).get('stationary', False)

        majority = sum([adf, kpss, zivot, dfgls, kss]) >= 3
        
        # if adf and kpss:
        #     return True
        # elif not adf and not kpss:
        #     return False
        if majority:
            return True
        else:
            # Handle conflicting results
            test_results['Conflicting'] = True
            if self.verbose:
                print("Conflicting test results - applying conservative differencing")
            return False

    def check_stationarity(self, df):
        """
        Enhanced stationarity check with multiple diagnostics
        Returns:
        - Stationary DataFrame
        - Differencing report
        - Full test results
        """
        stationary_df = df.copy()
        report = {}
        full_results = {}
        pbar = tqdm(df.columns, desc="Stationarity Check")
        for col in pbar:
            pbar.set_description(f"Stationarity {col}")
            current_series = df[col]
            diff_count = 0
            col_results = []

            for _ in range(self.test_config['adf']['max_diff'] + 1):
                test_results = self._run_test_battery(current_series.dropna())
                col_results.append(test_results)
                
                if self._is_stationary(test_results):
                    break
                    
                # Apply differencing
                current_series = current_series.diff()
                diff_count += 1

            # Store results with a new column name
            # stationary_df[col+f'_{diff_count}'] = current_series
            stationary_df[col] = current_series
            # If non-stationary drop column 
            # if self._is_stationary(test_results): stationary_df = stationary_df.drop(columns = [col])
            full_results[col] = col_results
            report[col] = {
                'diffs_applied': diff_count,
                'final_status': 'stationary' if diff_count < self.test_config['adf']['max_diff'] else 'non-stationary'
            }
            
            if self.verbose:
                print(f"{col}: {diff_count} differences applied")
                print("Last test results:", {k:v for k,v in test_results.items() if k not in ['seasonal_decomp', 'canova_hansen']})
        
        max_diff = max([v['diffs_applied'] for v in report.values()])
        
        stationary_df = stationary_df.iloc[max_diff:]
        return stationary_df, report, full_results


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