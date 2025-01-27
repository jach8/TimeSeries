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
            'seasonal': {'period': 12}
        }
    verbose : bool
        Whether to print detailed test results
    """
    
    def __init__(self, test_config=None, verbose=False):
        self.default_config = {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'seasonal': None,
            'structural_break': False,
            'gls': False,
            'nonlinear': True
        }
        self.test_config = test_config or self.default_config
        self.verbose = verbose
        self._test_history = []
        warnings.filterwarnings("ignore")
        
    def _kss_test(self, series, alpha=0.05):
        """
        Custom implementation of Kapetanios-Snell-Shin nonlinear stationarity test
        References:
        - Kapetanios, G., Shin, Y., & Snell, A. (2003). Testing for a unit root 
          in the nonlinear STAR framework. Journal of Econometrics, 112(2), 359-379.
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
            'test': 'KSS (custom)',
            'statistic': t_stat,
            'p': pval,
            'critical_values': cv,
            'stationary': t_stat < cv['5%'],
            'alpha': alpha,
            'warning': 'Critical values approximated for T=100, use bootstrap for exact values'
        }

    def _run_test_battery(self, series):
        """Execute all configured stationarity tests"""
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

        # Seasonal tests
        if self.test_config.get('seasonal'):
            period = self.test_config['seasonal'].get('period', 12)
            results.update(self.seasonal_tests(series, period))

        # Advanced tests
        if self.test_config.get('gls'):
            results['dfgls'] = self.dfgls_test(series)
            
        if self.test_config.get('nonlinear'):
            results['kss'] = self._kss_test(series)

        return results

    @staticmethod
    def adf_test(series, alpha=0.05):
        """Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        return {
            'p': result[1],
            'stationary': result[1] < alpha,
            'test': 'ADF',
            'alpha': alpha
        }

    @staticmethod
    def kpss_test(series, alpha=0.05, regression='c'):
        """Kwiatkowski-Phillips-Schmidt-Shin test"""
        result = kpss(series, regression=regression)
        return {
            'p': result[1],
            'stationary': result[1] > alpha,  # KPSS has inverse logic
            'test': 'KPSS',
            'alpha': alpha
        }

    def phillips_perron_test(self, series, alpha=0.05):
        """Phillips-Perron test"""
        result = PhillipsPerron(series.dropna())
        return {
            'p': result.pvalue,
            'stationary': result.pvalue < alpha,
            'test': 'Phillips-Perron',
            'alpha': alpha
        }

    def zivot_andrews_test(self, series, alpha=0.05):
        """Zivot-Andrews structural break test"""
        result = ZivotAndrews(series.dropna())
        return {
            'p': result.pvalue,
            'stationary': result.pvalue < alpha,
            'breakpoint': result.break_date,
            'test': 'Zivot-Andrews',
            'alpha': alpha
        }

    def dfgls_test(self, series, alpha=0.05):
        """Elliott-Rothenberg-Stock GLS detrended test"""
        result = DFGLS(series.dropna())
        return {
            'p': result.pvalue,
            'stationary': result.pvalue < alpha,
            'test': 'DFGLS',
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
        
        if adf and kpss:
            return True
        elif not adf and not kpss:
            return False
        else:
            # Handle conflicting results
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
            stationary_df[col+f'_{diff_count}'] = current_series
            # If non-stationary drop column 
            if self._is_stationary(test_results):
                stationary_df = stationary_df.drop(columns = [col])
            full_results[col] = col_results
            report[col] = {
                'diffs_applied': diff_count,
                'final_status': 'stationary' if diff_count < self.test_config['adf']['max_diff'] else 'non-stationary'
            }

            if self.verbose:
                print(f"{col}: {diff_count} differences applied")
                print("Last test results:", {k:v for k,v in test_results.items() if k not in ['seasonal_decomp', 'canova_hansen']})
        
        stationary_df = stationary_df.dropna()
        return stationary_df, report, full_results


if __name__ == "__main__":
    print(""" 7.4: Earth, water, fire, air, ether, mind, spiritual intelligence and false ego; thus these are the eightfold divisions of my external energy.\n""")
    ###########################################################
    ########################################################### 
    # data = pd.read_csv('examples/data/stock_returns.csv', parse_dates=['Date'], index_col='Date').iloc[1:].cumsum()
    # data = data["2000-01-01":].dropna(axis = 1)
    # random_20_stocks = np.random.choice(data.columns, 60, replace = False)
    # random_y = np.random.choice(random_20_stocks, 1, replace = False)[0]
    # x = data.drop(columns = random_y).iloc[:-1]
    # y = data[random_y].iloc[:-1]
    # print('\n\n', y.name, '\n\n')
    
    # st = StationaryTests(verbose=False)
    # x, report, results = st.check_stationarity(x)
    # print(report)
    
    data = pd.read_csv("examples/data/hdd.csv", parse_dates=['date'], index_col='date')
    # data['target'] = data['target'].shift(-1)
    x = data.drop(columns='target')
    # y = data['target']
    sc = StationaryTests(verbose=True)
    df, report, summary = sc.check_stationarity(x)