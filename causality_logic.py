import pandas as pd
import numpy as np
import warnings
from io import StringIO
from statsmodels.tsa.vector_ar.var_model import VAR
from granger import grangercausalitytests
from itertools import combinations
from stationary_checks import StationaryTests  
from tqdm import tqdm 

class CausalityAnalyzer:
    """
    Handles different types of causality tests with consistent interface
    """
    def __init__(self, significance_level=0.05, max_lag=4, verbose = False):
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.verbose = verbose
        
    
    @staticmethod
    def __granger_causality(target, cause, maxlags = 3): 
        """
        Wrapper Method for statsmodels.tsa.stattools.grangercausalitytests: 
            - Only pass a dataframe or np.arry of 2 variables in the grangercausalitytests function.
            - Four tests for granger non-causality of 2 time series:
                1. 'ssr_ftest': F-Test ()
                2. 'ssr_chi2test': Chi-squared test
                3. 'lrtest': Likelihood ratio test
                4. 'params_ftest': Parameter F test
        
        To granger cause, means that the past values of x2 (the cause) have a statistically significant effect on the current value of x1 (the target). 
        
        
        - Taking past values of x1 into account as regressors. 
            If the p-values are below a threshold, we reject the null hypothesis that x2 does not granger cause x1. 
        
        - H0: The time series in the second column, x2, DOES NOT Granger Cause the time series in the first column, x1.    
        - H1: The time series in the second column, x2 does Granger cause the time series in the first column, x1.
        
        We want to dis-prove the null hypothess that x2 does not cause x1, ie. We are looking for small p-values. 
        
        
        Parameters:
            - target pd.Series: The target variable
            - cause: pd.Series: The cause variable
            - maxlags: The maximum number of lags to use in the test
        
        Returns: 
        
        """
        data = pd.concat([target, cause], axis=1)
        result = grangercausalitytests(data, maxlag=maxlags)
        lags = list(result.keys())
        res = []
        for lag in lags:
            res.append({
                'lag': lag,
                'ssr_ftest': result[lag][0]['ssr_ftest'][1],
                'ssr_chi2test': result[lag][0]['ssr_chi2test'][1],
                'lrtest': result[lag][0]['lrtest'][1],
                'params_ftest': result[lag][0]['params_ftest'][1]
            })
        return {(target.name, cause.name): pd.DataFrame(res).set_index('lag')}
        
    @staticmethod  
    def _get_column_pairs(data, target):
        """
        Helper method to get all possible column pairs pertaining to the target variable
        
        Parameters:
            - data: pd.DataFrame: The data to analyze
            - target: str: The target variable
        
        Returns:
            - list of lists: A list of column pairs
        
        """
        column_pairs = list(combinations(data.columns, 2))
        out = []
        for x1,x2 in column_pairs:
            if x1 == target: 
                out.append([x1,x2])
        return out
        
        
    def granger_test(self, data, target):
        """ run granger causality test on all columns in the data"""
        results = {}
        pbar = tqdm(self._get_column_pairs(data, target), desc="Granger Causality")
        for x1, x2 in pbar:
            pbar.set_description(f"Granger Causality: {x1} -> {x2}")
            d = self.__granger_causality(data[x1], data[x2])
            results[(x1, x2)] = d[(x1, x2)]
        pbar.close()
        return results
    
        
    def causality_tests(self, data, target):
        """Unified causality test interface"""
        results = {
            'granger': [],
            'instantaneous': [],
            'contemporaneous': []
        }
        
        granger_tests = self.granger_test(data, target)
        for k, v in granger_tests.items():
            # Check p-values:
            mask = v < self.significance_level
            # Only save if there is unanimous agreement for any of the lags
            mask_sums = mask.sum(axis=1)
            mask_sums = mask_sums[mask_sums == mask.shape[1]]
            new_v = v.loc[mask_sums.index]
            if not new_v.empty:
                results['granger'].append((k, new_v.index.values))
    
        return results


if __name__ == "__main__":
    print("\n6.30: For one who sees Me everywhere and sees everything in Me, I am never forgotten by them and they are never forgotten by Me.\n")
    data = pd.read_csv('examples/data/stock_returns.csv', index_col=0, parse_dates=['Date'])
    data = data.dropna(axis = 1).iloc[:, :]
    from stationary_checks import StationaryTests
    st = StationaryTests()
    df, report, summary = st.check_stationarity(data)
    
    ca = CausalityAnalyzer()
    res = ca.causality_tests(df, 'SPY')
    
    g = res['granger']
    print(g)
    