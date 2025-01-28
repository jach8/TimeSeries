import pandas as pd
import numpy as np
import warnings
from io import StringIO
from statsmodels.tsa.vector_ar.var_model import VAR
from itertools import combinations
from tqdm import tqdm 

# Import granger causality tests
from src.granger import grangercausalitytests

class CausalityAnalyzer:
    """
    Handles different types of causality tests with consistent interface
    """
    def __init__(self, causality_config = None, verbose = False):
        self.default_config = {
            'significance_level': 0.05,
            'max_lag': 3
        }
        self.config = causality_config or self.default_config
        self.significance_level = self.config.get('significance_level', 0.05)
        self.max_lag = self.config.get('max_lag', 3)
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
            if x2 == target:
                out.append([x2,x1])
                
        return out
    def granger_test(self, data, target):
        """ run granger causality test on all columns in the data"""
        results = {}
        cp = self._get_column_pairs(data, target)
        pbar = tqdm(cp, desc="Granger Causality")
        for x1, x2 in pbar:
            pbar.set_description(f"Granger Causality: {x1} -> {x2}")
            d = self.__granger_causality(data[x1], data[x2])
            results[(x1, x2)] = d[(x1, x2)]
        pbar.close()
        return results
    
    def instantaneous_causality(self, fit, data, target):
        """
        Perform Instantaneous causality test (A form of Granger Causality test) from the Var Model Fit.
        Instantaneous causality reflects a non-zero correlation between the variables in the system. 
        
        Parameters:
            - fit: The VAR model fit
            - data: The data to analyze
            - target: The target variable
        
        Returns:
            - list: A list of tuples representing instantaneous causality

        """
        instaneous_cause = []
        features = [x for x in data.columns if x != target]
        for i in features:
            try:
                t = fit.test_inst_causality(causing=i).summary()
            except Exception as e:
                raise ValueError(f"{i} Error in Instaneous Causality test: {str(e)}")
                
            if t[1][2].data < self.significance_level:
                if t[1][2].data == 0:
                    # Raise warnings for perfect causality
                    warnings.warn(f"{i} Perfect Instaneous Causality detected for {target}.")
                instaneous_cause.append((target, i))
                if self.verbose:
                    confi = int(100 - (100*self.significance_level))
                    print(f'{i} has an instantaneous causal effect on {target} @ {confi}% confidence level, p-value: {t[1][2].data}')
                    
        return instaneous_cause
        
    
    def causality_tests(self, data, target, model = None):
        """
        Unified causality test interface
        Runs the granger causality test and the instantaneous causality test on the data
        Contemporary causality is not implemented yet.
        
        Arguments:
            - data: pd.DataFrame: The data to analyze
            - target: str: The target variable
            - model: The VAR model fit
        
        Returns:
            - dict: A dictionary of causality test results
            Example:
            {
                'granger': [(('x1', 'x2'), [1, 2, 3])],
                'instantaneous': [('x1', 'x2')],
                'contemporaneous': [('x1', 'x2')]
            }
        
        """
        results = {
            'granger': [],
            'instantaneous': [],
            # 'contemporaneous': []
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
            if self.verbose and not new_v.empty:
                print(f'{k[1]} Does Granger Cause {k[0]} @ {self.significance_level}% confidence level')
                print(new_v)
        print('\n')

        if model:
            # Instantaneous causality test
            results['instantaneous'] = self.instantaneous_causality(model, data, target)
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
    