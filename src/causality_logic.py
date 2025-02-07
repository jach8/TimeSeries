import pandas as pd
import numpy as np
import warnings
import logging
from io import StringIO
from statsmodels.tsa.vector_ar.var_model import VAR
from itertools import combinations
from tqdm import tqdm 
from typing import Dict, List, Tuple, Optional, Union, Any
from statsmodels.tsa.vector_ar.var_model import VARResults

# Import granger causality tests
from src.granger import grangercausalitytests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CausalityAnalyzer:
    """
    Handles different types of causality tests with consistent interface
    
    default_config: Dict[str, float]
    config: Dict[str, float]
    significance_level: float
    max_lag: int
    verbose: bool
    """
    def __init__(self, causality_config: Optional[Dict[str, float]] = None, verbose: bool = False):
        self.default_config = {
            'significance_level': 0.05,
            'max_lag': 3
        }
        self.config = causality_config or self.default_config
        
        # Validate configuration values
        if not isinstance(self.config.get('significance_level', 0.05), (int, float)):
            raise ValueError("significance_level must be a number")
        if not isinstance(self.config.get('max_lag', 3), (int)):
            raise ValueError("max_lag must be an integer")
            
        self.significance_level = self.config.get('significance_level', 0.05)
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
            
        self.max_lag = self.config.get('max_lag', 3)
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")
            
        self.verbose = verbose
        logger.info(f"CausalityAnalyzer initialized with significance_level={self.significance_level}, max_lag={self.max_lag}")
    
    @staticmethod
    def __granger_causality(target: pd.Series, cause: pd.Series, maxlags: int = 3) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
            To granger cause, means that the past values of x2 (the cause) have a statistically significant effect on the current value of x1 (the target). 
            - Taking past values of x1 into account as regressors. 
                If the p-values are below a threshold, we reject the null hypothesis that x2 does not granger cause x1. 
            
            - H0: The time series in the second column, x2, DOES NOT Granger Cause the time series in the first column, x1.    
            - H1: The time series in the second column, x2 does Granger cause the time series in the first column, x1.
            
            We want to dis-prove the null hypothess that x2 does not cause x1, ie. We are looking for small p-values. 
        
        Wrapper Method for statsmodels.tsa.stattools.grangercausalitytests: 
            - Only pass a dataframe or np.arry of 2 variables in the grangercausalitytests function.
            - Four tests for granger non-causality of 2 time series:
                1. 'ssr_ftest': F-Test ()
                2. 'ssr_chi2test': Chi-squared test
                3. 'lrtest': Likelihood ratio test
                4. 'params_ftest': Parameter F test
        
        Parameters:
            - target pd.Series: The target variable
            - cause: pd.Series: The cause variable
            - maxlags: The maximum number of lags to use in the test
        
        Returns: 
            Dict[Tuple[str, str], pd.DataFrame]: Dictionary containing test results
        
        Raises:
            ValueError: If input parameters are invalid
        """
        # Validate inputs
        if not isinstance(target, pd.Series):
            raise ValueError("target must be a pandas Series")
        if not isinstance(cause, pd.Series):
            raise ValueError("cause must be a pandas Series")
        if not isinstance(maxlags, int) or maxlags < 1:
            raise ValueError("maxlags must be a positive integer")
        if len(target) != len(cause):
            raise ValueError("target and cause must have the same length")
            
        logger.debug(f"Running Granger causality test between {target.name} and {cause.name}")
        try:
            data = pd.concat([target, cause], axis=1)
            result = grangercausalitytests(data, maxlag=maxlags)
            lags = list(result.keys())
            res: List[Dict[str, Union[int, float]]] = []
            for lag in lags:
                res.append({
                    'lag': lag,
                    'ssr_ftest': result[lag][0]['ssr_ftest'][1],
                    'ssr_chi2test': result[lag][0]['ssr_chi2test'][1],
                    'lrtest': result[lag][0]['lrtest'][1],
                    'params_ftest': result[lag][0]['params_ftest'][1]
                })
            logger.debug(f"Granger causality test completed for {target.name} and {cause.name}")
            return {(target.name, cause.name): pd.DataFrame(res).set_index('lag')}
        except Exception as e:
            logger.error(f"Error in Granger causality test: {str(e)}")
            raise
        
    @staticmethod  
    def _get_column_pairs(data: pd.DataFrame, target: str) -> List[List[str]]:
        """
        Helper method to get all possible column pairs pertaining to the target variable
        
        Parameters:
            - data: pd.DataFrame: The data to analyze
            - target: str: The target variable
        
        Returns:
            List[List[str]]: A list of column pairs
            
        Raises:
            ValueError: If target is not in data columns
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.debug(f"Getting column pairs for target: {target}")
        column_pairs = list(combinations(data.columns, 2))
        out: List[List[str]] = []
        for x1,x2 in column_pairs:
            if x1 == target: 
                out.append([x1,x2])
            if x2 == target:
                out.append([x2,x1])
        logger.debug(f"Found {len(out)} column pairs")
        return out

    def granger_test(self, data: pd.DataFrame, target: str) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Run granger causality test on all columns in the data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Starting Granger causality tests for target: {target}")
        results: Dict[Tuple[str, str], pd.DataFrame] = {}
        cp = self._get_column_pairs(data, target)
        pbar = tqdm(cp, desc="Granger Causality")
        
        try:
            for x1, x2 in pbar:
                pbar.set_description(f"Granger Causality: {x1} -> {x2}")
                d = self.__granger_causality(data[x1], data[x2])
                results[(x1, x2)] = d[(x1, x2)]
            pbar.close()
            logger.info(f"Completed Granger causality tests for target: {target}")
            return results
        except Exception as e:
            logger.error(f"Error in Granger test: {str(e)}")
            raise
    
    def instantaneous_causality(self, fit: VARResults, data: pd.DataFrame, target: str) -> List[Tuple[str, str]]:
        """
        Instantaneous causality reflects a non-zero correlation between the variables in the system. 
        Perform Instantaneous causality test from the Var Model Fit.
        
        Parameters:
            - fit: The VAR model fit
            - data: The data to analyze
            - target: The target variable
        
        Returns:
            List[Tuple[str, str]]: A list of tuples representing instantaneous causality
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Starting instantaneous causality test for target: {target}")
        instaneous_cause: List[Tuple[str, str]] = []
        features = [x for x in data.columns if x != target]
        
        for i in features:
            try:
                t = fit.test_inst_causality(causing=i).summary()
                if t[1][2].data < self.significance_level:
                    if t[1][2].data == 0:
                        msg = f"{i} Perfect Instaneous Causality detected for {target}."
                        warnings.warn(msg)
                        logger.warning(msg)
                    instaneous_cause.append((target, i))
                    if self.verbose:
                        confi = int(100 - (100*self.significance_level))
                        logger.info(f'{i} has an instantaneous causal effect on {target} @ {confi}% confidence level, p-value: {t[1][2].data}')
            except Exception as e:
                logger.error(f"Error in instantaneous causality test for {i}: {str(e)}")
                raise ValueError(f"{i} Error in Instaneous Causality test: {str(e)}")
                    
        logger.info(f"Completed instantaneous causality test for target: {target}")
        return instaneous_cause
    
    def impulse_response(self, fit: VARResults, data: pd.DataFrame, target: str) -> Any:
        """
        Calculate the impulse response of the target variable
        
        Parameters:
            - fit: The VAR model fit
            - data: The data to analyze
            - target: The target variable
        
        Returns:
            Any: The impulse response
            
        Raises:
            ValueError: If error occurs in calculation
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Calculating impulse response for target: {target}")
        try:
            irf = fit.irf(10)
            logger.info(f"Completed impulse response calculation for target: {target}")
            return irf
        except Exception as e:
            logger.error(f"Error in impulse response calculation: {str(e)}")
            raise ValueError(f"Error in Impulse Response Function: {str(e)}")
    
    def causality_tests(self, data: pd.DataFrame, target: str, model: Optional[VARResults] = None) -> Dict[str, List[Any]]:
        """
        Unified causality test interface
        
        Arguments:
            - data: pd.DataFrame: The data to analyze
            - target: str: The target variable
            - model: The VAR model fit
        
        Returns:
            Dict[str, List[Any]]: A dictionary of causality test results
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Starting unified causality tests for target: {target}")
        results: Dict[str, List[Any]] = {
            'granger': [],
            'instantaneous': [],
            'impulse_response': [],
        }
        
        try:
            granger_tests = self.granger_test(data, target)
            for k, v in granger_tests.items():
                mask = v < self.significance_level
                mask_sums = mask.sum(axis=1)
                mask_sums = mask_sums[mask_sums == mask.shape[1]]
                new_v = v.loc[mask_sums.index]
                if not new_v.empty:
                    results['granger'].append((k, new_v.index.values))
                    if self.verbose:
                        logger.info(f'{k[1]} Does Granger Cause {k[0]} @ {self.significance_level}% confidence level, Lags: {new_v.index.values}')

            if model:
                results['instantaneous'] = self.instantaneous_causality(model, data, target)
                results['impulse_response'] = self.impulse_response(model, data, target)
                
            logger.info(f"Completed unified causality tests for target: {target}")
            return results
        except Exception as e:
            logger.error(f"Error in causality tests: {str(e)}")
            raise

if __name__ == "__main__":
    print("\n6.30: For one who sees Me everywhere and sees everything in Me, I am never forgotten by them and they are never forgotten by Me.\n")
    try:
        data = pd.read_csv('examples/data/stock_returns.csv', index_col=0, parse_dates=['Date'])
        data = data.dropna(axis = 1).iloc[:, :]
        from stationary_checks import StationaryTests
        st = StationaryTests()
        df, report, summary = st.check_stationarity(data)
        
        ca = CausalityAnalyzer()
        res = ca.causality_tests(df, 'SPY')
        
        g = res['granger']
        print(g)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise