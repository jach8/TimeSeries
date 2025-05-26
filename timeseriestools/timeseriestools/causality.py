"""Module for implementing various causality tests between time series."""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults
from tqdm import tqdm

from .utils.granger import grangercausalitytests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CausalityAnalyzer:
    """Handles different types of causality tests with a consistent interface."""

    def __init__(
        self,
        causality_config: Optional[Dict[str, Union[float, int]]] = None,
        verbose: bool = False
    ) -> None:
        self.default_config = {
            'significance_level': 0.05,
            'max_lag': 3
        }
        self.config = causality_config or self.default_config
        
        # Validate configuration values
        if not isinstance(self.config.get('significance_level', 0.05), (int, float)):
            raise ValueError("significance_level must be a number")
        if not isinstance(self.config.get('max_lag', 3), int):
            raise ValueError("max_lag must be an integer")
            
        self.significance_level = float(self.config.get('significance_level', 0.05))
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
            
        self.max_lag = int(self.config.get('max_lag', 3))
        if self.max_lag < 1:
            raise ValueError("max_lag must be positive")
            
        self.verbose = verbose
        logger.info(f"CausalityAnalyzer initialized with significance_level={self.significance_level}, max_lag={self.max_lag}")

    @staticmethod
    def __granger_causality(
        target: pd.Series,
        cause: pd.Series,
        maxlags: int = 3
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Run Granger causality test between two time series."""
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
            data = pd.concat([target, cause], axis=1).values
            result = grangercausalitytests(data, maxlag=maxlags)
            lags = list(result.keys())
            res: List[Dict[str, Union[int, float]]] = []
            
            for lag in lags:
                res.append({
                    'lag': lag,
                    'ssr_ftest': result[lag]['ssr_ftest'][1],
                    'ssr_chi2test': result[lag]['ssr_chi2test'][1],
                    'lrtest': result[lag]['lrtest'][1],
                    'params_ftest': result[lag]['params_ftest'][1]
                })
                
            target_name = str(target.name or "target")
            cause_name = str(cause.name or "cause")
            logger.debug(f"Granger causality test completed for {target_name} and {cause_name}")
            return {(target_name, cause_name): pd.DataFrame(res).set_index('lag')}
        except Exception as e:
            logger.error(f"Error in Granger causality test: {str(e)}")
            raise

    def instantaneous_causality(
        self,
        model: VARResults,
        data: pd.DataFrame,
        target: str
    ) -> List[Tuple[str, str]]:
        """Test for instantaneous causality using VAR model results.
        
        Args:
            model (VARResults): Fitted VAR model
            data (pd.DataFrame): Input time series data
            target (str): Target variable name
            
        Returns:
            List[Tuple[str, str]]: List of variable pairs with instantaneous causality
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Starting instantaneous causality test for target: {target}")
        instantaneous_causes: List[Tuple[str, str]] = []
        features = [x for x in data.columns if x != target]
        
        for feature in features:
            try:
                test_result = model.test_inst_causality(causing=feature)
                if test_result.pvalue < self.significance_level:
                    if test_result.pvalue == 0:
                        msg = f"{feature} Perfect Instantaneous Causality detected for {target}."
                        warnings.warn(msg)
                        logger.warning(msg)
                    instantaneous_causes.append((target, feature))
                    if self.verbose:
                        conf = int(100 - (100 * self.significance_level))
                        logger.info(
                            f'{feature} has an instantaneous causal effect on {target} '
                            f'@ {conf}% confidence level, p-value: {test_result.pvalue}'
                        )
            except Exception as e:
                logger.error(f"Error in instantaneous causality test for {feature}: {str(e)}")
                raise ValueError(f"Error in Instantaneous Causality test: {str(e)}")
                    
        logger.info(f"Completed instantaneous causality test for target: {target}")
        return instantaneous_causes

    def impulse_response(
        self,
        model: VARResults,
        data: pd.DataFrame,
        target: str,
        periods: int = 10
    ) -> Any:
        """Calculate impulse response function from VAR model.
        
        Args:
            model (VARResults): Fitted VAR model
            data (pd.DataFrame): Input time series data
            target (str): Target variable name
            periods (int, optional): Number of periods for IRF. Defaults to 10.
            
        Returns:
            Any: Impulse response function results
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
        if not isinstance(periods, int) or periods < 1:
            raise ValueError("periods must be a positive integer")
            
        logger.info(f"Calculating impulse response for target: {target}")
        try:
            irf = model.irf(periods)
            logger.info(f"Completed impulse response calculation for target: {target}")
            return irf
        except Exception as e:
            logger.error(f"Error in impulse response calculation: {str(e)}")
            raise ValueError(f"Error in Impulse Response Function: {str(e)}")

    def causality_tests(
        self,
        data: pd.DataFrame,
        target: str,
        model: Optional[VARResults] = None
    ) -> Dict[str, List[Any]]:
        """Run all available causality tests."""
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
            # Granger causality
            granger_tests = self.granger_test(data, target)
            for k, v in granger_tests.items():
                mask = v < self.significance_level
                mask_sums = mask.sum(axis=1)
                mask_sums = mask_sums[mask_sums == mask.shape[1]]
                new_v = v.loc[mask_sums.index]
                if not new_v.empty:
                    results['granger'].append((k, new_v.index.values))
                    if self.verbose:
                        logger.info(
                            f'{k[1]} Does Granger Cause {k[0]} @ {self.significance_level} '
                            f'confidence level, Lags: {new_v.index.values}'
                        )

            # Additional tests if VAR model provided
            if model:
                results['instantaneous'] = self.instantaneous_causality(model, data, target)
                results['impulse_response'] = self.impulse_response(model, data, target)
                
            logger.info(f"Completed unified causality tests for target: {target}")
            return results
        except Exception as e:
            logger.error(f"Error in causality tests: {str(e)}")
            raise

    def granger_test(
        self,
        data: pd.DataFrame,
        target: str
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Run Granger causality test on all columns in relation to target."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.info(f"Starting Granger causality tests for target: {target}")
        results: Dict[Tuple[str, str], pd.DataFrame] = {}
        column_pairs = self._get_column_pairs(data, target)
        pbar = tqdm(column_pairs, desc="Granger Causality")
        
        try:
            for x1, x2 in pbar:
                pbar.set_description(f"Granger Causality: {x1} -> {x2}")
                d = self.__granger_causality(data[x1], data[x2], maxlags=self.max_lag)
                results[(x1, x2)] = d[(x1, x2)]
            pbar.close()
            logger.info(f"Completed Granger causality tests for target: {target}")
            return results
        except Exception as e:
            logger.error(f"Error in Granger test: {str(e)}")
            raise

    @staticmethod
    def _get_column_pairs(data: pd.DataFrame, target: str) -> List[List[str]]:
        """Get all possible column pairs involving the target variable."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"target variable '{target}' not found in data columns")
            
        logger.debug(f"Getting column pairs for target: {target}")
        pairs = []
        for col in data.columns:
            if col != target:
                pairs.append([target, col])
        logger.debug(f"Found {len(pairs)} column pairs")
        return pairs