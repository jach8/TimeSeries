"""Module for analyzing correlations between time series with stationarity checks."""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults, VARResultsWrapper
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from io import StringIO

from .stationarity import StationaryTests
from .causality import CausalityAnalyzer

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class CorrelationAnalysisError(Exception):
    """Base exception class for correlation analysis errors."""
    pass


class DataValidationError(CorrelationAnalysisError):
    """Exception raised for input data validation errors."""
    pass


class StationarityError(CorrelationAnalysisError):
    """Exception raised when data cannot be made stationary."""
    pass


class VARModelError(CorrelationAnalysisError):
    """Exception raised for VAR model fitting errors."""
    pass


class AnalyzeCorrelation:
    """Time series correlation analysis with stationarity checks."""
    
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        decompose: bool = False,
        verbose: bool = False,
        stationarity_config: Optional[Dict[str, Any]] = None,
        causality_config: Optional[Dict[str, Any]] = None
    ) -> None:
        logger.info("Initializing AnalyzeCorrelation")
        self._validate_inputs(x, y)
        
        self.verbose = verbose
        self.cause = str(y.name)
        self.decompose = decompose
        self.stationarity_report: Optional[Dict[str, Any]] = None
        self.df: pd.DataFrame = pd.DataFrame()  # Initialize as empty DataFrame
        self.features: List[str] = []
        self.var_model: Optional[VARResults] = None
        self.vif_report: Optional[pd.DataFrame] = None
        self.model_search_results: Optional[pd.DataFrame] = None
        self.scaler: RobustScaler = RobustScaler()

        # Initialize modules
        try:
            self.stationary_tester = StationaryTests(
                test_config=stationarity_config or {},
                verbose=verbose
            )
            self.causality_analyzer = CausalityAnalyzer(
                causality_config=causality_config or {},
                verbose=verbose
            )
            self._setup_data(x, y)
            logger.info("Initialization completed successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise CorrelationAnalysisError(f"Failed to initialize analysis: {str(e)}")

    def _validate_inputs(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data formats and compatibility."""
        if not isinstance(x, pd.DataFrame):
            raise DataValidationError("x must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise DataValidationError("y must be a pandas Series")
        if y.name is None:
            raise DataValidationError("y must have a name")
        if x.empty or y.empty:
            raise DataValidationError("Input data cannot be empty")
        if not x.index.equals(y.index):
            raise DataValidationError("x and y must have matching indices")

    def _setup_data(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Prepare data for analysis."""
        logger.info("Setting up data for analysis")
        try:
            # Merge and clean data
            df = pd.concat([x, y], axis=1)
            df = self._convert_to_period_index(df)

            # Feature processing
            processed_features = self._process_features(x)
            self.df = pd.DataFrame(
                data=processed_features,
                index=processed_features.index
            )
            self.df[y.name] = y
            self.features = processed_features.columns.tolist()
            logger.info("Data setup completed successfully")
        except Exception as e:
            logger.error(f"Data setup failed: {str(e)}")
            raise DataValidationError(f"Failed to setup data: {str(e)}")

    @staticmethod
    def _convert_to_period_index(df: pd.DataFrame) -> pd.DataFrame:
        """Convert index to period format."""
        try:
            df.index = pd.to_datetime(df.index).to_period('D')
            return df
        except Exception as e:
            logger.error(f"Index conversion failed: {str(e)}")
            raise DataValidationError(f"Failed to convert index: {str(e)}")

    def _process_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Process features with optional PCA decomposition."""
        try:
            if self.decompose:
                return self._pca_decomposition(x)
            return pd.DataFrame(
                self.scaler.fit_transform(x),
                columns=x.columns,
                index=x.index
            )
        except Exception as e:
            logger.error(f"Feature processing failed: {str(e)}")
            raise DataValidationError(f"Failed to process features: {str(e)}")

    def _pca_decomposition(self, x: pd.DataFrame, n_components: int = 4) -> pd.DataFrame:
        """Perform PCA decomposition."""
        logger.info(f"Performing PCA with {n_components} components")
        try:
            centered_x = x - x.mean()
            scaled_x = self.scaler.fit_transform(centered_x)
            pca = PCA(n_components=n_components).fit(scaled_x)
            return pd.DataFrame(
                pca.transform(scaled_x),
                columns=[f'PC{i}' for i in range(1, n_components+1)],
                index=x.index
            )
        except Exception as e:
            logger.error(f"PCA decomposition failed: {str(e)}")
            raise DataValidationError(f"Failed to perform PCA: {str(e)}")

    def _fit_var_model(self, trend: str = 'c', ic: str = 'aic') -> VARResults:
        """Fit VAR model."""
        logger.info(f"Fitting VAR model with {ic.upper()} criterion")
        try:
            model = VAR(self.df)
            order_df = model.select_order(trend=trend).summary()
            
            html_data = order_df.as_html()
            search_results = pd.read_html(
                StringIO(html_data),
                header=0,
            )[0].rename(
                columns={'Unnamed: 0': 'Lags'}
            )
            search_results.columns = [x.lower() for x in search_results.columns]
            
            self.model_search_results = search_results
            best_model = search_results.loc[search_results[ic.lower()].str.contains('\*')]['lags'].values[0]
            
            if self.verbose:
                print(f"Optimal lag order ({ic.upper()}): {best_model}")
                
            fitted_model = model.fit(maxlags=best_model, trend=trend)
            logger.info(f"VAR model fitted successfully with {best_model} lags")
            # Cast VARResultsWrapper to VARResults
            return cast(VARResults, fitted_model)
        except Exception as e:
            logger.error(f"VAR model fitting failed: {str(e)}")
            raise VARModelError(f"Failed to fit VAR model: {str(e)}")

    def analyze_relationships(self) -> Dict[str, Any]:
        """Perform complete analysis pipeline."""
        logger.info("Starting relationship analysis")
        
        # Check stationarity
        stationary_df, report, tests = self.stationary_tester.check_stationarity(self.df)
        self.stationarity_report = report
        self.df = stationary_df

        if self.verbose:
            print("Stationarity transformation complete")
            print(pd.DataFrame(report))

        try:
            # Fit VAR model
            self.var_model = self._fit_var_model()
            
            # Run causality tests
            causality = self.causality_analyzer.causality_tests(
                data=self.df,
                target=self.cause,
                model=self.var_model
            )
            
            logger.info("Analysis completed successfully")
            return {
                'stationarity_report': self.stationarity_report,
                'var_model': self.var_model,
                'causality': causality,
                'new_data': self.df
            }
        except Exception as e:
            logger.error(f"Relationship analysis failed: {str(e)}")
            raise VARModelError(f"Failed to analyze relationships: {str(e)}")