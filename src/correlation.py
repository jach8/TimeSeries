import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults
from sklearn.preprocessing import StandardScaler, RobustScaler
from itertools import combinations
from sklearn.decomposition import PCA
from io import StringIO
from src.stationary_checks import StationaryTests
from src.causality_logic import CausalityAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class CorrelationAnalysisError(Exception):
    """Base exception class for correlation analysis errors"""
    pass


class DataValidationError(CorrelationAnalysisError):
    """Exception raised for input data validation errors"""
    pass


class StationarityError(CorrelationAnalysisError):
    """Exception raised when data cannot be made stationary"""
    pass


class VARModelError(CorrelationAnalysisError):
    """Exception raised for VAR model fitting errors"""
    pass


class AnalyzeCorrelation:
    def __init__(self,
                 x: pd.DataFrame,
                 y: pd.Series,
                 decompose: bool = True,
                 verbose: bool = False,
                 stationarity_config: Optional[Dict[str, Any]] = None,
                 causality_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the correlation analysis with enhanced error handling and validation.

        Parameters:
            x: Input features dataframe
            y: Target variable series
            decompose: Whether to perform PCA decomposition
            verbose: Enable verbose output
            stationarity_config: Configuration for stationarity tests
            causality_config: Configuration for causality analysis

        Raises:
            DataValidationError: If input data validation fails
        """
        logger.info("Initializing AnalyzeCorrelation")
        self._validate_inputs(x, y)
        
        self.verbose: bool = verbose
        self.cause: str = y.name
        self.decompose: bool = decompose
        self.stationarity_report: Optional[Dict[str, Any]] = None
        self.df: Optional[pd.DataFrame] = None
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
        """
        Validate input data formats and compatibility.

        Raises:
            DataValidationError: If validation fails
        """
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
        """
        Prepare data for analysis with enhanced error handling.
        
        Raises:
            DataValidationError: If data preparation fails
        """
        logger.info("Setting up data for analysis")
        try:
            # Merge and clean data
            df = x.merge(y, left_index=True, right_index=True).dropna()
            df = self._convert_to_period_index(df)

            # Feature processing
            processed_features = self._process_features(x, df.index)
            processed_features = self.__drop_high_vif(processed_features)
            
            # Create analysis-ready dataframe
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

    def _convert_to_period_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert index to period format with error handling.
        
        Returns:
            DataFrame with period index
        """
        try:
            df.index = pd.to_datetime(df.index).to_period('D')
            return df
        except Exception as e:
            logger.error(f"Index conversion failed: {str(e)}")
            raise DataValidationError(f"Failed to convert index to period format: {str(e)}")

    def __drop_high_vif(self, df: pd.DataFrame, threshold: float = 200) -> pd.DataFrame:
        """
        Drop features with high VIF values.
        
        Parameters:
            df: Input dataframe
            threshold: VIF threshold for dropping features
            
        Returns:
            DataFrame with high VIF features removed
        """
        logger.info(f"Checking VIF with threshold {threshold}")
        try:
            df['Intercept'] = 1
            vif_df = pd.DataFrame()
            vif_df["VIF"] = [vif(df.values, i) for i in range(df.shape[1])]
            vif_df["features"] = df.columns.tolist()
            
            high_vif = vif_df[vif_df["VIF"] > threshold]
            dropcols = high_vif['features'].tolist()
            if 'Intercept' not in dropcols:
                dropcols.append('Intercept')
            self.vif_report = vif_df
            
            if high_vif.shape[0] > 0:
                logger.warning(f"Dropping {len(dropcols)} features with high VIF")
                if self.verbose:
                    print(f"High VIF features: {dropcols}")
                return df.drop(columns=dropcols)
            return df.drop(columns=['Intercept'])
        except Exception as e:
            logger.error(f"VIF calculation failed: {str(e)}")
            raise DataValidationError(f"Failed to process VIF: {str(e)}")

    def _process_features(self, x: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        """
        Process features with optional PCA decomposition.
        
        Returns:
            Processed feature DataFrame
        """
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
        """
        Perform PCA decomposition with error handling.
        
        Returns:
            DataFrame with PCA components
        """
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

    def ensure_stationarity(self) -> bool:
        """
        Ensure data stationarity with enhanced error handling.
        
        Returns:
            bool indicating if stationarity was achieved
            
        Raises:
            StationarityError: If stationarity check fails
        """
        logger.info("Checking stationarity")
        try:
            stationary_df, report, tests = self.stationary_tester.check_stationarity(self.df)
            self.stationary_report = report
            self.df = stationary_df

            if self.verbose:
                print("Stationarity transformation complete")
                print(pd.DataFrame(report))
            logger.info("Stationarity check completed successfully")
            return True
        except Exception as e:
            logger.error(f"Stationarity check failed: {str(e)}")
            warnings.warn(f"Stationarity failed: {str(e)}")
            return False

    def analyze_relationships(self) -> Dict[str, Any]:
        """
        Perform complete analysis pipeline with enhanced error handling.
        
        Returns:
            Dictionary containing analysis results
            
        Raises:
            StationarityError: If data cannot be made stationary
            VARModelError: If VAR model fitting fails
        """
        logger.info("Starting relationship analysis")
        if not self.ensure_stationarity():
            raise StationarityError("Data could not be made stationary")
            
        try:
            self.var_model = self._fit_var_model()
            causality = self.causality_analyzer.causality_tests(
                data=self.df,
                target=self.cause,
                model=self.var_model
            )
            
            logger.info("Analysis completed successfully")
            return {
                'stationarity_report': self.stationary_report,
                'var_model': self.var_model,
                'causality': causality,
                'new_data': self.df
            }
        except Exception as e:
            logger.error(f"Relationship analysis failed: {str(e)}")
            raise VARModelError(f"Failed to analyze relationships: {str(e)}")

    def _fit_var_model(self, trend: str = 'c', ic: str = 'aic') -> VARResults:
        """
        Fit VAR model with error handling.
        
        Parameters:
            trend: Trend term type
            ic: Information criterion for lag selection
            
        Returns:
            Fitted VAR model
            
        Raises:
            VARModelError: If model fitting fails
        """
        logger.info(f"Fitting VAR model with {ic.upper()} criterion")
        try:
            model = VAR(self.df.dropna())
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
            return fitted_model
        except Exception as e:
            logger.error(f"VAR model fitting failed: {str(e)}")
            raise VARModelError(f"Failed to fit VAR model: {str(e)}")

    def __enter__(self) -> 'AnalyzeCorrelation':
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Optional[type],
                 exc_val: Optional[Exception],
                 exc_tb: Optional[Any]) -> None:
        """Context manager exit with cleanup"""
        if exc_type is not None:
            logger.error(f"Error during analysis: {str(exc_val)}")


if __name__ == "__main__":
    ###########################################################
    # Example configuration
    stationarity_cfg = {
        'adf': {'max_diff': 4, 'significance': 0.05},
        'kpss': {'significance': 0.05},
        'structural_break': False
    }
    
    causality_cfg = {
        'significance_level': 0.05,
        'max_lag': 4
    }
    ###########################################################
    # Example usage
    with AnalyzeCorrelation(
        x=x, y=y,
        stationarity_config=stationarity_cfg,
        causality_config=causality_cfg,
        verbose=False,
        decompose=False
    ) as analyzer:
        results = analyzer.analyze_relationships()
        print("\nNew Data with Stationarity Transformation:")
        print(results['new_data'])
        
        print("\nKey Findings:")
        print(f"Significant Granger causes: {results['causality']}")
        print()
