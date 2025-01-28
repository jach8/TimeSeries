import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler, RobustScaler
from itertools import combinations
from sklearn.decomposition import PCA
from io import StringIO
from src.stationary_checks import StationaryTests  
from src.causality_logic import CausalityAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


class AnalyzeCorrelation:
    def __init__(self, x, y, decompose=True, verbose=False,
                 stationarity_config=None, causality_config=None):
        """
        Enhanced initialization with modular components
        
        Parameters:
        - stationarity_config: Dict for StationaryTests configuration
        - causality_config: Dict for CausalityAnalyzer configuration
        """
        self.verbose = verbose
        self.cause = y.name
        self.decompose = decompose
        self.stationarity_report = None
        
        # Initialize modules
        self.stationary_tester = StationaryTests(
            test_config=stationarity_config or {},
            verbose=verbose
        )
        
        self.causality_analyzer = CausalityAnalyzer(
            causality_config=causality_config or {},
            verbose=verbose
        )
        self._setup_data(x, y)
        
    def _setup_data(self, x, y):
        """Data preparation pipeline"""
        # Merge and clean data
        df = x.merge(y, left_index=True, right_index=True).dropna()
        df = self._convert_to_period_index(df)

        
        # Feature processing
        self.scaler = RobustScaler()
        processed_features = self._process_features(x, df.index)
        processed_features = self.__drop_high_vif(processed_features)
        # Create analysis-ready dataframe
        self.df = pd.DataFrame(
            data=processed_features,
            index=processed_features.index
        )
        self.df[y.name] = y
        self.features = processed_features.columns.tolist()

    def _convert_to_period_index(self, df):
        """Uniform index formatting"""
        df.index = pd.to_datetime(df.index).to_period('D')
        return df
    
    def __drop_high_vif(self, df, threshold=200):
        """Drop features with high VIF
        Recommended threshold is 5, if not then the explanatory variables are highly collinear, 
            and the parameter estimates will have large standard errors.
            
        Parameters:
        - df: pd.DataFrame: The input dataframe
        - threshold: float: The VIF threshold to drop features
        """
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
            out = df.drop(columns=dropcols)
            if self.verbose:
                print(f"High VIF features: {dropcols}")
            return out
        return df.drop(columns=['Intercept'])


    def _process_features(self, x, index):
        """Feature engineering pipeline"""
        if self.decompose:
            return self._pca_decomposition(x)
        return pd.DataFrame(
            self.scaler.fit_transform(x),
            columns=x.columns,
            index=x.index
        )


    def _pca_decomposition(self, x, n_components=9):
        """Dimensionality reduction"""
        centered_x = x - x.mean()
        scaled_x = self.scaler.fit_transform(centered_x)
        pca = PCA(n_components=n_components).fit(scaled_x)
        out= pd.DataFrame(
            pca.transform(scaled_x),
            columns=[f'PC{i}' for i in range(1, n_components+1)],
            index=x.index
        ) 
        return out

    def ensure_stationarity(self):
        """Full stationarity pipeline"""
        try:
            stationary_df, report, tests = self.stationary_tester.check_stationarity(self.df)
            self.stationary_report = report
            self.df = stationary_df

            if self.verbose:
                print("Stationarity transformation complete")
                print(pd.DataFrame(report))
            return True
        except Exception as e:
            print(self.df)
            warnings.warn(f"Stationarity failed: {str(e)}")
            
            return False

    def analyze_relationships(self):
        """Complete analysis pipeline"""
        if not self.ensure_stationarity():
            raise ValueError("Data could not be made stationary")
            
        self.var_model = self._fit_var_model()
        causality = self.causality_analyzer.causality_tests(data = self.df, target = self.cause, model = self.var_model)
        
        return {
            'stationarity_report': self.stationary_report,
            'var_model': self.var_model,
            'causality': causality,
            'new_data': self.df
        }

    def _fit_var_model(self, trend='c', ic='aic'):
        """Optimized VAR model fitting"""
        model = VAR(self.df.dropna())
        order_df = model.select_order(trend=trend).summary()
        
        # get html string
        html_data = order_df.as_html()
        # Wrap string in StringIO and convert to DataFrame
        search_results = pd.read_html(
            StringIO(html_data), header=0,
        )[0].rename(
            columns = {'Unnamed: 0': 'Lags'}
        )
        search_results.columns = [x.lower() for x in search_results.columns] 
        # Store the search results
        self.model_search_results = search_results
        # The best model will have a * in the value. 
        best_model = search_results.loc[search_results[ic.lower()].str.contains('\*')]['lags'].values[0]
        
        if self.verbose:
            print(f"Optimal lag order ({ic.upper()}): {best_model}")
        
            
        return model.fit(maxlags=best_model, trend=trend)

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources if needed"""
        pass

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


   
    ###########################################################
    with AnalyzeCorrelation(
        x=x, y = y,
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
        # print(f"VAR Model Summary:\n{results['var_model'].summary()}")
