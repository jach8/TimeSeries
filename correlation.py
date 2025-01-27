import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import PCA
from io import StringIO
from stationary_checks import StationaryTests  
from causality_logic import CausalityAnalyzer


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
        
        # Initialize modules
        self.stationary_tester = StationaryTests(
            test_config=stationarity_config or {},
            verbose=verbose
        )
        
        self.causality_analyzer = CausalityAnalyzer(
            **(causality_config or {})
        )
        self._setup_data(x, y)

    def _setup_data(self, x, y):
        """Data preparation pipeline"""
        # Merge and clean data
        df = x.merge(y, left_index=True, right_index=True).dropna()
        df = self._convert_to_period_index(df)
        
        # Feature processing
        self.scaler = StandardScaler()
        processed_features = self._process_features(x, df.index)
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

    def _process_features(self, x, index):
        """Feature engineering pipeline"""
        if self.decompose:
            return self._pca_decomposition(x)
        return pd.DataFrame(
            self.scaler.fit_transform(x),
            columns=x.columns,
            index=x.index
        )

    def _pca_decomposition(self, x, n_components=3):
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
            self.df = stationary_df
            if self.verbose:
                print("Stationarity transformation complete")
                print(pd.DataFrame(report).T)
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
        causality = self.causality_analyzer.causality_tests(data = self.df, target = self.cause)
        
        return {
            'stationarity_report': self.stationary_tester._test_history,
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

    # # Updated test code
    data = pd.read_csv('examples/data/stock_returns.csv', parse_dates=['Date'], index_col='Date').dropna(axis=1)
    data = data.dropna(axis = 1)
    target = np.random.choice(data.columns, 1)[0]
    target = "SPY"
    features = data.drop(columns=target).iloc[:, :]
    x = features.copy()
    y = data[target].copy()
    
    print('\n\n', y.name, '\n\n')
    
    # data = pd.read_csv("examples/data/hdd.csv", parse_dates=['date'], index_col='date')
    # data = data.drop_duplicates().dropna()
    # # Shift the target variable, forward by 1
    # data['target'] = data['target'].shift(-1)
    # data = data.dropna()
    # x = data.drop(columns='target')
    # y = data['target']
    
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
