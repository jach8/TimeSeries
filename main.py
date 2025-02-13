from src.correlation import AnalyzeCorrelation
from src.causality_logic import CausalityAnalyzer
from src.stationary_checks import StationaryTests


class Analyze:
    def __init__(self, verbose=False, stationarity_config=None, causality_config=None):
        """
        Enhanced initialization with modular components
        
        Parameters:
        - stationarity_config: Dict for StationaryTests configuration
        - causality_config: Dict for CausalityAnalyzer configuration
        """
        self.verbose = verbose
        
        # Initialize modules
        self.default_stationary_config = {
            'adf': {'max_diff': 5, 'significance': 0.05},
            'kpss': {'significance': 0.05},
            'pp': {'significance': 0.05},
            'structural_break': True,
            'gls': True,
            'nonlinear': True
        }
    
        self.default_causality_config = {
            'significance_level': 0.05,
            'max_lag': 3
        }
        
        self.stationary_config = stationarity_config or self.default_stationary_config
        self.causality_config = causality_config or self.default_causality_config
        
        
    def analyze_correlation(self, x, y, decompose=False):
        """
        Analyze the correlation between two time series
        
        Parameters:
        - x: pd.Series: The first time series
        - y: pd.Series: The second time series
        - decompose: bool: Whether to decompose via PCA or not.
        
        Returns:
        - AnalyzeCorrelation: An instance of the correlation analysis
        """
        self.AC = AnalyzeCorrelation(
            x, y,
            decompose=decompose,
            verbose=self.verbose, 
            stationarity_config=self.stationary_config, 
            causality_config=self.causality_config
        )

        # Save results: 
        self.results = self.AC.analyze_relationships()
        return self.results
        
    
    def results(self, x, y, decompose=False):
        """
        Display the results of the correlation analysis
        
        Parameters:
        - x: pd.Series: The first time series
        - y: pd.Series: The second time series
        """
        return self.analyze_correlation(x, y, decompose)
        
        
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.data import *
    x, y = test_data1(return_xy=True, path_to_src='src/')
    x = x.dropna(axis = 1)
    print(x.shape, y.shape)
    
    y.name = 'spy'
    # print(x.head().iloc[:,:])
    # Initialize the analysis
    a = Analyze(verbose=False)
    d = a.results(x, y, decompose=False)
    
    print("\nNew Data with Stationarity Transformation:")
    print(d['new_data'])

    print("\Stationary Analysis:")
    print(d['stationarity_report'])
    
    print("\nKey Findings:")
    print(f"Significant Granger causes: {d['causality']}")

    print("\nVAR Model Summary:")
    print(d['var_model'].summary())