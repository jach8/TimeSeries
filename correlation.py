import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

class analyze_correlation:
    def __init__(self, x, y, verbose=False):
        """
        Initialize the analyze_correlation with features and target.

        Parameters:
        - x (pd.DataFrame): Features for analysis.
        - y (pd.Series): Target variable for causality and classification.
        - verbose (bool): If True, print detailed model metrics and updates.
        - max_iterations (int): Maximum number of iterations for VAR model fitting.
        """
        self.verbose = verbose
        self.cause = y.name
        self.set_xy(x, y)
        

    def setup_period_index(self, df):
        """
        Set the index of the DataFrame to a period index, for sktime processing. 
        - To convert back to datetime index, use: df.index.to_timestamp()   
        
        Args: 
            - df (pd.DataFrame): DataFrame to set the index for.
        
        Returns: 
            - pd.DataFrame: DataFrame with the index set to a period index.
        
        
        """
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period('B')
        return df

    def set_xy(self, x, y):
        """
        Initialize features and target variables.
        
        Args: 
            x (pd.DataFrame): Features for analysis.
            y (pd.Series): Target variable for causality and classification.
            
        """
        # Set scaler and feature names
        self.scaler = StandardScaler()
        self.features = x.columns.to_list()
        self.target = y.name  
        
        df = x.merge(y, left_index=True, right_index=True)
        choose_random_features = df.drop(columns=[self.target]).columns
    
        # Set the index to a period index
        self.df = self.setup_period_index(df)
        
        # Feature DataFrame
        self.x = df.drop(columns=[self.target]).copy()[self.features]
        self.y = df[self.target]
        
        # Scaled DataFrame
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df), 
            columns=self.df.columns, 
            index=self.df.index
        )
        
        if self.verbose:
            print(f"Dataframe shape: {self.df.shape}\n")

    def adf_test(self, timeseries, lag_method = 'AIC' ):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        -   Null Hypothesis (H0): the time series has a unit root. 
        -   Alternate Hypothesis (H1): the time series has no unit root
        
        Large p-values indicate a unit root in the timeseries, confirming its non-stationarity. 
        
        Args:
            timeseries (pd.Series): Time series data to test for stationarity.
            ci (float): Confidence interval for the test.
            lag_method (str): Method to use for determining the number of lags in the test:
                - 'AIC' or 'BIC' minimize the corresponding information criterion to get the number of lags in the test.
                - 't-stat': t-statistic: based on maxlag and drops a lag until the t-statistic on the last length is significant at the 95 % level.
                - 'None': then number of lags is set to maxlag
                
            
            
        Returns:
            p-value (float): P-value of the ADF test.
        """
        
        #Perform Dickey-Fuller test:
        dftest = adfuller(timeseries, autolag=lag_method)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    
        # return pd.DataFrame(dfoutput, columns = ['ADF Test']).round(2)        
        return dfoutput['p-value']
    
    
    def check_stationarity(self, ci = 0.05):
        """
        Performs the Augmented Dickey-Fuller test on the features to check for stationarity.
        -   If the p-value is less than the confidence interval (small p-value), the feature is stationary.
        -   If the p-value is greater than the confidence interval (large p-value), the feature is non-stationary.
        
        Args: 
            ci (float): Confidence interval for the test. default: 0.05
        
        """
        if self.verbose:
            print(f"Checking stationarity of features using ADF test at {ci} confidence level...\n")
        stationary = []; non_stationary = []
        for i in self.x.columns:
            p = self.adf_test(self.df_scaled[i])
            if p < ci:
                stationary.append(i)
            else:
                non_stationary.append(i)
                if self.verbose:
                    print(f"{i} is not stationary, p-value: {p:.3f} Consider dropping or differencing the feature")
        
        self.features = stationary
        if non_stationary != []:
            if self.verbose:
                print(f"\nNon-stationary features: {non_stationary}\n")
        else:
            if self.verbose:
                print("All features are stationary.\n")
        
        
    def _fit_var_model(self):
        """ 
        Fit a Vector Auto Regression Model to the features and target variable.
        
        Returns:
            - 
        
        """
        x = self.df_scaled[self.features]
        y = self.df_scaled[self.target]
        model = VAR(endog=x, exog=y, freq = 'B', dates = x.index)
        try:
            fit = model.fit(maxlags=20, ic='aic', )
            if self.verbose:
                print(f'VAR Model Summary: \n{fit.summary()}\n\n')
            return fit
        except np.linalg.LinAlgError as e:
            if 'not positive definite' in str(e):
                # Handle non-positive definite matrix:
                # Option 1: Reduce lags
                for lags in range(19, 0, -1):  # Try fewer lags
                    try:
                        if self.verbose:
                            print(f"Trying fewer lags: {lags}")
                        fit = model.fit(maxlags=lags, ic='aic')
                        if self.verbose:
                            print(f'Adjusted VAR Model Summary (lags={lags}): \n{fit.summary()}\n\n')
                        return fit
                    except np.linalg.LinAlgError:
                        continue  # If still not working, try fewer lags
            if self.verbose:
                print(f"VAR model fitting failed due to non-positive definite matrix. Error: {str(e)}")
        except Exception as e:  # Catch other exceptions
            if self.verbose:
                print(f"VAR model fitting failed to converge. Error: {str(e)}")
        
        return None  # Return None if all attempts fail

    def _granger_causality(self, ci = 0.05):
        """
        Perform Granger causality tests if VAR model fits.
        Granger Causality Null Hypothesis testing for causlity between features and the target variable. 
        
        """
        if self.verbose:
            print(f"Checking Granger Causality tests...\n")
        fit = self._fit_var_model()
        if fit is None:
            return []
        
        checks = self.df.columns.to_list()
        granger_caused = []
        print(checks)
        
        for i in checks:
            t = fit.test_causality(caused=self.cause, causing=i, kind='f').summary()
            if t[1][2].data < ci:
                granger_caused.append((self.cause, i))
                if self.verbose:
                    print(f'{i} Does Granger Cause {self.cause} @ 90% confidence level, p-value: {t[1][2].data:.3f}')
        
        return granger_caused

    def _contemporaneous_causality(self):
        """Perform contemporaneous causality tests."""
        checks = self.x.columns.to_list()
        contemporaneous_causality = []
        for i in checks:
            t = grangercausalitytests(self.df_scaled[[self.cause, i]], maxlag=1, verbose=False)
            if t[1][0]['ssr_ftest'][1] < 0.10:
                contemporaneous_causality.append((self.cause, i))
                if self.verbose:
                    print(f'{i} Does Contemporaneously Cause {self.cause} @ 90% confidence level, p-value: {t[1][0]["ssr_ftest"][1]:.3f}')
        return contemporaneous_causality

    

    def analyze(self):
        """
        Run correlation analysis and return the results

        Returns:
        - dict: Dictionary containing the granger_causality and contemporaneous_causality results.
        """
        return {
            "stationarity_tests": self.check_stationarity(),
            "var_model": self._fit_var_model(),
            "granger_causality": self._granger_causality(),
            "contemporaneous_causality": self._contemporaneous_causality(),
        }

# Example usage:
# analyzer = analyze_correlation(x, y, verbose=True)
# results = analyzer.analyze()

if __name__ == "__main__":
    print(""" 7.4: Earth, water, fire, air, ether, mind, spiritual intelligence and false ego; thus these are the eightfold divisions of my external energy.\n""")

    import sys
    sys.path.append("/Users/jerald/Documents/Dir/Python/Stocks")
    from models.anom.stocks.connect import data
    d = data('../Stocks/')
    
    x, y  = d._returnxy('spy', keep_close=False)
    a = analyze_correlation(x, y, verbose=True)
    
    results = a.analyze()
    print(results)
    