import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from sklearn.decomposition import PCA

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
        # Deprecated: 
        df.index = df.index.to_period('D')
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
        
        # Merge Endogenous (X) and Exogenous (Y) variables
        df = x.merge(y, left_index=True, right_index=True).dropna()
    
        # Set the index to a period index
        self.df = self.setup_period_index(df)
        
        # Feature DataFrame
        self.x = self.df.drop(columns=[self.target]).copy()[self.features]
        self.y = df[self.target]
        
        # ###Scaled DataFrame
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df), 
            columns=self.df.columns, 
            index=self.df.index
        )
        
        # # PCA Decomposition
        # self.pca = PCA(n_components = 3).fit(self.df_scaled)
        # self.df_scaled = pd.DataFrame(
        #     self.pca.transform(self.df_scaled), 
        #     columns=[f'PC{i}' for i in range(1, 4)], 
        #     index=self.df_scaled.index
        # )
        # self.df_scaled[self.target] = self.y
        # self.features = self.df_scaled.columns.to_list()
        # self.df = self.df_scaled.copy()
        
        # Initialize the model  
        try: 
            self.model = self._fit_var_model()
        except Exception as e:
            if self.verbose:
                print(f"Error in fitting the VAR model: {str(e)}")
            self.model = None
        
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
    
    
    # def check_stationarity(self, ci = 0.05):
    #     """
    #     Performs the Augmented Dickey-Fuller test on the features to check for stationarity.
    #     -   If the p-value is less than the confidence interval (small p-value), the feature is stationary.
    #     -   If the p-value is greater than the confidence interval (large p-value), the feature is non-stationary.
        
    #     Args: 
    #         ci (float): Confidence interval for the test. default: 0.05
        
    #     """
    #     if self.verbose:
    #         print(f"Checking stationarity of features using ADF test at {ci} confidence level...")
    #     stationary = []; non_stationary = []
    #     for i in self.df.columns:
    #         p = self.adf_test(self.df_scaled[i])
    #         if p < ci:
    #             stationary.append(i)
    #         else:
    #             non_stationary.append(i)
    #             if self.verbose:
    #                 print(f"{i} is not stationary, p-value: {p:.3f} Consider dropping or differencing the feature")
        
    #     self.features = stationary
    #     if non_stationary != []:
    #         if self.verbose:
    #             print(f"\nNon-stationary features: {non_stationary}\n")
    #     else:
    #         if self.verbose:
    #             print("All features are stationary.\n")

    def check_stationarity(self, ci=0.05, max_differences=3):
        """
        Performs the Augmented Dickey-Fuller test on the features to check for stationarity.
        - If the p-value is less than the confidence interval (small p-value), the feature is stationary.
        - If the p-value is greater than the confidence interval (large p-value), the feature is non-stationary.
        This function will attempt to make non-stationary series stationary by differencing.

        Args: 
            ci (float): Confidence interval for the test. Default: 0.05
            max_differences (int): Maximum number of times to difference a feature. Default: 3

        Raises:
            ValueError: If the number of differences exceeds max_differences for any feature.
        """
        if self.verbose:
            print(f"Checking stationarity of features using ADF test at {ci} confidence level...")

        stationary = []
        non_stationary = []
        original_columns = list(self.df.columns)

        for col in original_columns:
            diff_count = 0
            while diff_count <= max_differences:
                try:
                    series = self.df_scaled[col].dropna()
                    result = adfuller(series)
                    p_value = result[1]

                    if p_value < ci:
                        stationary.append(col)
                        break
                    else:
                        # If not stationary, difference the series
                        self.df_scaled[col] = self.df_scaled[col].diff().dropna()
                        diff_count += 1
                        if self.verbose:
                            print(f"{col} is not stationary, applying differencing. P-value: {p_value:.3f}, Differences applied: {diff_count}")

                except Exception as e:
                    if self.verbose:
                        print(f"Error in ADF test for column {col}: {str(e)}")
                    non_stationary.append(col)
                    break

            if diff_count > max_differences:
                non_stationary.append(col)
                if self.verbose:
                    print(f"{col} could not be made stationary after {max_differences} differences")

        self.features = stationary
        
        if non_stationary:
            if self.verbose:
                print(f"\nNon-stationary features after attempting to make stationary: {non_stationary}\n")
            raise ValueError(f"Features {non_stationary} remain non-stationary after {max_differences} differences.")
        else:
            if self.verbose:
                print("All features are now stationary.\n")
        
        
    # def _fit_var_model(self):
    #     """ 
    #     Fit a Vector Auto Regression Model to the features and target variable.
        
    #     Returns:
    #         - 
        
    #     """
    #     # x = self.df_scaled[self.features]
    #     # y = self.df_scaled[self.target]
    #     # model = VAR(endog=x, exog=y, freq = 'B', dates = x.index)
    #     model = VAR(self.df_scaled)
    #     try:
    #         fit = model.fit(maxlags=10, ic='aic', )
    #         self.model = fit
    #         if self.verbose:
    #             print(f'Var Fit Successfull with lags: {fit.k_ar}')
    #         return fit
    #     except np.linalg.LinAlgError as e:
    #         if 'not positive definite' in str(e):
    #             # Handle non-positive definite matrix:
    #             # Option 1: Reduce lags
    #             for lags in range(9, 2, -1):  # Try fewer lags
    #                 try:
    #                     if self.verbose:
    #                         print(f"Trying fewer lags: {lags}")
    #                     fit = model.fit(maxlags=lags, ic='aic')
    #                     self.model = model.fit(maxlags=lags)
    #                     if self.verbose:
    #                         print(f'Adjusted VAR Model fitting successful (lags={lags})')
    #                     return fit
    #                 except np.linalg.LinAlgError:
    #                     continue  # If still not working, try fewer lags
    #         if self.verbose:
    #             print(f"VAR model fitting failed due to non-positive definite matrix. Error: {str(e)}")
    #     except Exception as e:  # Catch other exceptions
    #         if self.verbose:
    #             print(f"VAR model fitting failed to converge. Error: {str(e)}")
    #         raise e
    #     return None  # Return None if all attempts fail

    def _fit_var_model(self):
        """ 
        Fit a Vector Auto Regression Model to the features and target variable.

        Raises:
            - SystemExit: If the model can't be fitted after all attempts.
        """
        try:
            model = VAR(self.df_scaled, exog=self.df_scaled[self.target])
            trend_types = {'n', 'c', 'ct'}
            
            max_attempts = 2  # Number of attempts to fit the model
            for attempt in range(max_attempts):
                try:
                    for lags in range(max_attempts, 1, -1):  # Start with higher lags and reduce
                        fit = model.fit(maxlags=lags,trend = 'ct')
                        if self.verbose:
                            print(f'VAR Model Fit Successful with {lags} lags on attempt {attempt + 1}')
                        return  fit
                except np.linalg.LinAlgError as e:
                    if 'not positive definite' in str(e):
                        if self.verbose:
                            print(f"Non-positive definite matrix encountered on attempt {attempt + 1}. Error: {str(e)}")
                        continue  # Continue to the next attempt with reduced lags
                    else:
                        if self.verbose:
                            print(f"Unexpected linear algebra error on attempt {attempt + 1}. Error: {str(e)}")
                        raise e  # Re-raise unexpected errors
                except ValueError as ve:
                    if "Insufficient degrees of freedom" in str(ve):
                        if self.verbose:
                            print(f"Insufficient degrees of freedom on attempt {attempt + 1}. Error: {str(ve)}")
                        continue  # Try with fewer lags
                    else:
                        if self.verbose:
                            print(f"Unexpected ValueError on attempt {attempt + 1}. Error: {str(ve)}")
                        raise ve  # Re-raise unexpected errors
                except Exception as e:
                    if self.verbose:
                        print(f"An unexpected error occurred on attempt {attempt + 1}. Error: {str(e)}")
                    raise e  # Re-raise other exceptions

            # If we've made it here, all attempts to fit the model have failed
            if self.verbose:
                print(f"Fatal Error: Failed to fit VAR model after {max_attempts} attempts.\n")
            raise SystemExit("Could not fit VAR model. Program terminated.")

        except Exception as e:  # Catch any exception from outside the loop
            if self.verbose:
                print(f"Fatal Error: An exception occurred outside fitting loop: {str(e)}")
            raise SystemExit(f"Unexpected error in VAR model fitting. Program terminated: {str(e)}")
    
    def _granger_causality(self, ci = 0.05):
        """
        Perform Granger causality from the Var Model Fit. 

        The degrees of freedom in the F-test are based on the number of variables in the VAR system,
        that is, degrees of freedom are equal to the number of equations in the VAR times dof of a single equation.
        if the p-value is less than the significance level (0.05), then, the corresponding X series causes the Y series.
            
        Test H0: “causing does not Granger-cause the remaining variables of the system”  
        Test H1: “causing is Granger-causal for the remaining variables”.

        
        """
        if self.verbose:
            print(f"Checking Granger Causality tests...\n")
    
        fit = self.model
        
        # Columns to check for causality
        granger_caused = []
        
        # Iterate through each of the features and check for causality        
        for i in self.df.drop(columns=[self.target]).columns:
            try:
                # Error happens in the below function: 
                # t = fit.test_causality(caused=self.cause, causing=i, kind='wald').summary()
                t = fit.test_inst_causality(causing=i).summary()
            except Exception as e:
                raise ValueError(f"Error in Granger Causality test: {str(e)}")
                
            if t[1][2].data < ci:
                granger_caused.append((self.cause, i))
                if self.verbose:
                    confi = int(100 - (100*ci))
                    # print(f'{i} Does Granger Cause {self.cause} @ {confi}% confidence level, p-value: {t[1][2].data}')
                    print(f'{i} has an instantaneous causal effect on {self.cause} @ {confi}% confidence level, p-value: {t[1][2].data:.3f}')
        
        if self.verbose:
            print(f"\nFinished Checking Granger Causality tests...\n")
        return granger_caused

    def _contemporaneous_causality(self, ci = 0.05):
        """
        Perform contemporaneous causality tests, using the grangercausalitytests function.
        
            The Null hypothesis for grangercausalitytests is that the time series in
            the second column, x2, does NOT Granger cause the time series in the first
            column, x1. 
            
            Grange causality means that past values of x2 have a
            statistically significant effect on the current value of x1, taking past
            values of x1 into account as regressors.
            
            We reject the null hypothesis:
                that x2 does not Granger cause x1 
                if the pvalues are below a desired size of the test.

            The null hypothesis for all four test is that the coefficients
            corresponding to past values of the second time series are zero
            i.e. x2 does not cause x1.
            
        Args:
            ci (float): Confidence interval for the test.
        Returns
            - list: List of tuples containing the features that contemporaneously cause the target variable.
        """
        # Get a list of tuples with one of the values being: target
        checks = combinations(self.df.columns.to_list(), 2)
        checks = [i for i in checks if self.cause in i]
        checks = [(i, j) for i, j in checks if j == self.cause]
        contemporaneous_causality = []
        for caused, target in checks:
            t = grangercausalitytests(self.df_scaled[[target, caused]], maxlag = 4)
            if t[1][0]['ssr_ftest'][1] > ci:
                contemporaneous_causality.append((target, caused))
                if self.verbose:
                    print(f'{caused} Does Contemporaneously Cause {target} @ {int(100 - (100*ci))}% confidence level, p-value: {t[1][0]["ssr_ftest"][1]:.3f}')
        return contemporaneous_causality

    

    def analyze(self, ci = 0.05):
        """
        Run correlation analysis and return the results

        Returns:
        - dict: Dictionary containing the granger_causality and contemporaneous_causality results.
        """
        return {
            "stationarity_tests": self.check_stationarity(ci),
            "var_model": self.model,
            "granger_causality": self._granger_causality(ci),
            # "contemporaneous_causality": self._contemporaneous_causality(),
        }


if __name__ == "__main__":
    print(""" 7.4: Earth, water, fire, air, ether, mind, spiritual intelligence and false ego; thus these are the eightfold divisions of my external energy.\n""")
    ###########################################################
    
    import sys
    sys.path.append("../Stocks")
    from models.anom.stocks.connect import data
    d = data('../Stocks/')
    stocks = d.Optionsdb.all_stocks
    stock = np.random.choice(stocks)
    x, y  = d._returnxy(stock, keep_close=False)
    random_features = np.random.choice(x.columns, 5)
    x = x[random_features]
    
    
    y.name = '$' + str(stock).upper()
    a = analyze_correlation(x, y, verbose=True)
    results = a.analyze(ci = 0.1)
    
    # print(a.model.summary())
    
    d.close_connection()
    ###########################################################
    # from pickle import dump, load
    # data = load(open('examples/data/data.pkl', 'rb'))
    # x = data['xvar']; y = data['yvar']
    # y.name = 'target'
    # print(x.shape, y.shape)
    # a = analyze_correlation(x, y, verbose=True)
    # results = a.analyze()
    # print(results)
    
    ###########################################################
    # np.random.seed(1)
    # n = 500
    # indexvals = np.arange(n)    
    # x = np.random.normal(n, 1, size = (n, 2))
    # y = np.random.normal(0, 1, size = n)
    # dte_index = pd.DatetimeIndex(indexvals)
    # df = pd.DataFrame(x, index = dte_index, columns = ['x1', 'x2'])
    # df['x3'] = df['x1'] * 3.4 + df['x2'] * 2.3 + y * 0.5
    # df['x4'] = df['x3'] + df['x2'].shift(1)
    # df['x4'] = df['x4'].fillna(df['x4'].mean())
    # df['y'] = y
    
    # ac = analyze_correlation(df[['x1', 'x2', 'x3']], df['y'], verbose=True)
    # results = ac.analyze()
    
    
    ########################################################### 
    # import statsmodels.api as sm
    # macrodata = sm.datasets.macrodata.load_pandas().data
    # macrodata.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
    # macrodata.index = macrodata.index.to_timestamp()
    # macrodata = macrodata.drop(columns = ['year', 'quarter'])
    # macrodata = macrodata.diff().dropna()
    # x = macrodata.drop(columns = 'realgdp')
    # y = macrodata['realgdp']
    # ac = analyze_correlation(x, y, verbose=True)
    # results = ac.analyze()