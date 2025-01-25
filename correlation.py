import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from sklearn.decomposition import PCA
from granger import grangercausalitytests
from io import StringIO

class analyze_correlation:
    def __init__(self, x, y, decompose = True, verbose=False):
        """
        Initialize the analyze_correlation with features and target.

        Parameters:
        - x (pd.DataFrame): Features for analysis.
        - y (pd.Series): Target variable for causality and classification.
        - decompose (bool): If True, perform PCA decomposition on the features. 
        - verbose (bool): If True, print detailed model metrics and updates.
        - max_iterations (int): Maximum number of iterations for VAR model fitting.
        """
        self.verbose = verbose
        self.cause = y.name
        self.decompose = decompose
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
    
    def pca_decomposition(self, df, n_components = 3):
        """
        Perform PCA decomposition on the features in the dataframe .
        
        Args:
            - df (pd.DataFrame): DataFrame to perform PCA decomposition on.
            - n_components (int): Number of components to decompose to.
        
        Returns:
            - pd.DataFrame: DataFrame with the PCA components.
        """
        if self.verbose: 
            print(f"Performing PCA decomposition on the features...")
        pca = PCA(n_components = n_components).fit(df)
        df = pd.DataFrame(
            pca.transform(df), 
            columns=[f'PC{i}' for i in range(1, n_components + 1)], 
            index=df.index
        )
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
        
        # Merge Endogenous (X) and Exogenous (Y) variables
        # Set up the period index. 
        df =self.setup_period_index( 
            x.merge(y, left_index=True, right_index=True).dropna()
        )
        
        # # PCA Decomposition
        if self.decompose:
            centered_x = x - x.mean()
            # scaled 
            self.df_scaled = pd.DataFrame(
                self.scaler.fit_transform(centered_x), 
                columns=x.columns, 
                index=x.index
            )
            self.df_scaled = self.pca_decomposition(self.df_scaled)
            self.features = self.df_scaled.columns.to_list()
            self.df_scaled[y.name] = y.values
            self.target = y.name
            
        else:
            # ###Scaled DataFrame
            self.df_scaled = pd.DataFrame(
                self.scaler.fit_transform(x), 
                columns=x.columns, 
                index=x.index
            )
            self.df_scaled[y.name] = y.values
            self.features = x.columns.to_list()
            self.target = y.name  
        
        self.feature_change = False
        # # Initialize the model  
        # try: 
        #     self.check_stationarity()
        #     self.model = self._fit_var_model()
        # except Exception as e:
        #     if self.verbose:
        #         print(f"Error in fitting the VAR model: {str(e)}")
        #     self.model = None
        
        # if self.verbose:
        #     print(f"Dataframe shape: {df.shape}\n")

    def adf_test(self, timeseries, lag_method = 'AIC' ):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        -   Null Hypothesis (H0): the time series has a unit root. (Non-Stationary)
        -   Alternate Hypothesis (H1): the time series has no unit root (Stationary)
        
        Large p-values indicate a unit root in the timeseries, confirming its non-stationarity. 
        
        Args:
            timeseries (pd.Series): Time series data to test for stationarity.
            lag_method (str): Method to use for determining the number of lags in the test:
                - 'AIC' or 'BIC' minimize the corresponding information criterion to get the number of lags in the test.
                - 't-stat': t-statistic: based on maxlag and drops a lag until the t-statistic on the last length is significant at the 95 % level.
                - 'None': then number of lags is set to maxlag
        
        Returns:
            p-value (float): P-value of the ADF test. (Probability that the null hypothesis is rejected. )
        """    
        #Perform Dickey-Fuller test:
        dftest = adfuller(timeseries, autolag=lag_method)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    
        # return pd.DataFrame(dfoutput, columns = ['ADF Test']).round(2)        
        return dfoutput['p-value']
    
        
    def __differnce_series(self, column, significance_level=0.05, max_differences=5):
        """
        Difference a time series to make it stationary, using the ADF test for stationarity.
        This function will difference the series until it is stationary or until the maximum number of differences is reached.

        Args:
            column (str): The column name in self.df_scaled to difference.
            significance_level (float): P-value threshold for considering a series stationary.
            max_differences (int): Maximum number of times to difference the series.

        Returns:
            None: Modifies self.df_scaled in-place.

        Raises:
            ValueError: If the series cannot be made stationary within max_differences.
        """
        diff_count = 0
        original_column = column  # Store the original column name for reference
        while diff_count <= max_differences:
            series = self.df_scaled[column].dropna()
            try:
                p_value = self.adf_test(series)
                if p_value < significance_level:
                    if diff_count > 0:
                        # Rename the column in df_scaled to reflect differencing
                        self.df_scaled.rename(columns={column: original_column + f'_diff_{diff_count}'}, inplace=True)
                    return  # Series is now stationary, no need to return anything as we modify in-place
                else:
                    # Difference the series
                    diff_count += 1
                    column = original_column + f'_diff_{diff_count}'  # New column name
                    self.df_scaled[column] = series.diff().dropna()
                    self.df_scaled.drop(columns=[original_column], inplace=True)  # Drop the original column
                    self.feature_change = True
                    if self.verbose:
                        print(f"Differenced {original_column} ({diff_count} times) --> new_name: {column}, p-value: {p_value:.3f}")
            except Exception as e:
                if self.verbose:
                    print(f"Error in differencing series {original_column}: {str(e)}")
                return  # Exit the function on exception

        if diff_count > max_differences:
            if self.verbose:
                print(f"Series {original_column} could not be made stationary after {max_differences} differences.")
            raise ValueError(f"Series {original_column} remains non-stationary after {max_differences} differences.")

    def check_stationarity(self, significance_level=0.05, max_differences=3):
        """
        Performs the Augmented Dickey-Fuller test on the features to check for stationarity.
        - If the p-value is less than the confidence interval (small p-value), the feature is stationary.
        - If the p-value is greater than the confidence interval (large p-value), the feature is non-stationary.
        This function will attempt to make non-stationary series stationary by differencing.

        Args: 
            significance_level (float): Significance level for ADF test. Default: 0.05
            max_differences (int): Maximum number of times to difference a feature. Default: 3

        Raises:
            ValueError: If the number of differences exceeds max_differences for any feature.
        """
        if self.verbose:
            print(f"Checking stationarity of features using ADF test at {significance_level} confidence level...")

        stationary = []
        non_stationary = []
        original_columns = list(self.df_scaled.columns)

        for col in original_columns:
            self.__differnce_series(
                column = col, 
                significance_level=significance_level,
                max_differences = max_differences
            )

        # Reset features after differencing 
        self.features = self.df_scaled.drop(columns=[self.target]).columns.to_list()
        self.df_scaled = self.df_scaled.dropna()
        if self.verbose:
            print("All features are now stationary.\n")
        return self.df_scaled
    
    
    def _get_best_model(self, df, criterion='fpe', trend = 'ct'):
        """
        Fits the VAR model and selects the best model based on the the information criterion specified. 
        Default Criterion is AIC (Alkaike Information Criterion)
        This function will store the results internally as self.model_search_results
        
            In the VAR().select_order() method, we have 2 arguments:
                1. maxlags: The maximum number of lags to check for the model. Not specifying this will check for lags automatically
                2. trend: The trend parameter to include in the model. Options are ['n', 'c', 'ct', 'ctt']
                    - 'n': No Deterministic Terms
                    - 'c': Constant trend
                    - 'ct': Constant and linear trend
                    - 'ctt': Constant, linear and quadratic trend
                If the model fails, try to run it with one of the availible trends

        Args:
            df (pd.DataFrame): DataFrame containing the features and target variable.
            criterion (str, optional): Acceptable Criterions are ['aic','bic','fpe','hqic'],. Defaults to 'aic'.
            maxlags
            
        Returns:
            - lag (int): Number of lags for the best model, based on the criterion. 
        """
        
        # Initialize the model search results  
        order_df = VAR(df).select_order(trend = trend).summary()
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
        best_model = search_results.loc[search_results[criterion.lower()].str.contains('\*')]['lags'].values[0]

        if self.verbose:
            print(f"Lag Model Selection based on {criterion.upper()} criterion: {best_model}")
        
        return best_model


    def _fit_var_model(self, criterion='fpe', trend='c'):
        """
        Fit a Vector Auto Regression Model to the features and target variable, using the statsmodels VAR class.
        Selects the best model based on the specified information criterion using the helper function _get_best_model.

        Args:
            criterion (str): Information criterion to use for model selection. Options are 'aic', 'bic', 'hqic', 'fpe'.
                            Default is 'bic'.
            trend (str): Type of trend to include in the model. Options are 'n', 'c', 'ct', 'ctt'. Default is 'c'.

        Returns:
            - model: Best fitted VAR model based on the specified criterion.

        Raises:
            - SystemExit: If the model can't be fitted with the chosen trend and criterion.
        """
        try:
            model = VAR(self.df_scaled, exog=self.df_scaled[self.target])
            
            best_lag = self._get_best_model(self.df_scaled, criterion=criterion)

            # Fit the model with the best number of lags from _get_best_model
            fit = model.fit(maxlags=best_lag, trend=trend)
            self.model = fit
            
            if self.verbose:
                # Note: Indexing should be adjusted because indices start at 0 and lags at 1
                best_stats = self.model_search_results[self.model_search_results['lags'] == best_lag].iloc[0]
                print(f'Best VAR Model Fit with {best_lag} lags - Trend: {trend}, {criterion.upper()}: {best_stats[criterion]}')
            return fit

        except ValueError as ve:
            if self.verbose:
                print(f"ValueError in fitting the VAR model: {str(ve)}")
            raise SystemExit(f"Could not fit VAR model with trend {trend}. Program terminated due to: {str(ve)}")
        except Exception as e:
            if self.verbose:
                print(f"Error in fitting the VAR model: {str(e)}")
            raise SystemExit(f"Could not fit VAR model with trend {trend}. Program terminated due to: {str(e)}")

    
    def _granger_causality(self, significance_level = 0.05):
        """
        Perform Granger causality from the Var Model Fit. 

        The degrees of freedom in the F-test are based on the number of variables in the VAR system,
        that is, degrees of freedom are equal to the number of equations in the VAR times dof of a single equation.
        if the p-value is less than the significance level (0.05), then, the corresponding X series causes the Y series.
            
        Test H0: “causing does not Granger-cause the remaining variables of the system”  
        Test H1: “causing is Granger-causal for the remaining variables”.

            - Small p-values (< 5%): Reject H0 and conclude that there is statistically significant Granger causality.
            - Moderate p-values (5-20%): The results may be marginally significant or inconclusive, and further investigation is needed.
            - Large p-values (> 20%): Fail to reject H0, suggesting no statistically significant Granger causality
        
        """
        if self.verbose:
            print(f"Checking Granger Causality tests...\n")
    
        fit = self.model
        
        # Columns to check for causality
        granger_caused = []
                
        # Iterate through each of the features and check for causality        
        # for i in self.df_scaled.drop(columns=[self.target]).columns:
        for i in self.features:
            try:
                # Error happens in the below function: 
                t = fit.test_causality(caused=self.cause, causing=i, kind='wald').summary()
            except Exception as e:
                if 'singular matrix' in str(e):
                    warnings.warn(f"\n\n{i} Singular matrix encountered for Granger Causality test.\n\n")
                    continue
                raise ValueError(f"Error in Granger Causality test: {str(e)}")
                
            if t[1][2].data < significance_level:
                if t[1][2].data == 0:
                    # Raise warnings for perfect causality
                    warnings.warn(f"{i} Perfect Granger Causality detected for {self.cause}.")
                granger_caused.append((self.cause, i))
                if self.verbose:
                    confi = int(100 - (100*significance_level))
                    print(f'{i} Does Granger Cause {self.cause} @ {confi}% confidence level, p-value: {t[1][2].data}')

        
        if self.verbose:
            print(f"\nFinished Checking Granger Causality tests...\n")
        return granger_caused
    
    
    def _instantaneous_causality(self, significance_level = 0.05):
        """
        Perform Instantaneous causality test (A form of Granger Causality test) from the Var Model Fit.
        Instantaneous causality reflects a non-zero correlation between the variables in the system. 
        
        To Do: Handle the case where the matrix is singular. By definition the 
        
        """
        if self.verbose:
            print(f"Checking For Instaneous Causality...\n")
    
        fit = self.model
        
        # Columns to check for causality
        instaneous_cause = []
        
        # Iterate through each of the features and check for causality        
        # for i in self.df_scaled.drop(columns=[self.target]).columns:
        for i in self.features:
            try:
                t = fit.test_inst_causality(causing=i).summary()
            except Exception as e:
                raise ValueError(f"{i} Error in Instaneous Causality test: {str(e)}")
                
            if t[1][2].data < significance_level:
                if t[1][2].data == 0:
                    # Raise warnings for perfect causality
                    warnings.warn(f"{i} Perfect Instaneous Causality detected for {self.cause}.")
                instaneous_cause.append((self.cause, i))
                if self.verbose:
                    confi = int(100 - (100*significance_level))
                    print(f'{i} has an instantaneous causal effect on {self.cause} @ {confi}% confidence level, p-value: {t[1][2].data}')
        
        if self.verbose:
            print(f"\nFinished Checking Instaneous Causality...\n")
        return instaneous_cause

    def _contemporaneous_causality(self, significance_level = 0.05):
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
            significance_level (float): Confidence interval for the test.
        Returns
            - list: List of tuples containing the features that contemporaneously cause the target variable.
        """
        if self.verbose: 
            print(f"Checking for Contemporaneous Causality...\n")
        # Get a list of tuples with one of the values being: target
        checks = combinations(self.df_scaled.columns.to_list(), 2)
        checks = [i for i in checks if self.cause in i]
        checks = [(i, j) for i, j in checks if j == self.cause]
        
        # Save contemporaneous causality to a list 
        contemporaneous_causality = []
        
        for caused, target in checks:
            t = grangercausalitytests(self.df_scaled[[target, caused]], maxlag = 4)
            if t[1][0]['ssr_ftest'][1] < significance_level:
                contemporaneous_causality.append((target, caused))
                if self.verbose:
                    cl = int(100 - (100*significance_level))
                    print(f'{caused} Does Contemporaneously Cause {target} @ {cl}% confidence level, p-value: {t[1][0]["ssr_ftest"][1]}')
        if self.verbose:
            print(f"\nFinished Checking Contemporaneous Causality...\n")
        return contemporaneous_causality

    def analyze(self, significance_level = 0.05):
        """
        Run correlation analysis and return the results

        Returns:
        - dict: Dictionary containing the granger_causality and contemporaneous_causality results.
        """
        return {
            "stationarity_tests": self.check_stationarity(significance_level=significance_level),
            "var_model": self._fit_var_model(),
            "granger_causality": self._granger_causality(significance_level),
            "instantaneous_causality": self._instantaneous_causality(significance_level),
            "contemporaneous_causality": self._contemporaneous_causality(),
        }


if __name__ == "__main__":
    print(""" 7.4: Earth, water, fire, air, ether, mind, spiritual intelligence and false ego; thus these are the eightfold divisions of my external energy.\n""")
    ###########################################################
    
    # import sys
    # sys.path.append("../Stocks")
    # from models.anom.stocks.connect import data
    # d = data('../Stocks/')
    # stocks = d.Optionsdb.all_stocks
    # stock = np.random.choice(stocks)
    # x, y  = d._returnxy(stock, keep_close=False)
    # # Remove all columns with 'chng' in the name
    # # x = x[[i for i in x.columns if 'chng' not in i]]
    # random_features = np.random.choice(x.columns, 15, replace = False)
    # x = x[random_features]
    
    
    # y.name = '$' + str(stock).upper()
    # a = analyze_correlation(x, y, verbose=True)
    # results = a.analyze(significance_level = 0.01)
    # b = analyze_correlation(x, y, verbose=True, decompose=False)
    # results = b.analyze(significance_level = 0.01)
    
    # print(a.model.summary())
    
    # d.close_connection()
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
    # n = 500
    # start_date = '2020-01-01'
    # end_date = pd.Timedelta(days=n) + pd.to_datetime(start_date)
    # dte_index = pd.date_range(start_date, end_date, freq='D')
    # n = len(dte_index)
    
    # x = np.random.normal(n, 1, size = (n, 4))
    # y = np.random.normal(0, 1, size = n)

    # df = pd.DataFrame(x, index = dte_index, columns = ['x1', 'x2', 'x3', 'x4'])
    # df['y'] = y
    
    # ac = analyze_correlation(df[['x1', 'x2', 'x3','x4']], df['y'], verbose=True)
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
    
    
    ########################################################### 
    data = pd.read_csv('examples/data/stock_returns.csv', parse_dates=['Date'], index_col='Date').iloc[1:]
    data = data["2000-01-01":].dropna(axis = 1)
    random_20_stocks = np.random.choice(data.columns, 60, replace = False)
    random_y = np.random.choice(random_20_stocks, 1, replace = False)[0]
    x = data.drop(columns = random_y).iloc[:-1]
    y = data[random_y].iloc[:-1]
    ac = analyze_correlation(x, y, verbose=True, decompose=False)
    results = ac.analyze()
    # print(results)