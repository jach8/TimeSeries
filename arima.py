import re
import warnings
import numpy as np 
import pandas as pd 
import datetime as dt 
from tqdm import tqdm 
from arch import arch_model
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class arima_trend:
    """
    Time Series Analysis (arima_trend) class for handling time series data operations, including conversion, 
    We will use this class to create a custom SARIMAX model for time series forecasting. 
    """

    def __init__(self):
        pass

    def get_series(self, x: pd.DataFrame) -> pd.Series:
        """
        Convert a dataframe to a time series object, by inferring the frequency of the data.
        
        Args:
            x (pd.DataFrame): DataFrame to convert to time series.
        
        Returns:
            pd.Series: Time series object as a pandas Series, with a period index.
        """
        # Check for duplicate labels 
        if x.index.duplicated().any():
            x = x.loc[~x.index.duplicated()]
            
        freq = pd.infer_freq(x.index)
        x.index = pd.to_datetime(x.index)
        return x.asfreq(freq).bfill()

    def adf_test(self, x: pd.Series) -> pd.DataFrame:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        -   Null Hypothesis (H0): the time series has a unit root. 
        -   Alternate Hypothesis (H1): the time series has no unit root

        Large p-values indicate a unit root in the timeseries, confirming its non-stationarity. 

        Args:
            timeseries (pd.Series): Time series data to test for stationarity.
        
        Returns:
            pd.DataFrame: DataFrame containing test results including p-value, test statistic, etc.
        """
        timeseries = self.get_series(x)
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput[f'Critical Value ({key})'] = value
            
        col_name = x.name + ' ADF Test' if type(x) == pd.Series else 'ADF Test'
        return pd.DataFrame(dfoutput, columns = [col_name])
    
    
    
    def results_dataframe(self, x:pd.DataFrame, model: ARIMA) -> pd.DataFrame:
        """
        Returns the following Accuarcy Measures: 
            - Mean Absolute Error (MAE)
            - Mean Squared Prediction Error (MSPE)
            - Mean Absolute Percentage Error (MAPE)
            - Prediction Measure (PM)
            - R2 Score
            - AIC
            - BIC
            - HQIC
            - ADF p-value
        """
        # Get the model orders from the ARIMA object: 
        mo = model.model_orders # A dictionary 
        p = mo['ar'] # Autoregressive Factor
        d = mo['trend'] # Differencing Factor
        q = mo['ma'] # Moving Average Factor
        
        prediction = model.predict()
        mse = mean_squared_error(x, prediction)
        mae = mean_absolute_error(x, prediction)
        r2 = r2_score(x, prediction)
        resid = x - prediction
        mape = np.mean(np.abs((resid) / x))
        mspe = np.mean(resid)**2
        pm = np.sum((resid)**2) / np.sum((x - np.mean(x))**2)
        

        results = pd.DataFrame({
            'name': x.name if hasattr(x, 'name') else 'series',
            'model': f'ARIMA({p}, {d}, {q})',
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'mspe': mspe,
            'pm': pm,
            'r2': r2,
            'aic': model.aic,
            'bic': model.bic,
            'hqic': model.hqic,
            'adf': adfuller(x)[1]
        }, index=[f'({p}, {d}, {q})'])
        
        return results
    
    

    def arima_model(self, x: pd.DataFrame, maxord: dict = dict(p=5, d=0, q=5), result_df: bool = True) -> dict:
        """

        ARIMA model for time series data. 
        This function uses the max_order parameter to determine the optimal order of the ARIMA model.
        
        The ARIMA model uses parameters to describe these three components:
            p (autoregressive order): The number of lagged values used in the AR component
            d (differencing order): The degree of differencing required for non-stationarity
                - The model will difference the data d times to make it statistically stationary
                - That means by passing in a difference series x, there is no need to difference the data
            q (moving average order): The number of lagged errors used in the MA component
            
        
        Returns the following Accuarcy Measures: 
            - Mean Absolute Error (MAE)
            - Mean Squared Prediction Error (MSPE)
            - Mean Absolute Percentage Error (MAPE)
            - Prediction Measure (PM)
            - R2 Score
            - AIC
            - BIC
            - HQIC
            - ADF p-value
        
        Args:
            x (pd.DataFrame): Time series data to model.
            max_order (dict): dict with keys 'p', 'd', 'q' denoting the maximum value to use in ARIMA model. 
                                default is {'p': 5, 'd': 2, 'q': 5}
            result_df (bool): Return the results dataframe if True, else return the best model.
        
        Returns:
            dict or ARIMA: Returns either a dictionary with results and model or the best model.
        """
        if maxord is None:
            maxord = {'p': 5, 'd': 2, 'q': 5}
        
        x = self.get_series(x)
        
        # Check for stationarity
        _, p_value, _, _, _, _ = adfuller(x)
        is_stationary = p_value < 0.05
        max_d = maxord['d'] if not is_stationary else 0
        print(f"Data is {'not ' if not is_stationary else ''}stationary, Assessing max_d = {max_d}")

        # Generate model orders
        p_range = np.arange(1, maxord['p'] + 1)
        d_range = np.arange(max_d + 1)
        q_range = np.arange(1, maxord['q'] + 1)
        max_order = np.array(np.meshgrid(p_range, d_range, q_range)).T.reshape(-1, 3)

        print(f'Fitting: {len(max_order)} models')

        out = {}
        best_mse = np.inf
        
        # Early stopping parameters
        stop_mse = 1e-7  # You can adjust this based on your requirement
        
        for p, d, q in tqdm(max_order, desc="Fitting ARIMA models"):
            try:
                model = ARIMA(x, order=(int(p), int(d), int(q))).fit()
                results = self.results_dataframe(x, model)
                mse = results['mse'].values[0]
            
                # Early stopping if we've found a good enough model
                if mse < stop_mse:
                    print(f"Stopping early: MSE {mse} below threshold {stop_mse}")
                    best_mse = mse
                    out['best'] = {'model': model, 'results': results}
                    break

                if mse < best_mse:
                    best_mse = mse
                    out['best'] = {'model': model, 'results': results}
    
                out[(p, d, q)] = {'model': model, 'results': results}

            except Exception as e:
                print(f"Failed to fit model ARIMA({p}, {d}, {q}): {str(e)}")

        if 'best' in out:
            if result_df:
                summary = pd.concat([v['results'] for k, v in out.items() if k != 'best'])
                out['summary'] = summary
                out['models_fit'] = [tuple(x) for x in max_order]
            return out
        else:
            return {}
        
    
    def _split_training_testing_data(self, x: pd.DataFrame, train_set: float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets. 
        
        Args:
            x (pd.DataFrame): Time series data to model.
            train_set (float): Proportion of the data to use for training.
        
        Returns:
            X_train (pd.DataFrame): DataFrame containing the training data.
            X_test (pd.DataFrame): DataFrame containing the testing data.
        
        """
        xtrain, xtest = train_test_split(x, train_size=train_set, shuffle=False)
        return xtrain, xtest
        

    # def train_model(self, x: pd.DataFrame, model: Dict[str, Union[pd.DataFrame, ARIMA]], train_set: float = 0.9) -> pd.DataFrame:
    #     """
    #     Fit the model by finding the optimal parameters on the training set 
    #     Then evaluate the model on the test set and obtain the results information. 
        
    #     Args:
    #         x (pd.DataFrame): Time series data to model.
    #         model (dict): Dictionary containing the results of the ARIMA or ARMA model.
    #         train_set (float): Proportion of the data to use for training.
        
    #     Returns:
    #         pd.DataFrame: DataFrame with accuracy measures of the model.
    #     """
        
    #     pass

    def preds(self, mod: ARIMA, fh: int, ci: float = 0.10) -> pd.DataFrame:
        """
        Obtain predictions from the ARIMA or ARMA model.
        
        Args:
            mod (ARIMA): ARIMA model to predict.
            fh (int): Number of steps to forecast.
            ci (float): Confidence interval for the prediction.
        
        Returns:
            pd.DataFrame: DataFrame with prediction results.
        """
        pred = mod.get_forecast(steps=fh)
        pred_ci = pred.conf_int(alpha=ci)
        pred_out = pred_ci.merge(pred.predicted_mean, left_index=True, right_index=True)
        pred_out.columns = ['Lower', 'Upper', 'Mean Prediction']
        pred_out = pred_out[['Mean Prediction', 'Lower', 'Upper']]
        pred_out.index = pd.to_datetime(pred_out.index)
        return pred_out

    def plot_preds(self, mod: ARIMA, fh: int = 3, ci: float = 0.10):
        """
        Plots the predictions of the ARIMA or ARMA model. Note: 'y' is not defined in the class context,
        thus, this method should be used with caution or modified to accept 'y' as an argument.

        Args:
            mod (ARIMA): ARIMA model to predict.
            fh (int): Number of steps to forecast.
            ci (float): Confidence interval for the prediction.
        """
        pred_out = self.preds(mod, fh, ci)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot the data before the forecast (2 * fh) 
        og_data = mod.data.orig_endog.iloc[-(2 * fh):]
        ax.plot(og_data, label='Original Data')
        # Plot the Forecast
        ax.plot(pred_out['Mean Prediction'], label='Prediction')
        # Plot the Confidence Interval
        ax.fill_between(
            pred_out.index, 
            pred_out['Lower'], 
            pred_out['Upper'], 
            alpha=0.3, 
            color='grey'
        )
        ax.set_title(mod.data.ynames)
        ax.legend()
        fig.autofmt_xdate()
        plt.show()
        
        
if __name__ == "__main__":
    import pandas as pd 
    
    # Load Data
    # data = pd.read_csv('examples/data/ohlcv.csv', parse_dates=['timestamp'], index_col='timestamp')
    data = pd.read_csv("examples/data/tsne_data.csv", parse_dates=['date'], index_col='date')
    
    # Initialize ARIMA class
    arima = arima_trend()
    
    y = data['Close'].diff().dropna()
    print(arima.adf_test(y))
    
    arima_model = arima.arima_model(y)
    print(arima_model['best']['results'])