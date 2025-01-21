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
warnings.filterwarnings('ignore')

class arima_trend:
    """
    Time Series Analysis (arima_trend) class for handling time series data operations, including conversion, 
    stationarity testing, and ARIMA/ARMA modeling.
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
        print(f"Data is {'not ' if not is_stationary else ''}stationary")

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
                prediction = model.predict()
                mse = mean_squared_error(x, prediction)
                mae = mean_absolute_error(x, prediction)
                r2 = r2_score(x, prediction)
                
                results = pd.DataFrame({
                    'name': x.name if hasattr(x, 'name') else 'series',
                    'model': f'ARIMA({p}, {d}, {q})',
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'aic': model.aic,
                    'bic': model.bic,
                    'hqic': model.hqic,
                    'adf': adfuller(x)[1]
                }, index=[f'({p}, {d}, {q})'])
                
                # Early stopping if we've found a good enough model
                if mae < stop_mse:
                    print(f"Stopping early: MSE {mae} below threshold {stop_mse}")
                    best_mse = mse
                    out['best'] = {'model': model, 'results': results}
                    break

                if mse < best_mse:
                    best_mse = mse
                    out['best'] = {'model': model, 'results': results}

        
                # out[f'{p},{d},{q}'] = {'model': model, 'results': results}
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



    def train_model(self, x: pd.DataFrame, model: Dict[str, Union[pd.DataFrame, ARIMA]], train_set: float = 0.9) -> pd.DataFrame:
        """
        Train the ARIMA or ARMA model on the data and obtain accuracy measures.
        
        Args:
            x (pd.DataFrame): Time series data to model.
            model (dict): Dictionary containing the results of the ARIMA or ARMA model.
            train_set (float): Proportion of the data to use for training.
        
        Returns:
            pd.DataFrame: DataFrame with accuracy measures of the model.
        """
        x = self.get_series(x)

        train_ind = int(len(x) * train_set)
        test_ind = int(len(x) * (1 - train_set))

        train = x[:train_ind]
        test = x[train_ind:]

        p, q, d = model['model'].order
        m = str(model['model'].__class__).split('.')[-1].split("'")[0]

        model_fit = ARIMA(train, order = (p, d, q)).fit()

        pred = model_fit.get_forecast(len(test)).predicted_mean
        tm = pd.concat([test, pred], axis = 1)
        tm.columns = ['actual', 'predicted']
        tm['resid'] = tm['predicted'] - tm['actual']

        out = pd.DataFrame(columns = ['name', 'model', 'mae', 'mspe', 'mape', 'pm'])
        out = out.append({
            'name': x.columns[0],
            'model': f'{m}({p}, {d}, {q})',
            'mae': np.mean(np.abs(tm.resid)),
            'mspe': np.mean(tm.resid)**2,
            'mape': np.mean(np.abs((tm.resid) / tm.actual)),
            'pm': np.sum((tm.resid)**2) / np.sum((tm.actual - np.mean(tm.actual))**2)
        }, ignore_index = True)
        return out

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
        # This line will raise an error as 'y' is not defined. Consider passing 'y' as an argument or 
        # use the last part of the actual data from the model.
        # ax.plot(y.tail(20), label='Actual')
        ax.plot(pred_out['Mean Prediction'], label='Prediction')
        ax.fill_between(pred_out.index, pred_out['Lower'], pred_out['Upper'], alpha=0.3, color='grey')
        ax.set_title(mod.data.ynames)
        ax.legend()
        plt.show()
        
        
if __name__ == "__main__":
    import pandas as pd 
    
    # Load Data
    data = pd.read_csv('examples/data/ohlcv.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # Initialize ARIMA class
    arima = arima_trend()
    
    y = data['target'].diff().dropna()
    print(arima.adf_test(y))
    
    arima_model = arima.arima_model(y)
    print(arima_model['best']['results'])