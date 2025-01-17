import pandas as pd 
import pandas_datareader.data as web
import datetime as dt 
import numpy as np 
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import re
import warnings
from typing import Dict, Optional, Tuple
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
        return pd.DataFrame(dfoutput, columns = ['ADF Test']).round(2)

    def arima_model(self, x: pd.DataFrame, max_order: int = 5, result_df: bool = True) -> Dict[str, Union[pd.DataFrame, ARIMA]]:
        """
        ARIMA model for time series data. 
        This function uses the max_order parameter to determine the optimal order of the ARIMA model.
        The best model is selected based on the minimum mean squared error.
        
        Args:
            x (pd.DataFrame): Time series data to model.
            max_order (int): Maximum order of the ARIMA model.
            result_df (bool): Return the results dataframe if True, else return the best model.
        
        Returns:
            dict or ARIMA: Returns either a dictionary with results and model or the best model.
        """
        x = self.get_series(x)
        results = pd.DataFrame(columns = ['name', 'model', 'mse', 'mae', 'r2', 'aic', 'bic', 'hqic', 'adf'])
        for p in range(1, max_order):
            for q in range(1, max_order):
                for d in range(2):
                    try:
                        model = ARIMA(x, order = (p, d, q)).fit()
                        results = results.append({
                            'name': x.columns[0], 
                            'model': f'ARIMA({p}, {d}, {q})', 
                            'mse': mean_squared_error(x, model.predict()),
                            'mae': mean_absolute_error(x, model.predict()),
                            'r2': r2_score(x, model.predict()),
                            'aic': model.aic,
                            'bic': model.bic,
                            'hqic': model.hqic,
                            'adf': adfuller(x)[1]
                        }, ignore_index = True)
                    except:
                        pass
        # return the best model
        bm = results.sort_values(by = 'mse').iloc[0]
        p, d, q = map(int, re.findall(r'\d+', bm['model']))
        mod = ARIMA(x, order = (p, d, q)).fit()
        if result_df:
            return {'results': results, 'model': mod}
        else:
            return mod

    def arma_model(self, x: pd.DataFrame, max_order: int, result_df: bool = True) -> Dict[str, Union[pd.DataFrame, ARIMA]]:
        """
        ARMA model for time series data.
        This function uses the max_order parameter to determine the optimal order of the ARMA model.
        The best model is selected based on the minimum mean squared error.
        
        Args:
            x (pd.DataFrame): Time series data to model.
            max_order (int): Maximum order of the ARMA model.
            result_df (bool): Return the results dataframe if True, else return the best model
        
        Returns:
            dict or ARIMA: Returns either a dictionary with results and model or the best model.
        """
        x = self.get_series(x)
        results = pd.DataFrame(columns = ['name', 'model', 'mse', 'mae', 'r2', 'aic', 'bic', 'hqic', 'adf'])

        for p in range(1, max_order):
            for q in range(1, max_order):
                try:
                    d = 0
                    model = ARIMA(x, order = (p, d, q)).fit()
                    results = results.append({
                        'name': x.columns[0], 
                        'model': f'ARMA({p}, {d}, {q})', 
                        'mse': mean_squared_error(x, model.predict()),
                        'mae': mean_absolute_error(x, model.predict()),
                        'r2': r2_score(x, model.predict()),
                        'aic': model.aic,
                        'bic': model.bic,
                        'hqic': model.hqic,
                        'adf': adfuller(x)[1]
                    }, ignore_index = True)
                except:
                    pass
        
        # return the best model 
        bm = results.sort_values(by = 'mse').iloc[0]
        p, d, q = map(int, re.findall(r'\d+', bm['model']))
        mod = ARIMA(x, order = (p, d, q)).fit()
        if result_df:
            return {'results': results, 'model': mod}
        else:
            return mod

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