import numpy as np 
import pandas as pd 
##########################################################
from pickle import dump, load

def test_data1():
    data = load(open('examples/data/data.pkl', 'rb'))
    x = data['xvar']; y = data['yvar']
    y.name = 'target'
    return x, y

##########################################################

def random_test_data(n = 500, start_date = '2020-01-01', return_xy = False):
    end_date = pd.Timedelta(days=n) + pd.to_datetime(start_date)
    dte_index = pd.date_range(start_date, end_date, freq='D')
    n = len(dte_index)

    x = np.random.normal(n, 1, size = (n, 4))
    y = np.random.normal(0, 1, size = n)

    df = pd.DataFrame(x, index = dte_index, columns = ['x1', 'x2', 'x3', 'x4'])
    df['y'] = y
    
    if return_xy:
        return df.drop(columns = 'y'), df['y']
    else:
        return df

########################################################## 
import statsmodels.api as sm

def test_data2(return_xy = False):
    macrodata = sm.datasets.macrodata.load_pandas().data
    macrodata.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
    macrodata.index = macrodata.index.to_timestamp()
    macrodata = macrodata.drop(columns = ['year', 'quarter'])
    macrodata = macrodata.diff().dropna()
    x = macrodata.drop(columns = 'realgdp')
    y = macrodata['realgdp']
    if return_xy:
        return x, y
    else:
        return macrodata


########################################################### 
def stock_returns_data(start_date = '2020-01-01', return_xy = False, target = "SPY"):
    data = pd.read_csv('examples/data/stock_returns.csv', parse_dates=['Date'], index_col='Date')
    data = data[start_date:].dropna(axis=1)
    if return_xy:
        return data.drop(columns = target), data[target]
    else:
        return data
