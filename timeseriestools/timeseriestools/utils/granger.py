"""Implementation of Granger causality tests."""

import warnings
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
from statsmodels.regression.linear_model import OLS
from scipy import stats
from typing import Dict, Any, Union, List, Tuple


def grangercausalitytests(x: np.ndarray, maxlag: int, addconst: bool = True) -> Dict[int, Dict[str, Tuple]]:
    """Run Granger causality tests with multiple lag orders.
    
    Implements four tests for Granger non-causality:
    1. F-test based on sum of squared residuals
    2. Chi-squared test
    3. Likelihood ratio test  
    4. Parameter F-test
    
    The null hypothesis is that the time series in the second column does NOT 
    Granger cause the time series in the first column.
    
    Args:
        x (np.ndarray): 2-D array with time series in columns
        maxlag (int): Maximum lag order to test
        addconst (bool, optional): Include constant term. Defaults to True.
        
    Returns:
        Dict[int, Dict[str, Tuple]]: Test results for each lag order
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> x = np.random.randn(100, 2)
        >>> results = grangercausalitytests(x, maxlag=3)
    """
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values")
    
    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            f"lag is {int((x.shape[0] - int(addconst)) / 3) - 1}"
        )

    resli: Dict[int, Dict[str, Tuple]] = {}

    for mlag in range(1, maxlag + 1):
        # Create lagmat of both time series
        dta = lagmat2ds(x, mlag, trim="both", dropex=1)
        
        # Add constant
        if addconst:
            dtaown = add_constant(dta[:, 1:(mlag + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
        else:
            raise NotImplementedError("Must include constant")
            
        # Run OLS
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # Calculate test statistics
        tss = res2djoint.centered_tss if res2djoint.model.k_constant else res2djoint.uncentered_tss

        # Check for perfect fit
        if (tss == 0 or res2djoint.ssr == 0 or np.isnan(res2djoint.rsquared) or
            (res2djoint.ssr / tss) < np.finfo(float).eps):
            raise ValueError(
                "The Granger causality test statistic cannot be computed "
                "because the VAR has a perfect fit of the data."
            )

        # F test
        fgc1 = ((res2down.ssr - res2djoint.ssr) / res2djoint.ssr / mlag 
                * res2djoint.df_resid)

        # Chi-square test
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        
        # Likelihood ratio test
        lr = -2 * (res2down.llf - res2djoint.llf)
        
        # Parameter F test
        rconstr = np.column_stack((
            np.zeros((mlag, mlag)),
            np.eye(mlag, mlag),
            np.zeros((mlag, 1))
        ))
        ftres = res2djoint.f_test(rconstr)

        # Store results
        resli[mlag] = {
            'ssr_ftest': (
                fgc1,
                stats.f.sf(fgc1, mlag, res2djoint.df_resid),
                res2djoint.df_resid,
                mlag
            ),
            'ssr_chi2test': (
                fgc2,
                stats.chi2.sf(fgc2, mlag),
                mlag
            ),
            'lrtest': (
                lr,
                stats.chi2.sf(lr, mlag),
                mlag
            ),
            'params_ftest': (
                float(ftres.fvalue),
                float(ftres.pvalue),
                ftres.df_denom,
                ftres.df_num
            )
        }

    return resli