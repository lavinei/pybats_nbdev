# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_shared.ipynb (unless otherwise specified).

__all__ = ['load_interpolators', 'cov2corr', 'load_sales_example', 'load_sales_example2',
           'load_dcmm_latent_factor_example', 'load_dbcm_latent_factor_example', 'load_standard_holidays']

# Internal Cell
#exporti
import numpy as np
import pandas as pd
import scipy as sc
import pickle
from scipy.special import digamma
from pandas.tseries.holiday import AbstractHolidayCalendar, USMartinLutherKingJr, USMemorialDay, Holiday, USLaborDay, \
    USThanksgivingDay
import os
import pickle
import zlib

# Cell

def load_interpolators():

    pkg_data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data'
    #pkg_data_dir = os.getcwd().split('pybats_nbdev')[0] + 'pybats_nbdev/pybats_nbdev/pkg_data'
    #pkg_data_dir = globals()['_dh'][0] + '/pkg_data'

    try:
        with open(pkg_data_dir + '/interp_beta.pickle.gzip', 'rb') as fl:
            interp_beta = pickle.loads(zlib.decompress(fl.read()))

        with open(pkg_data_dir + '/interp_gamma.pickle.gzip', 'rb') as fl:
            interp_gamma = pickle.loads(zlib.decompress(fl.read()))

    except:
        print('WARNING: Unable to load interpolator. Code will run slower.')
        interp_beta, interp_gamma = None, None

    return interp_beta, interp_gamma

# Internal Cell
# I need this helper in a module file for pickle reasons ...
def transformer(ft, qt, fn1, fn2):
    return np.exp(np.ravel(fn1(ft, np.sqrt(qt), grid=False))), \
           np.exp(np.ravel(fn2(ft, np.sqrt(qt), grid=False)))

# Internal Cell
def gamma_transformer(ft, qt, fn):
    alpha = np.ravel(np.exp(fn(np.sqrt(qt))))
    beta = np.exp(digamma(alpha) - ft)
    return alpha, beta

# Internal Cell
def trigamma(x):
    return sc.special.polygamma(x=x, n=1)

# Internal Cell
def save(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file=file)

# Internal Cell
def load(filename):
    with open(filename, "rb") as file:
        tmp = pickle.load(file)
    return tmp

# Internal Cell
def define_holiday_regressors(X, dates, holidays=None):
    """
    Add columns to the predictor matrix X for a specified list of holidays

    :param X: (Array) Predictor matrix without columns for the holidays
    :param dates: Dates
    :param holidays: (List) holidays
    :return: Updated predictor matrix
    """
    if holidays is not None:
        if len(holidays) > 0:
            if X is None:
                n = len(dates)
            else:
                n = X.shape[0]

            for holiday in holidays:
                cal = AbstractHolidayCalendar()
                cal.rules = [holiday]
                x = np.zeros(n)
                x[dates.isin(cal.holidays())] = 1
                if X is None:
                    X = x
                else:
                    X = np.c_[X, x]

            return X
        else:
            return X
    else:
        return X

# Cell
def cov2corr(cov):
    """
    Transform a covariance matrix into correlation. Useful to understand the state vector correlation (mod.C)

    :param cov: Covariance matrix
    :return: Correlation matrix
    """
    D = np.sqrt(cov.diagonal()).reshape(-1, 1)
    return cov / D / D.T

# Cell
def load_sales_example():
    """
    Read data for the first sales forecasting example

    :return: A Pandas data frame
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    return pd.read_csv(data_dir + 'sales.csv', index_col=0)[['Sales', 'Advertising']]

# Cell
def load_sales_example2():
    """
    Read data for the second sales forecasting example

    :return: A Pandas data frame
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    data = pd.read_pickle(data_dir + 'sim_sales_data')
    data = data.set_index('Date')
    return data

# Cell
def load_dcmm_latent_factor_example():
    """
    Read data for the DCMM latent factor example

    :return: A list of data
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    data = load(data_dir + 'dcmm_latent_factor_data')
    return data

# Cell
def load_dbcm_latent_factor_example():
    """
    Read data for the DCMM latent factor example

    :return: A list of data
    """
    data_dir = os.path.dirname(os.path.abspath(__file__)) + '/pkg_data/'
    data = load(data_dir + 'dbcm_latent_factor_data')
    return data

# Cell
def load_standard_holidays():
    holidays = [USMartinLutherKingJr,
                USMemorialDay,
                Holiday('July4', month=7, day=4),
                USLaborDay,
                # Holiday('Thanksgiving_1DB', month=11, day=1, offset=pd.DateOffset(weekday=WE(4))),
                USThanksgivingDay,
                # Holiday('Christmas_1DB', month=12, day=24),
                Holiday('Christmas', month=12, day=25),
                Holiday('New_Years_Eve', month=12, day=31),
                ]
    return holidays