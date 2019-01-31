import pandas as pd
import numpy as np
from scipy.stats import boxcox
from matplotlib import pyplot as plt
from pyswarm import pso

def reshaper(y):
    y = np.matrix(y)
    rows = y.shape[0]
    cols = y.shape[1]

    if rows > cols:
        y = np.transpose(y)

    return y


def stable_variance(y, plot=False):
    #y = reshaper(y)
    y_log = boxcox(np.transpose(y))
    if plot:
        plt.plot(np.tranpose(y))
        plt.show()
        plt.plot(y_log)
        plt.show()

    return y_log


def remove_polynomial(y,degree, plot=False):
    y = pd.DataFrame(y)
    y = y.squeeze()
    t = np.arange(0, y.shape[0])
    fit = np.polyfit(t, y, degree)
    p = np.poly1d(fit)
    poly = p(t)
    residual = y-poly
    residual = np.matrix(residual)

    if plot:
        plt.plot(y)
        plt.plot(p(t))
        plt.show()
        plt.plot(np.transpose(residual))
        plt.show()

    poly = np.matrix(poly)
    return residual, poly, p


def create_time_series_features(y,lags):
    y = reshaper(y)
    y = y[0,:]
    rows = y.shape[0]
    cols = y.shape[1]
    X = np.matrix([]).reshape(rows*lags,0)

    for i in range(lags-1,cols-1):
        lagged_columns = y[:,i-lags+1:i+1]
        lagged_columns = np.transpose(lagged_columns)
        lagged_columns = np.transpose(lagged_columns.flatten())
        X = np.append(X,lagged_columns,1)

    return X


def create_measurement_series(y,spacing):
    y = reshaper(y)
    y = y[:,spacing:]

    return y


def ts_test_train_split(y,size_of_train_set):
    y = reshaper(y)
    cols = y.shape[1]
    cut = int(np.ceil(cols * size_of_train_set))
    y_train = y[:,0:cut]
    y_test = y[:,cut:]

    return y_train, y_test


def create_all_seasonality_dummys(base_frequency: str, start_time=False, length=False):

    if base_frequency == "dayly":
        indicator_weekly = np.arange(7)+1
        divider_weekly = int(np.floor(cols / len(indicator_weekly)))
        if divider_weekly > 0:
            indicator_weekly = np.tile(indicator_weekly, divider_weekly + 1)
            indicator_weekly = indicator_weekly[0:cols]
            X = np.vstack([X, indicator_weekly])

        indicator_monthly = np.arange(30)+1
        divider_monthly = int(np.floor(cols / len(indicator_monthly)))
        if divider_monthly > 0:
            indicator_monthly = np.tile(indicator_monthly, divider_monthly + 1)
            indicator_monthly = indicator_monthly[0:cols]
            X = np.vstack([X, indicator_monthly])

        indicator_quarterly = np.arange(90) + 1
        divider_quarterly = int(np.floor(cols / len(indicator_quarterly)))
        if divider_quarterly > 0:
            indicator_quarterly = np.tile(indicator_quarterly, divider_quarterly + 1)
            indicator_quarterly = indicator_quarterly[0:cols]
            X = np.vstack([X, indicator_quarterly])

        indicator_yearly = np.arange(365) + 1
        divider_yearly = int(np.floor(cols / len(indicator_yearly)))
        if divider_yearly > 0:
            indicator_yearly = np.tile(indicator_yearly, divider_yearly + 1)
            indicator_yearly = indicator_yearly[0:cols]
            X = np.vstack([X, indicator_yearly])

    elif base_frequency == "weekly":

        indicator_monthly = np.arange(4)+1
        divider_monthly = int(np.floor(cols/len(indicator_monthly)))
        if divider_monthly > 0:
            indicator_monthly = np.tile(indicator_monthly , divider_monthly+1)
            indicator_monthly = indicator_monthly[0:cols]
            X = np.vstack([X,indicator_monthly])

        indicator_quarterly = np.arange(12) + 1
        divider_quarterly = int(np.floor(cols / len(indicator_quarterly)))
        if divider_quarterly > 0:
            indicator_quarterly = np.tile(indicator_quarterly, divider_quarterly + 1)
            indicator_quarterly = indicator_quarterly[0:cols]
            X = np.vstack([X, indicator_quarterly])

        indicator_yearly = np.arange(52) + 1
        divider_yearly = int(np.floor(cols / len(indicator_yearly)))
        if divider_yearly > 0:
            indicator_yearly = np.tile(indicator_yearly, divider_yearly + 1)
            indicator_yearly = indicator_yearly[0:cols]
            X = np.vstack([X, indicator_yearly])

    elif base_frequency == "monthly":

        if start_time:
            start_time = int(start_time)
            if start_time > 12:
                start_time = 1

        qrt = np.full((1, 3), 1)
        indicator_quarterly = np.hstack((qrt, qrt*2, qrt*3,qrt*4))
        indicator_quarterly = np.matrix(np.tile(indicator_quarterly, 20))

        indicator_monthly = np.arange(12)+1
        indicator_monthly = np.matrix(np.tile(indicator_monthly, 20))

        indicator_numerator = np.arange(indicator_quarterly.shape[1])+1
        indicators = np.vstack((indicator_quarterly, indicator_monthly, indicator_numerator))

    if start_time:
        indicators = indicators[:,start_time-2:]

    if length:
        indicators = indicators[:,0:length]

    return indicators

def create_all_seasonality_dummys_direct(base_frequency: str, start_time=False, length=False):

    if base_frequency == "dayly":
        indicator_weekly = np.arange(7)+1
        divider_weekly = int(np.floor(cols / len(indicator_weekly)))
        if divider_weekly > 0:
            indicator_weekly = np.tile(indicator_weekly, divider_weekly + 1)
            indicator_weekly = indicator_weekly[0:cols]
            X = np.vstack([X, indicator_weekly])

        indicator_monthly = np.arange(30)+1
        divider_monthly = int(np.floor(cols / len(indicator_monthly)))
        if divider_monthly > 0:
            indicator_monthly = np.tile(indicator_monthly, divider_monthly + 1)
            indicator_monthly = indicator_monthly[0:cols]
            X = np.vstack([X, indicator_monthly])

        indicator_quarterly = np.arange(90) + 1
        divider_quarterly = int(np.floor(cols / len(indicator_quarterly)))
        if divider_quarterly > 0:
            indicator_quarterly = np.tile(indicator_quarterly, divider_quarterly + 1)
            indicator_quarterly = indicator_quarterly[0:cols]
            X = np.vstack([X, indicator_quarterly])

        indicator_yearly = np.arange(365) + 1
        divider_yearly = int(np.floor(cols / len(indicator_yearly)))
        if divider_yearly > 0:
            indicator_yearly = np.tile(indicator_yearly, divider_yearly + 1)
            indicator_yearly = indicator_yearly[0:cols]
            X = np.vstack([X, indicator_yearly])

    elif base_frequency == "weekly":

        indicator_monthly = np.arange(4)+1
        divider_monthly = int(np.floor(cols/len(indicator_monthly)))
        if divider_monthly > 0:
            indicator_monthly = np.tile(indicator_monthly , divider_monthly+1)
            indicator_monthly = indicator_monthly[0:cols]
            X = np.vstack([X,indicator_monthly])

        indicator_quarterly = np.arange(12) + 1
        divider_quarterly = int(np.floor(cols / len(indicator_quarterly)))
        if divider_quarterly > 0:
            indicator_quarterly = np.tile(indicator_quarterly, divider_quarterly + 1)
            indicator_quarterly = indicator_quarterly[0:cols]
            X = np.vstack([X, indicator_quarterly])

        indicator_yearly = np.arange(52) + 1
        divider_yearly = int(np.floor(cols / len(indicator_yearly)))
        if divider_yearly > 0:
            indicator_yearly = np.tile(indicator_yearly, divider_yearly + 1)
            indicator_yearly = indicator_yearly[0:cols]
            X = np.vstack([X, indicator_yearly])

    elif base_frequency == "monthly":

        qrt = np.full((1, 3), 1)
        indicator_quarterly = np.hstack((qrt, qrt*2, qrt*3,qrt*4))
        indicator_quarterly = np.matrix(np.tile(indicator_quarterly, 200))

        indicator_monthly = np.arange(12)+1
        indicator_monthly = np.matrix(np.tile(indicator_monthly, 200))

        indicators = np.vstack((indicator_quarterly, indicator_monthly))

    if start_time:
        indicators = indicators[:,start_time-2:]

    if length:
        indicators = indicators[:,0:length]

    return indicators



def sin_error(params,*args):
    y = args[0]
    t = np.arange(0, y.shape[1])
    amplitude = params[0]
    freqpar = params[1]
    phase = params[2]
    absolute = params[3]
    s = amplitude * np.sin(2*np.pi*(t/freqpar)+phase) + absolute
    sqr_diff = np.sqrt(np.sum(np.square(y-s)))

    return sqr_diff


def combine_lagged_features_with_indicators(X,indicators):
    X = np.vstack((X,indicators))
    return X


def find_opt_sin_and_remove(y, opt_datails=False, plot=False):
    y = reshaper(y)
    lb = [0, 0, -10, -np.absolute(y).max()]
    ub = [np.absolute(y).max(), 150, 10, np.absolute(y).max()]
    params, fopt = pso(sin_error, lb, ub, args=y, debug=opt_datails, minfunc=1e-16, minstep=1e-16, maxiter=150, swarmsize=1000)

    t = np.arange(0, y.shape[1])
    sin = params[0] * np.sin(2 * np.pi * (t / params[1]) + params[2]) + params[3]
    residual = y-sin

    if plot:
        t = np.arange(0, y.shape[1])
        sin = params[0] * np.sin(2 * np.pi * (t / params[1]) + params[2]) + params[3]
        plt.plot(sin)
        plt.plot(np.transpose(y))
        plt.show()
        plt.plot(np.transpose(residual))
        plt.show()

    return residual


def reshaper_for_random_forrest(X,y):
    y = np.matrix(y)
    X = np.matrix(X)
    rows_X = X.shape[0]
    cols_X = X.shape[1]
    rows_y = y.shape[0]
    cols_y = y.shape[1]

    if rows_X < cols_X:
        X = np.transpose(X)

    if rows_y < cols_y:
        y = np.transpose(y)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    y = y.squeeze()

    return X, y


def create_folds_for_ts_direct_crossval(X,y,number_of_folds):
    X = reshaper(X)
    y = reshaper(y)

    folds_train_X = []
    folds_train_y = []
    folds_test_X = []
    folds_test_y = []

    for i in range(X.shape[1]-number_of_folds, X.shape[1]):
        data_X = X[:, 0:i]
        data_y = y[:, 0:i]
        folds_train_X.append(data_X)
        folds_train_y.append(data_y)

        folds_test_X.append(X[:, i])
        folds_test_y.append(y[:, i])

    return folds_train_X, folds_train_y, folds_test_X, folds_test_y
















