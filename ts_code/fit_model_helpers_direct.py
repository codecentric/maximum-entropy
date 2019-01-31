from ts_code import ts_helpers as tsh
import pandas as pd
import numpy as np
from pyswarm import pso
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import HoltWintersResults

z = pd.read_csv('/Users/dominikballreich/PycharmProjects/max_ent/data/monthly-sales-of-us-houses-thous.csv')
y = np.matrix(z.iloc[0:-1,1])
y = np.matrix(np.asarray(y))

#y, poly, _ = tsh.remove_polynomial(y, 3)
#y = (y, poly)
#y = np.matrix(np.asarray(y[0]))+200

#period=36
#start = y[:,0:-period+1]
#end = y[:,period-1:]
#proz = (end-start)
print('start')


def h_step_forecast(trained_model, start_values, start_time, base_frequency, h, lags):
    indicators = tsh.create_all_seasonality_dummys(base_frequency, start_time)
    start_values = np.matrix(start_values)
    series = start_values[:, 0:lags]
    forecasts = []
    for i in range(0, h):
        predictors = np.hstack((series, np.transpose(indicators[:, i])))
        result = trained_model.predict(predictors)
        forecasts.append(result)
        result = result[:, np.newaxis]
        series = np.hstack((series, result))
        series = series[:, 1:]

    forecasts = np.transpose(np.matrix(forecasts))

    return forecasts


def fit_model_with_parameters(params, *args):
    # First index is the starting_time of the origin time series. E.g., January is 1 for monthly series
    y = arguments[0]
    base_frequency = arguments[1]
    first_index = arguments[2]
    h = arguments[3]
    cv = arguments[5]

    n_estimators = int(np.floor(params[0]))
    max_depth = int(np.ceil(params[1]))
    # perc_min_samples_split = np.ceil(params[3])
    # perc_min_samples_leaf = params[3]
    # min_weight_fraction_leaf = params[4]
    lags = int(np.floor(params[2]))
    degree = int(np.round(params[3]))
    # max_leaf_nodes = int(np.ceil(params[4]))
    stable_var = int(np.round(params[4]))
    trend_det = int(np.round(params[5]))

    if stable_var == 1:
        y = tsh.stable_variance(y)
        bc_param = y[1]
        y = tsh.reshaper(y[0])

    if trend_det == 1:
        y, poly, _ = tsh.remove_polynomial(y, degree)
        y = (y, poly)
        y = np.matrix(np.asarray(y))

    elif trend_det == 2:
        y_origin = y
        y = np.diff(y)

    X = tsh.create_time_series_features(y, lags)

    if trend_det == 2:
        start_time = first_index + h + lags + 1
        counter = np.arange(y.shape[1])
        counter = np.matrix(counter)
        y = (y, counter)
        y = np.matrix(np.asarray(y))
    else:
        start_time = first_index + h + lags

    indicators = tsh.create_all_seasonality_dummys_direct(base_frequency, start_time, length=X.shape[1])
    X = tsh.combine_lagged_features_with_indicators(X, indicators)

    y = tsh.create_measurement_series(y, lags + h - 1)

    len_y = y.shape[1]
    len_X = X.shape[1]
    min_len = min(len_y, len_X)
    y = y[:, 0:min_len]
    X = X[:, 0:min_len]

    folds_train_X, folds_train_y, folds_test_X, folds_test_y = tsh.create_folds_for_ts_direct_crossval(X, y, cv)

    rfr = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth)

    smape = []
    for i in range(0, len(folds_train_X)):

        X_train = pd.DataFrame(np.transpose(folds_train_X[i]))
        y_train = pd.DataFrame(np.transpose(folds_train_y[i][0, :]))

        X_test = pd.DataFrame(np.transpose(folds_test_X[i]))
        y_test = pd.DataFrame(np.transpose(folds_test_y[i]))

        X_train.iloc[:, lags:] = X_train.iloc[:, lags:].astype('category')
        X_test.iloc[:, lags:] = X_test.iloc[:, lags:].astype('category')
        rfr.fit(X_train, y_train.squeeze())
        forecast = rfr.predict(X_test)

        if trend_det == 1:
            polynomial = y_test[1]
            y_test = y_test[0] + polynomial
            forecast = forecast + polynomial

        if trend_det == 2:
            forecast = y_origin[0, int(y_test[1])] + forecast
            y_test = y_origin[0, int(y_test[1] + 1)]

        if stable_var == 1:
            if bc_param == 0:
                y_test = np.exp(y_test)
                forecast = np.exp(forecast)
            else:
                y_test = (np.exp(np.log(bc_param * y_test + 1) / bc_param))
                forecast = (np.exp(np.log(bc_param * forecast + 1) / bc_param))

        y_test = np.float32(y_test)
        forecast = np.float32(forecast)

        metric = np.abs(y_test/forecast-1)
        smape.append(metric)

    weights = np.arange(1, len(folds_train_X)+1)
    weights = weights/np.sum(weights)
    mean_performance = np.mean(smape)

    return mean_performance


def get_model_for_h_steps(params, arguments):
    # First index is the starting_time of the origin time series. E.g., January is 1 for monthly series
    y_origin = arguments[0]
    base_frequency = arguments[1]
    first_index = arguments[2]
    h = arguments[3]

    n_estimators = int(np.floor(params[0]))
    max_depth = int(np.ceil(params[1]))
    # perc_min_samples_split = np.ceil(params[3])
    # perc_min_samples_leaf = params[3]
    # min_weight_fraction_leaf = params[4]
    lags = int(np.floor(params[2]))
    degree = int(np.round(params[3]))
    # max_leaf_nodes = int(np.ceil(params[4]))
    stable_var = int(np.round(params[4]))
    trend_det = int(np.round(params[5]))

    y = y_origin

    if stable_var == 1:
        y = tsh.stable_variance(y)
        y = tsh.reshaper(y[0])

    if trend_det == 1:
        y, poly, _ = tsh.remove_polynomial(y, degree)
        y = (y, poly)
        y = np.matrix(np.asarray(y))

    elif trend_det == 2:
        y = np.diff(y)

    X = tsh.create_time_series_features(y, lags)

    if trend_det == 2:
        start_time = first_index + h + lags + 1
    else:
        start_time = first_index + h + lags

    indicators = tsh.create_all_seasonality_dummys_direct(base_frequency, start_time, length=X.shape[1])
    X = tsh.combine_lagged_features_with_indicators(X, indicators)

    y = tsh.create_measurement_series(y, lags + h - 1)

    len_y = y.shape[1]
    len_X = X.shape[1]
    min_len = min(len_y, len_X)
    y = y[:, 0:min_len]
    X = X[:, 0:min_len]

    rfr = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth)

    X_train = pd.DataFrame(np.transpose(X))
    y_train = pd.DataFrame(np.transpose(y[0, :]))
    X_train.iloc[:, lags:] = X_train.iloc[:, lags:].astype('category')
    rfr.fit(X_train, y_train.squeeze())
    return rfr


def get_h_step_prediction(model, y_test, base_value, params, arguments):
    y_train = arguments[0]
    base_frequency = arguments[1]
    first_index = arguments[2]
    h = arguments[3]
    train_ratio = arguments[4]
    start_time = first_index + 1

    lags = int(np.floor(params[2]))
    degree = int(np.round(params[3]))
    stable_var = int(np.round(params[4]))
    trend_det = int(np.round(params[5]))

    indicators = tsh.create_all_seasonality_dummys_direct(base_frequency, start_time,
                                                          length=(y_train.shape[1] + y_test.shape[1]))
    ind_train, ind_test = tsh.ts_test_train_split(indicators, train_ratio)

    if stable_var == 1:
        y_train = tsh.stable_variance(y_train)
        bc_param = y_train[1]
        y_train = tsh.reshaper(y_train[0])

    if trend_det == 1:
        y_train, poly, p = tsh.remove_polynomial(y_train, degree)

    elif trend_det == 2:
        y_train = np.diff(y_train)

    X = np.hstack((y_train[0, -lags:], np.transpose(ind_test[:, h - 1])))
    forecast = model.predict(X)

    if trend_det == 1:
        forecast = forecast + p(y_train.shape[1] + h - 1)

    if trend_det == 2:
        if stable_var == 1:
            forecast = boxcox(base_value, bc_param) + forecast
        else:
            forecast = base_value + forecast

    if stable_var == 1:
        if bc_param == 0:
            forecast = np.exp(forecast)
        else:
            forecast = (np.exp(np.log(bc_param * forecast + 1) / bc_param))

    return forecast

model = []
preds = []
# First index is the starting_time of the origin time series. E.g., January is 1 for monthly series
first_index = 1
steps_ahead = 36
train_ratio = 1 - steps_ahead / y.shape[1]
cv = 10
y_train, y_test = tsh.ts_test_train_split(y, train_ratio)
y_real = []
for i in range(1, steps_ahead + 1):
    if i == 1:
        base_value = y_train[0, -1]

    print(i)
    lb = [10, 1,   1,  0, 0, 0]
    ub = [50, 5,   20, 2, 2, 1]
    arguments = (y_train, 'monthly', first_index, i, train_ratio, cv)
    params, fopt = pso(fit_model_with_parameters, lb, ub, args=arguments, debug=True, phip=0.5, phig=0.5, omega=0.5,
                       minfunc=1e-8, minstep=1e-8, maxiter=7, swarmsize=50)
    rfr = get_model_for_h_steps(params, arguments)
    model.append(rfr)
    forecast = get_h_step_prediction(rfr, y_test, base_value, params, arguments)
    base_value = forecast
    preds.append(forecast)
    test = np.absolute((y_test[0, i - 1] / forecast) - 1)
    y_real.append(y_test[0, i - 1])
    print(test)

RMSE = np.sqrt(np.mean(np.square(np.array(y_real)-np.array(preds).squeeze())))
print("RMSE equals" " " + str(RMSE))
s = np.arange(y_train.shape[1], y.shape[1])
preds = np.matrix(preds)
plt.plot(s, preds, label="preds")
plt.plot(np.transpose(y), label="y")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

