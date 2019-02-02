from code import ts_helpers as tsh
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

z = pd.read_csv('/Users/dominikballreich/PycharmProjects/max_ent/data/AirPassengers.csv')
y = np.matrix(z['#Passengers'])


class TsManipulate:

    def __init__(
            self,
            y,
            remove_trend=True,
            number_of_folds=4,
            train_test_ratio=0.8,
            base_frequency="monthly",
            create_seasonal_dummys=True,
            degree=2,
            lags=4,
            stabilize_variance=True,
            remove_season=False):
        self.y = y
        self.number_of_folds = number_of_folds
        self.train_test_ratio = train_test_ratio
        self.remove_trend = remove_trend
        self.base_frequency = base_frequency
        self.create_seasonal_dummys = create_seasonal_dummys
        self.degree = degree
        self.lags = lags
        self.stabilize_variance = stabilize_variance
        self.remove_season = remove_season

    def stationarize(self):
        if self.stabilize_variance:
            self.y = tsh.stable_variance(self.y)

        if self.remove_trend:
            self.y = tsh.remove_polynomial(self.y, self.degree)

        if self.remove_season and self.remove_trend and self.stabilize_variance:
            self.y = tsh.find_opt_sin_and_remove(self.y)

        if self.remove_season and self.remove_trend and not self.stabilize_variance:
            self.y = tsh.stable_variance(self.y)
            self.y = tsh.find_opt_sin_and_remove(self.y)

        if self.remove_season and not self.remove_trend and self.stabilize_variance:
            self.y = tsh.remove_polynomial(self.y, self.degree)
            self.y = tsh.find_opt_sin_and_remove(self.y)

        if self.remove_season and not self.remove_trend and not self.stabilize_variance:
            self.y = tsh.stable_variance(self.y)
            self.y = tsh.remove_polynomial(self.y, self.degree)
            self.y = tsh.find_opt_sin_and_remove(self.y)

    def create_features_and_measurements(self):
        self.X = tsh.create_time_series_features(self.y, self.lags)
        if self.create_seasonal_dummys:
            self.X = tsh.create_all_seasonality_dummys(self.base_frequency,self.X)

        self.y = tsh.create_measurement_series(self.y, self.lags)

        return self.X, self.y

    def fitmodel(self):
        if self.remove_trend or self.stabilize_variance or self.remove_season:
            self.stationarize()

        self.create_features_and_measurements()
        X_train, X_test, y_train, y_test = tsh.ts_test_train_split(self.X, self.y, self.train_test_ratio)
        folds_X, folds_y = tsh.create_folds_for_ts_crossval(X_train,y_train, self.number_of_folds)

        rfr = RandomForestRegressor(random_state=0)
        mse = []
        for i in range(0,self.number_of_folds-1):
            X_train_cv = np.matrix(np.concatenate(folds_X[0:i+1], axis=1))
            y_train_cv = np.matrix(np.concatenate(folds_y[0:i+1], axis=1))
            X_test_cv = folds_X[i+1]
            y_test_cv = folds_y[i+1]
            X_train_cv, y_train_cv = tsh.reshaper_for_random_forrest(X_train_cv, y_train_cv)
            X_test_cv, y_test_cv = tsh.reshaper_for_random_forrest(X_test_cv, y_test_cv)
            rfr.fit(X_train_cv, y_train_cv)
            result = rfr.predict(X_test_cv)
            mse.append(np.sqrt(np.sum(np.square(result-y_test_cv))))

        mean_performance = np.mean(mse)