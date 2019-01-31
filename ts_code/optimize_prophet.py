from fbprophet import Prophet
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import maximum_entropy as me

y = pd.read_csv('/Users/dominikballreich/PycharmProjects/max_ent/data/AirPassengers.csv')
y.columns = ['ds', 'y']


labels = np.arange(49,61)
xt = np.arange(0,143,12)
plt.plot(y['y'])
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.show()


y_train = y.iloc[0:132, :]
y_test = y.iloc[132:, :]
m = Prophet(mcmc_samples=1000, seasonality_mode='multiplicative')
m.fit(y_train)
future = m.make_future_dataframe(periods=y_test.shape[0], freq='MS')
forecast = m.predict(future)
pred_samples = m.predictive_samples(future)
pred_samples_df = pd.DataFrame(pred_samples['yhat'])

prior_samples = pred_samples_df.iloc[138, :]

rules = [['-inf', 548, 0.01], [590,'inf', 0.75]]
points, weights, prior, max_ent_dist = me.opt_max_ent(rules, prior_samples)
plt.plot(points, prior, label='$\widetilde{p_{0}}(y)$')
#plt.plot(points, max_ent_dist)
plt.xlabel('Number of Passengers')
plt.ylabel('Density')
plt.legend()
plt.show()

prediction = np.sum(points * max_ent_dist * weights)
forecast_new = forecast['yhat'].copy()
forecast_new.iloc[138] = prediction

labels = np.arange(49,61)
xt = np.arange(0,143,12)
plt.plot(y['y'], c='b', label='Original')
plt.plot(forecast['yhat'], c='c', label='Forecasts')
#plt.plot(forecast_new, c='r')
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.plot(138, forecast['yhat'][138], 'ro', fillstyle='none')
plt.plot(136, forecast['yhat'][136], 'ro', fillstyle='none')
plt.legend()
plt.show()

