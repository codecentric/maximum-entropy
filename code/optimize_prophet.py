from fbprophet import Prophet
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import maximum_entropy_helpers as me

# Load data
y = pd.read_csv('/Users/dominikballreich/PycharmProjects/maximum_entropy/data/AirPassengers.csv')
y.columns = ['ds', 'y']


# Plot the timeseries and mark the test period
labels = np.arange(49,61)
xt = np.arange(0, 143, 12)
plt.plot(y['y'])
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.show()


# Generate forecasts for 1960 by using Facebook Prophet.
y_train = y.iloc[0:132, :]
y_test = y.iloc[132:, :]
m = Prophet(mcmc_samples=1000, seasonality_mode='multiplicative')
m.fit(y_train)
future = m.make_future_dataframe(periods=y_test.shape[0], freq='MS')
forecast = m.predict(future)


# Extract the samples from the posterior predictive distribution for July 1960
pred_samples = m.predictive_samples(future)
pred_samples_df = pd.DataFrame(pred_samples['yhat'])
prior_samples = pred_samples_df.iloc[138, :]


# Define the constraints for July 1960
rules = [['-inf', 548, 0.01], [590,'inf', 0.80]]


# Calculate the Maximum Entropy Distribution for July 1960
points, weights, prior, max_ent_dist = me.opt_max_ent(rules, prior_samples)


# Plot prior and the Maximum Entropy Distribution for July 1960
plt.plot(points, prior, label='$\widehat{p_{0}}(y)$')
plt.plot(points, max_ent_dist, label='Maximum Entropy Distribution')
plt.xlabel('Number of Passengers')
plt.ylabel('Density')
plt.legend()
plt.show()


# Calculate the Maximum Entropy forecast for July 1960
prediction = np.sum(points * max_ent_dist * weights)
forecast_new = forecast['yhat'].copy()
forecast_new.iloc[138] = prediction


# Extract the samples from the posterior predictive distribution for May 1960
pred_samples = m.predictive_samples(future)
pred_samples_df = pd.DataFrame(pred_samples['yhat'])
prior_samples = pred_samples_df.iloc[136, :]


# Define the constraints for May 1960
rules = [['-inf', 420, 0.01], [460, 490, 0.7]]


# Calculate the Maximum Entropy Distribution for May 1960
points, weights, prior, max_ent_dist = me.opt_max_ent(rules, prior_samples)


# Plot prior and the Maximum Entropy Distribution for May 1960
plt.plot(points, prior, label='$\widehat{p_{0}}(y)$')
plt.plot(points, max_ent_dist, label='Maximum Entropy Distribution')
plt.xlabel('Number of Passengers')
plt.ylabel('Density')
plt.legend()
plt.show()


# Calculate the Maximum Entropy forecast for May 1960
prediction = np.sum(points * max_ent_dist * weights)
forecast_new.iloc[136] = prediction


labels = np.arange(49,61)
xt = np.arange(0,143,12)
plt.plot(y['y'], c='b', label='Original')
plt.plot(forecast['yhat'], c='c', label='FB Prophet Forecasts')
plt.plot(forecast_new, c='m', label='Maximum Entropy Forecasts')
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.plot(138, forecast['yhat'][138], 'ro', fillstyle='none')
plt.plot(136, forecast['yhat'][136], 'ro', fillstyle='none')
plt.legend()
plt.show()

