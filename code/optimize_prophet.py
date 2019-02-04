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

# Calculate the RMSE for the Facebook Prophet forecast for May and July 1960
RMSE_May_July = np.sqrt(np.mean(np.square(forecast['yhat'][136]-y['y'][136])+np.square(forecast['yhat'][138]-y['y'][138])))
print(RMSE_May_July)

# Extract the samples from the posterior predictive distribution for May 1960
pred_samples = m.predictive_samples(future)
pred_samples_df = pd.DataFrame(pred_samples['yhat'])
prior_samples = pred_samples_df.iloc[136, :]


# Define the constraints for May 1960
rules = [['-inf', 420, 0.01], [451, 483, 0.8], [483, 'inf', 0.001]]


# Calculate the Maximum Entropy Distribution for May 1960
points, weights, prior, max_ent_dist = me.opt_max_ent(rules, prior_samples)


# Plot prior and the Maximum Entropy Distribution for May 1960
plt.plot(points, prior, label='$\widehat{p_{0}}(y)$')
plt.plot(points, max_ent_dist, label='Maximum Entropy Distribution')
plt.xlabel('Number of Passengers')
plt.ylabel('Density')
plt.legend()
plt.title('Prior and Maximum Entropy Distribution for May 1960')
plt.show()


# Calculate the Maximum Entropy forecast for May 1960
prediction = np.sum(points * max_ent_dist * weights)
forecast_new = forecast['yhat'].copy()
forecast_new.iloc[136] = prediction


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
plt.title('Prior and Maximum Entropy Distribution for July 1960')
plt.show()


# Calculate the Maximum Entropy forecast for July 1960
prediction = np.sum(points * max_ent_dist * weights)
forecast_new.iloc[138] = prediction

# Calculate the RMSE for the Maximum Entropy forecast for May and July 1960
RMSE_May_July_Max_Ent = np.sqrt(np.mean(np.square(forecast_new[136]-y['y'][136])+np.square(forecast_new[138]-y['y'][138])))
print(RMSE_May_July_Max_Ent)

# Plot the forecasts
y_p = y['y'][120:]
forcast_p = forecast['yhat'][120:]
forecast_new_p = forecast_new[120:]

labels = [59, 60]
xt = np.arange(120, 133, 12)
plt.plot(y_p, c='b', label='Original')
plt.plot(forcast_p, c='c', label='FB Prophet Forecasts')
plt.plot(forecast_new_p, c='m', label='Maximum Entropy Forecasts')
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.plot(136, forecast['yhat'][136], 'ro', fillstyle='none')
plt.plot(138, forecast['yhat'][138], 'ro', fillstyle='none')
plt.legend()
plt.show()

