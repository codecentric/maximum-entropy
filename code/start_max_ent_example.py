from fbprophet import Prophet
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import maximum_entropy_helpers as me

# Load data
y = pd.read_csv('../data/AirPassengers.csv')
y.columns = ['ds', 'y']

may_1960_index = 136
july_1960_index = 138

# Plot the timeseries and mark the test period
labels = np.arange(49,61)
xt = np.arange(0, 143, 12)
plt.plot(y['y'])
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
print("Monthly totals of international airline passengers between 1949 to 1960 in thousands from which we would like to predict the year 1960. We will use Facebook Prophet. In this example, we will concentrate on may 1960.")
input("Press Enter to continue...")
plt.show()


# Generate forecasts for 1960 by using Facebook Prophet.
y_train = y.iloc[0:132, :]
y_test = y.iloc[132:, :]
m = Prophet(mcmc_samples=1000, seasonality_mode='multiplicative')
m.fit(y_train)
future = m.make_future_dataframe(periods=y_test.shape[0], freq='MS')
forecast = m.predict(future)

# Calculate the RMSE for the Facebook Prophet forecast for may 1960
RMSE_May_FBP = np.sqrt(np.square(forecast['yhat'][may_1960_index]-y['y'][may_1960_index]))
print("The Facebbook Prophet RMSE for may 1960 (blue circle) is:", RMSE_May_FBP)

# Extracting the point forecasts of Facebook Prophet
forecast_new = forecast['yhat'].copy()

# Plot the forecasts
y_p = y['y'][120:]
forcast_p = forecast['yhat'][120:]
forecast_new_p = forecast_new[120:]

labels = [59, 60]
xt = np.arange(120, 133, 12)
plt.plot(y_p, c='b', label='Original')
plt.plot(forcast_p, c='c', label='FB Prophet Forecasts')
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.plot(may_1960_index, forecast['yhat'][may_1960_index], 'ro', fillstyle='none', color='c')
plt.annotate(s='', xy=(may_1960_index, forecast_new.iloc[may_1960_index]), xytext=(may_1960_index, forecast['yhat'][may_1960_index]), arrowprops=dict(arrowstyle='simple', color='g'))
plt.legend()
plt.show()

# Extract the samples from the posterior predictive distribution for may 1960
pred_samples = m.predictive_samples(future)
pred_samples_df = pd.DataFrame(pred_samples['yhat'])
prior_samples = pred_samples_df.iloc[may_1960_index, :]

print("We introduce th optinion of an expert:  \n This May we had 420.000 Passengers and we will definitely not have fewer in May 1960 (['-inf', 420] has probability 1%).\n Furthermore, given the numbers of the last three years, I am sure that a growth rate compared to this May between 7.5% and 15% is extremely probable ([451, 483] has probability 80%).  \n However, an increase of 15% or more compared to this May, in my opinion, is unrealistic ([483, 'inf'] has probability 1%).")
input("Press Enter to continue...")

# Define the constraints for may 1960
rules = [['-inf', 420, 0.01], [451, 483, 0.80], [483, 'inf', 0.01]]

# Calculate the Maximum Entropy Distribution for may 1960
points, weights, prior, max_ent_dist = me.opt_max_ent(rules, prior_samples)


# Plot prior and the Maximum Entropy Distribution for may 1960
plt.plot(points, prior, label='$\widehat{p_{0}}(z)$')
plt.plot(points, max_ent_dist, label='Maximum Entropy Distribution')
plt.xlabel('Number of Passengers')
plt.ylabel('Density')
plt.legend()
plt.title('Prior and Maximum Entropy distribution for may 1960')
print("Close plot")
plt.show()

print("Here we see the impact of the experts opinon on the basic posterior predictive distribution")
input("Press Enter to continue...")

# Calculate the Maximum Entropy forecast for may 1960
prediction = np.sum(points * max_ent_dist * weights)
forecast_new.iloc[may_1960_index] = prediction

# Calculate the RMSE for the Maximum Entropy forecast for may 1960
RMSE_May_Max_Ent = np.sqrt(np.square(forecast_new[may_1960_index]-y['y'][may_1960_index]))
print("The Maximium Entropy RMSE for may 1960 is:", RMSE_May_Max_Ent)

# Plot the forecasts
y_p = y['y'][120:]
forcast_p = forecast['yhat'][120:]
forecast_new_p = forecast_new[120:]

labels = [59, 60]
xt = np.arange(120, 133, 12)
plt.plot(y_p, c='b', label='Original')
plt.plot(forcast_p, c='c', label='FB Prophet Forecasts')
locs, labs = plt.xticks()
plt.xticks(xt, labels)
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.axvline(x=131, c='k', linestyle='dashed', linewidth=0.5)
plt.plot(may_1960_index, forecast['yhat'][may_1960_index], 'ro', fillstyle='none', color='c')
plt.plot(may_1960_index, forecast_new.iloc[may_1960_index], 'x', fillstyle='none', color='r', label='Maximum Entropy Forecasts')
plt.annotate(s='', xy=(may_1960_index, forecast_new.iloc[may_1960_index]), xytext=(may_1960_index, forecast['yhat'][may_1960_index]), arrowprops=dict(arrowstyle='simple', color='g'))
plt.legend()
plt.show()

