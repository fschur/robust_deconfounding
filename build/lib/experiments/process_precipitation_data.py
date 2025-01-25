import xarray as xr
import os
import statsmodels.api as sm
import numpy as np
import time

"""
Script that prepares the data for the precipitation experiments.
The data can be downloaded from https://www.climex-project.org/. We use data from the period (1950-2099) and
only the winter months (Dec, Jan, Feb).
"""

root = "..."

psl_data = xr.open_dataset(os.path.join(root, "...", "..."))
print("Import finished")

print("First time point: ",  psl_data.time.values[0], "\nLast time point: ", psl_data.time.values[-1])

# Save the time information for plotting purposes later
np.save("./data/time.npy", psl_data.indexes['time'].to_datetimeindex())

# Extract the predictor data and save it
X = psl_data.psl.values
print("Shape of the data. First dimension is time, second latitude, third is longitude: ", X.shape)
np.save("./data/X.npy", X)

# Initialize an empty array to store the smoothed values
X_smoothed = np.empty_like(X)
X_detrended = np.empty_like(X)

"""
Apply the soothing proposed by Sipple et al. (2019) to the data.
See https://journals.ametsoc.org/view/journals/clim/32/17/jcli-d-18-0882.1.xml.
"""

# Apply lowess smoothing to each grid cell
time_index = np.arange(len(psl_data.time))
#todo: parallelize
for i in range(X.shape[1]):
    for j in range(X.shape[2]):
        print(i, j)
        t = time.time()
        lowess = sm.nonparametric.lowess
        X_smoothed[:, i, j] = lowess(X[:, i, j], time_index, frac=0.3)[:, 1]
        X_detrended[:, i, j] = X[:, i, j] - X_smoothed[:, i, j]
        print(time.time() - t)

np.save("./data/X_smoothed.npy", X_smoothed)
np.save("./data/X_detrended.npy", X_detrended)

print("smoothing finished")

# Import the target data and save it
pr_train = xr.open_dataset(os.path.join(root, "...", "..."))
print("Import of 'y' finished")
print("First time point: ",  pr_train.time.values[0], "\nLast time point: ", pr_train.time.values[-1])

y = pr_train.pr.values

np.save("./data/y.npy", y)