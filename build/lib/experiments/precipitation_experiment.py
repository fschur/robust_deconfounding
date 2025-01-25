from robust_deconfounding import DecoR
from robust_deconfounding.robust_regression import Torrent
from robust_deconfounding.utils import cosine_basis
from utils_experiments import r_squared, plot_settings

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns


# Control the number of threads used by numpy and underlying libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# load pre-processed datasets
X = np.load("./data/X.npy", allow_pickle=True)
X = X.reshape(len(X), -1)
X = X - np.mean(X, axis=0)

X_detrended = np.load("./data/X_detrended.npy", allow_pickle=True)
X_detrended = X_detrended.reshape(len(X_detrended), -1)
X_detrended = X_detrended - np.mean(X_detrended, axis=0)

y = np.load("./data/y.npy", allow_pickle=True)
yn = np.mean(y, axis=(1, 2))
yn = yn - np.mean(yn)
n = yn.shape[0]

# print informations
print("X shape: ", X.shape, "X_detrended shape: ", X_detrended.shape, "yn shape: ", yn.shape)


"""
Method: "Uncovering the Forced Climate Response from a Single Ensemble Member Using Statistical Learning"
Authors: Sebastian Sippel, Nicolai Meinshausen, Anna Merrifield, Flavio Lehner, Angeline G. Pendergrass, 
Erich Fischer and Reto Knutti
url: https://journals.ametsoc.org/view/journals/clim/32/17/jcli-d-18-0882.1.xml
"""
model = Ridge(alpha=1.0, fit_intercept=False)
model.fit(X_detrended, yn)
ridge_coef = model.coef_

print("R squared Sipple (2019): ", r_squared(X, yn, ridge_coef))


"""
Our Method
"""
a = 0.9
basis = cosine_basis(n)

algo = Torrent(a=a, fit_intercept=False)
algon = DecoR(algo, basis)
algon.fit(X, yn)

decor_coefs = algon.estimate
idx = list(algon.algo.inliers_)
removed_freqs = list(set(range(n)) - set(idx))
print("R decor", r_squared(X, yn, decor_coefs))

"""
Plot predictions vs true values.
"""
colors, ibm_c = plot_settings()

plt.figure(figsize=(5,5))
fn = sns.histplot(removed_freqs, color=ibm_c[0])
fn.set(xlabel="Frequency")
fn.set(ylabel="Count")
plt.savefig("./figures/climate_freqs_m.png")
plt.clf()

"""
Compares predicted values with true values
"""

t = np.load("./data/time.npy", allow_pickle=True)

values = np.concatenate([np.expand_dims(yn, 1),
                         np.expand_dims(X@ridge_coef - np.mean(X@ridge_coef), 1),
                         np.expand_dims(X@decor_coefs - np.mean(X@decor_coefs), 1),
                         ], axis=1).ravel()

time = pd.Series(np.repeat(t, 3)).dt.date

method = np.tile(["ground truth", "Sippel et al. (2019)", "DecoR"], len(values) // 3)

df = pd.DataFrame({"value": values.astype(float),
                   "n": time,
                   "method": method})

# average over ~ 5 years
df['n'] = pd.to_datetime(df['n'])
df.set_index('n', inplace=True)
df = df.groupby('method').resample('4500D').mean().reset_index()

plt.figure(figsize=(5, 5))
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
fn = sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
             markers=True, dashes=False, palette=[colors[0][0], colors[1][0], colors[2][0]], legend=True)
fn.set(xlabel=None)
fn.set(ylabel="mean precipitation [$kg/m^2s$]")
fn.get_figure().autofmt_xdate()
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig("./figures/y_n.png")
plt.clf()
