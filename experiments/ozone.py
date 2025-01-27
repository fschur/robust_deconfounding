import os
from pygam import GAM, s, intercept
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from robust_deconfounding.utils import cosine_basis, get_funcbasis, get_funcbasis_multivariate
from utils_experiments import get_results, plot_settings, conf_help

"""
    We apply the nonlinear extension of DecoR to the ozon dataset.
"""

path_data=os.path.join(os.path.dirname(__file__), "data/")        #Path of data
colors, ibm_cb = plot_settings()

# ----------------------------------
# Read the data
# ----------------------------------

df = pd.read_stata(path_data+"ozone.dta")
n=df.shape[0]
x=np.array(df.loc[ : , "ozone"])
y=np.array(df.loc[: , "numdeaths"])
u=np.array(df.loc[:, "temperature"])
date=np.array(df.loc[:, "date"])

# ----------------------------------
# Plot the two time series against the time
# ----------------------------------

fig, axs = plt.subplots(2, 1)

#Plotting
axs[0].plot(date, x,'o', marker='.', color="black", markersize=3)
axs[1].plot(date, y, 'o', marker='.', color="black", markersize=3)

#Set labels, adjust grid and title
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Ozone ($\mu g/m^3$)')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('# Deaths')
axs[0].set_title("Ozone levels over time")
axs[1].set_title("Daily deaths over time")
axs[0].grid(linestyle='dotted')
axs[1].grid(linestyle='dotted')

plt.tight_layout()
plt.show()


# ----------------------------------
# Normalize the data
# ----------------------------------

#Adjust for delay from x to y in days
delay=1     #Delay between ozon exposure and outcome in days
x=x[0:(n-delay)]
y=y[delay:n]

#Normalize the ozone covariate
x_min=np.min(x)
x_max=np.max(x)
x_norm=(x-x_min)/(x_max-x_min)

#Normalize the temperature covariate
temp=u[delay:n]
t_min=np.min(temp)
t_max=np.max(temp)
temp_norm=(temp-t_min)/(t_max-t_min)

#Compute the design matrix for the spline
X=np.stack((x_norm, temp_norm))

# ----------------------------------
# Deconfounding and Estimation of Causal Relationship
# ----------------------------------

method_args = {
    "a": 0.95,
    "method": "torrent",            
    "basis_type": "cosine_cont",    # basis used for the approximation of f
}

L=np.array([6, 6])    #Number of coefficients, [ozone, temperature]
n_x=200     #Resolution of x-axis

#Compute matrices to obtain estimations of y
test_points=np.linspace(0, 1, num=n_x)
test_points_adjst=np.stack((test_points, np.repeat(0, n_x)))
test_points_adjst_temp=np.stack((np.repeat(0, n_x), test_points))
basis=get_funcbasis(x=test_points, L=L, type=method_args["basis_type"])
basis_adjst=get_funcbasis_multivariate(x=test_points_adjst, L=L, type=method_args["basis_type"])
basis_temp=get_funcbasis_multivariate(x=test_points_adjst_temp, L=L, type=method_args["basis_type"])

#Fit DecoR with adjustement for the temperature
estimates_decor=get_results(x=X, y=y, **method_args, basis=cosine_basis(n-1), L=L)
y_adjst=basis_adjst @ estimates_decor["estimate"]
ci_adjst_help=conf_help(**estimates_decor, L=L, alpha=0.95)
H=basis_adjst[:,1:(L[0]+1)]@(ci_adjst_help['H'])[1:(L[0]+1), :]
sigma=ci_adjst_help['sigma']*np.sqrt(np.diag(H@H.T))
ci_adjst=np.stack((y_adjst-ci_adjst_help['qt']*sigma, y_adjst+ci_adjst_help['qt']*sigma)).T

#Fit benchmark for comparison and to make the confounding visible    
gam = GAM(s(0)+intercept, lam=5).fit(np.reshape(X.T, (-1,2)), y) #, lam=Lmbd) 
y_bench=gam.predict(test_points_adjst.T)
ci_bench=gam.confidence_intervals(test_points_adjst.T, width=0.95)


# ----------------------------------
# Plotting
# ----------------------------------

test_ozone=(test_points)*(x_max-x_min)+x_min

#Compute estimate from Bhaskaran et al. 2013
y_ref=np.exp(0.0007454149*test_ozone)*np.mean(y)
y_ref_l=np.exp(0.00042087681*test_ozone)*np.mean(y)
y_ref_u=np.exp(0.0010698931*test_ozone)*np.mean(y)

#Plot the difference estimations
plt.scatter(x=x, y=y, color='w', edgecolors="gray", s=4) 
plt.plot(test_ozone, y_bench, '-', color=ibm_cb[4], linewidth=1.5)
plt.plot(test_ozone, y_adjst, '-', color=ibm_cb[1], linewidth=1.5)
plt.plot(test_ozone, y_ref, '--', color=ibm_cb[2], linewidth=1)

#Plot confidence intervals
plt.fill_between(test_ozone, y1=ci_bench[:, 0], y2=ci_bench[:, 1], color=ibm_cb[4], alpha=0.4)
plt.fill_between(test_ozone, y1=ci_adjst[:, 0], y2=ci_adjst[:, 1], color=ibm_cb[1], alpha=0.4)
plt.fill_between(test_ozone, y1=y_ref_l, y2=y_ref_u, color=ibm_cb[2], alpha=0.4)

def get_handles():
    point_1 = Line2D([0], [0], label='Observations', marker='o', mec="gray", markersize=3, linestyle='')
    point_2= Line2D([0], [0], label='Bhaskaran et al.', color=ibm_cb[2], marker='', mec="black", markersize=3, linestyle='--')
    point_3 = Line2D([0], [0], label="DecoR" , color=ibm_cb[1], linestyle='-')
    point_4= Line2D([0], [0], label="GAM" , color=ibm_cb[4], linestyle='-')
    return [point_1,  point_2, point_3, point_4]

#Labeling
plt.xlabel("Ozone ($\mu g/m^3$)")
plt.ylabel("# Deaths")
plt.title("Influence of Ozone on Health")
plt.legend(handles=get_handles(), loc="upper left")
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()

# ----------------------------------
# Plot the estimated outliers in a Histogramm
# ----------------------------------

inl=estimates_decor["inliers"]
out=np.delete(np.arange(0,n), list(inl))
freq_rem=(out+0.5)/(2*n*24*3600)*10**6        #Conver to mikrohertz
plt.hist(freq_rem,  color=ibm_cb[0], edgecolor='k', alpha=0.6, bins=15)
plt.xlabel("Frequency ($\mu Hz$)")
plt.ylabel("Count")
plt.title("Histogramm of Excluded Frequencies")
plt.tight_layout()
plt.show()

# ----------------------------------
# Plot the influence of temperature on #death
# ----------------------------------

y_temp=basis_temp @ estimates_decor["estimate"]
test_temp=(test_points)*(t_max-t_min)+t_min

#compute the confidence interval
ci_adjst=np.stack((y_adjst-ci_adjst_help['qt']*sigma, y_adjst+ci_adjst_help['qt']*sigma)).T
H=basis_temp[:,(L[0]+1):(L[1]+L[0]+1)]@(ci_adjst_help['H'])[(L[0]+1):(L[0]+L[1]+1), :]
sigma=ci_adjst_help['sigma']*np.sqrt(np.diag(H@H.T))
ci_temp=np.stack((y_temp-ci_adjst_help['qt']*sigma, y_temp+ci_adjst_help['qt']*sigma)).T

plt.scatter(x=temp, y=y, marker='o', color='w', edgecolors="gray", s=5) 
plt.plot(test_temp, y_temp, '-', color=ibm_cb[2], linewidth=1.5)
plt.fill_between(test_temp, y1=ci_temp[:, 0], y2=ci_temp[:, 1], color=ibm_cb[2], alpha=0.3)

plt.grid(linestyle='dotted')
plt.ylabel("# Deaths")
plt.xlabel("temperature ($C^\circ$)")
plt.title("Influence of Temperature")
plt.tight_layout()
plt.show()