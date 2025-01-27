import pickle, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils_experiments import  plot_results,  plot_settings

"""
We plot the results, i.e. L^1-error, obtained from experiments_nonlinear.py
The experiment can be selected by the variable "exp".
"""

exp="reflected_ou"     # "uniform" | "reflected_ou"

# ----------------------------------
# Load the plot setting and experiment configurations
# ----------------------------------

colors, ibm_cb = plot_settings()

path= os.path.dirname(__file__)                    #Path for the json file where experiment configurations are defined.
path_results=os.path.join(path, "results/")        #Path to the results

#Read in the parameters from the config.json file
with open(path+'/config.json', 'r') as file:
    config = json.load(file)

config=config["experiment_"+str(exp)]
m, noise_vars, num_data=  config["m"], np.array(config["noise_vars"]), np.array(config["num_data"])        

# ----------------------------------
# Load data and plotting
# ----------------------------------

plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')
for i in range(len(noise_vars)):
    with open(path_results+"experiment_" + exp +'_noise_='+str(noise_vars[i])+'.pkl', 'rb') as fp:
        res = pickle.load(fp)
    plot_results(res, num_data, m, colors=colors[i])


# ----------------------------------
# Set labels, legend and title
# ----------------------------------

titles = {"uniform": "Uniform", "reflected_ou": "Reflected OU"}

def get_handles():
    point_1 = Line2D([0], [0], label='GAM', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[0]), markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[1]), markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$\sigma_{\eta}^2 = $" + str(noise_vars[2]), markersize=10,
                     color=ibm_cb[2], linestyle='-')
    return [point_1, point_2, point_3, point_4, point_5]

plt.ylabel("$L^1$-error")
plt.xlabel("number of data points")
plt.title("Nonlinear (" + titles[exp] + ")") 
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.legend(handles=get_handles(), loc="upper right")
plt.tight_layout()

plt.show()