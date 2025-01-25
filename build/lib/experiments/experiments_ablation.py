import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils_experiments import get_results, plot_results, get_data, plot_settings


colors, ibm_cb = plot_settings()

SEED = 2
np.random.seed(SEED)
random.seed(SEED)

"""
For a short explanation of the variables and what they do see 'experiments.py'.
"""

data_args = {
    "process_type": "blp",       # "ou" | "blp
    "basis_type": "cosine",     # "cosine" | "haar"
    "noise_var": 1,
    "beta": np.array([[3.]]),
    "band": list(range(0, 50))  # list(range(0, 50)) | None
}

method_args = {
    "a": [0.7, 0.45, 0.15],
    "method": "torrent", #"torrent" | "bfs"
}

m = 1000
fractions = [0.25, 0.5, 0.8]

num_data = [4 * 2 ** k for k in range(1, 5)] + [1024]

# ----------------------------------
# run experiments
# ----------------------------------

for i in range(len(fractions)):
    print("fraction: ", fractions[i])
    res = {"DecoR": [], "ols": []}
    for n in num_data:
        print("number of data points: ", n)
        res["DecoR"].append([])
        res["ols"].append([])
        for _ in range(m):
            data_values = get_data(n, **data_args, fraction=fractions[i])

            estimates_decor = get_results(**data_values, method=method_args["method"], a=method_args["a"][i])
            res["DecoR"][-1].append(np.linalg.norm(estimates_decor - data_args["beta"].T, ord=1))

            estimates_ols = get_results(**data_values, method="ols", a=method_args["a"][i])
            res["ols"][-1].append(np.linalg.norm(estimates_ols - data_args["beta"].T, ord=1))

    res["DecoR"], res["ols"] = np.array(res["DecoR"]), np.array(res["ols"])

    plot_results(res, num_data, m, colors=colors[i])

# ----------------------------------
# plotting
# ----------------------------------

titles = {"blp": "Band-Limited", "ou": "Ornstein-Uhlenbeck"}
titles_basis = {"cosine": "", "haar": ", Haar basis"}
titles_dim = {1: "", 2: ", 2-dimensional"}
names = [0.75, 0.5, 0.2]


def get_handles():
    point_1 = Line2D([0], [0], label='OLS', marker='o',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_2 = Line2D([0], [0], label='DecoR', marker='X',
                     markeredgecolor='w', color=ibm_cb[5], linestyle='-')
    point_3 = Line2D([0], [0], label="$|G_n| = $" + str(names[0]) + "n", markersize=10,
                     color=ibm_cb[1], linestyle='-')
    point_4 = Line2D([0], [0], label="$|G_n| = $" + str(names[1]) + "n", markersize=10,
                     color=ibm_cb[4], linestyle='-')
    point_5 = Line2D([0], [0], label="$|G_n| = $" + str(names[2]) + "n", markersize=10,
                     color=ibm_cb[2], linestyle='-')

    return [point_1, point_2, point_3, point_4, point_5]


plt.xlabel("number of data points")
plt.ylabel("mean absolute error")
plt.title(titles[data_args["process_type"]]
          + titles_basis[data_args["basis_type"]]
          + titles_dim[len(data_args["beta"])])
plt.xscale('log')
plt.xlim(left=num_data[0] - 2)
plt.hlines(0, num_data[0], num_data[-1], colors='black', linestyles='dashed')

plt.legend(handles=get_handles(), loc="upper right")

plt.tight_layout()
plt.show()
