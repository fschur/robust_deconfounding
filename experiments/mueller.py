import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from robust_deconfounding import DecoR
from robust_deconfounding.robust_regression import Torrent, BFS
from robust_deconfounding.utils import cosine_basis, haarMatrix

from utils_experiments import plot_settings


"""
Provides the Figures in Appendix F.3 Additional Real-World Experiment.
"""

# --- Config ---
colors, ibm_cb = plot_settings()

NPZ_PATH = "data/data_only.npz"
NPZ_KEY = "yc_example_data"      # change if your array has a different name
basis_type = "cosine"            # "cosine" or "haar"
a = 0.95                       # upper bound on inlier fraction (DecoR hyperparameter)
method = "torrent"               # "torrent" or "bfs"

# --- Load saved data (first column = x, second column = y) ---
loaded = np.load(NPZ_PATH)
print("Variables saved:", loaded.files)
arr = loaded[NPZ_KEY]
assert arr.ndim == 2 and arr.shape[1] >= 2, "Expected an array with shape (n,2+) where [:,0]=x and [:,1]=y."

x = arr[:, 0].astype(float)
y = arr[:, 1].astype(float)

# Add intercept column
X_ = np.c_[np.ones_like(x), x]
y_vec = y.ravel()
n = X_.shape[0]

if basis_type == "cosine":
    basis = cosine_basis(n)
elif basis_type == "haar":
    basis = haarMatrix(n)
else:
    raise ValueError("basis_type must be 'cosine' or 'haar'.")

if method == "torrent":
    algo = Torrent(a=a, fit_intercept=True)
elif method == "bfs":
    algo = BFS(a=a, fit_intercept=True)
else:
    raise ValueError("method must be 'torrent' or 'bfs'.")

decor = DecoR(algo, basis)
decor.fit(X_, y_vec)
beta_decor = np.atleast_1d(decor.estimate).ravel()

# --- OLS baseline (with intercept) ---
beta_ols = np.linalg.lstsq(X_, y_vec, rcond=None)[0].ravel()

print(beta_decor)
print(beta_ols)

print("\n=== Estimates (with intercept) ===")
print(f"DecoR (method={method}, basis={basis_type}, a={a}): Intercept={beta_decor[0]:.6f}, Slope={beta_decor[1]:.6f}")
print(f"OLS: Intercept={beta_ols[0]:.6f}, Slope={beta_ols[1]:.6f}")

# --- Predictions for visualization ---
yhat_decor = X_ @ beta_decor[:2]
yhat_ols = X_ @ beta_ols[:2]

order = np.argsort(x)
# plt.figure(figsize=(4, 3))
plt.scatter(x, y, s=12, alpha=0.6, label="data points", color=ibm_cb[0])
plt.plot(x[order], yhat_ols[order], linewidth=2, linestyle="--", label="OLS", color=ibm_cb[3])
plt.plot(x[order], yhat_decor[order], linewidth=2, label="DecoR", color=ibm_cb[1])
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("mueller.pdf")
plt.close()
plt.close()

# --- Analyze which points were removed as outliers ---
inliers = list(decor.algo.inliers_) if hasattr(decor.algo, "inliers_") else []
removed_freqs = sorted(set(range(n)) - set(inliers))

print(f"\n# data points: {n}")
print(f"# inliers kept by DecoR: {len(inliers)}")
print(f"# removed (treated as outlier/contaminated) indices: {len(removed_freqs)}")

# plt.figure(figsize=(4, 3))
sns.histplot(removed_freqs,
             bins=min(30, max(5, len(set(removed_freqs)))) if removed_freqs else 5,
             color=ibm_cb[0],)
plt.xlabel("Frequency (index)")
plt.ylabel("Count")
plt.title("Removed Frequencies (DecoR)")
plt.tight_layout()
plt.savefig("mueller_removed_freqs.pdf")
