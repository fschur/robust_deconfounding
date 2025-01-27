import numpy as np
from numpy.typing import NDArray
import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import scipy as sp
import pylab

from robust_deconfounding.robust_regression import Torrent, BFS
from robust_deconfounding.decor import DecoR
from robust_deconfounding.utils import cosine_basis, haarMatrix, get_funcbasis, get_funcbasis_multivariate
from synthetic_data import BLPDataGenerator, OUDataGenerator, UniformNonlinearDataGenerator, OUReflectedNonlinearDataGenerator


def plot_settings():
    """
    Sets plot configuration parameters for a consistent look across plots.

    Returns:
        tuple[list[list[str]], list[str]]: A tuple containing color palettes and a list of colors.

    Reference: https://lospec.com/palette-list/ibm-color-blind-safe
    """
    size = 12
    params = {
        'legend.fontsize': size,
        'legend.title_fontsize': size,
        'figure.figsize': (5, 5),
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size
    }
    pylab.rcParams.update(params)

    ibm_cb = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000", "#808080"]
    return [[ibm_cb[1], ibm_cb[1]], [ibm_cb[4], ibm_cb[4]], [ibm_cb[2], ibm_cb[2]]], ibm_cb


def r_squared(x: NDArray, y_true: NDArray, beta: NDArray) -> float:
    y_pred = x @ beta
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1-u/v


def get_results(x: NDArray, y: NDArray, basis: NDArray, a: float, method: str, nonlinear=False, L: int|NDArray = 6, basis_type="cosine_cont" ) -> NDArray:
    """
    Estimates the causal coefficient(s) using DecorR with 'method' as robust regression algorithm.

    Args:
        x (NDArray): The data features.
        y (NDArray): The target variable.
        basis (NDArray): The basis for transformation.
        a (float): Hyperparameter for the robust regression method.
        method (str): The method to use for DecoR ("torrent", "bfs", or "ols").

    Returns:
        NDArray: The estimated coefficients.

    Raises:
        ValueError: If an invalid method is specified.
    """

    if nonlinear:
        if method!="torrent":
            raise Exception("Use Torrent for the nonlinear extensions of DecoR, BFS is not suitable for higher-dimensional problems.")
        
        if isinstance(L, (int, np.int64)):
            x=get_funcbasis(x=x, L=L, type=basis_type)
        else:
            x=get_funcbasis_multivariate(x=x, L=L, type=basis_type)

    if method == "torrent" or method == "bfs":
        if method == "torrent":
            algo = Torrent(a=a, fit_intercept=False)
        elif method == "bfs":
            algo = BFS(a=a, fit_intercept=False)

        algon = DecoR(algo, basis)
        algon.fit(x, y)

        return algon.estimate
   
    elif method == "ols":
        model_l = sm.OLS(y, x).fit()
        return model_l.params

    else:
        raise ValueError("Invalid method")


def get_data(n: int, process_type: str, basis_type: str, fraction: float, beta: NDArray, noise_var: float,
             band: list) -> dict:
    """
    Generates data for deconfounding experiments with different settings.

    Args:
        n (int): Number of data points.
        process_type (str): Type of data generation process ("ou" or "blp").
        basis_type (str): Type of basis transformation ("cosine" or "haar").
        fraction (float): Fraction of outliers in the data.
        beta (NDArray): True coefficient vector for the linear relationship.
        noise_var (float): Variance of the noise added to the data.
        band (list): Frequency band for concentrated confounding (BLP process only).

    Returns:
        dict: A dictionary containing generated data (x, y), and the basis matrix.

    Raises:
        ValueError: If an invalid process type or basis type is specified.
    """
    if process_type == "ou":
        generator = OUDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var)
    elif process_type == "blp":
        generator = BLPDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, band=band)
    elif process_type=="uniform":
        generator =  UniformNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var)
    elif process_type=="ourre":
        generator= OUReflectedNonlinearDataGenerator(basis_type=basis_type, beta=beta, noise_var=noise_var, )
    else:
        raise ValueError("process_type not implemented")

    if basis_type == "cosine":
        basis = cosine_basis(n)
    elif basis_type == "haar":
        basis = haarMatrix(n)
    else:
        raise ValueError("basis not implemented")

    n_outliers = int(fraction*n)
    outlier_points = np.array([1]*n_outliers + [0]*(n - n_outliers)).reshape(-1, 1)
    np.random.shuffle(outlier_points)

    if beta.shape[0] == 2:
        x, y = generator.generate_data_2_dim(n=n, outlier_points=outlier_points)
    else:
        x, y = generator.generate_data(n=n, outlier_points=outlier_points)

    return {"x": x, "y": y, "basis": basis}


def plot_results(res: dict, num_data: list, m: int, colors) -> None:
    """
    Plots the estimated coefficients using DecoR and OLS methods across different data sizes.

    Args:
        res (dict): A dictionary containing estimated coefficients for DecoR and OLS.
        num_data (list): A list of data sizes used in the experiments.
        m (int): Number of repetitions for each data size.
        colors (list): A list of colors for plotting the methods.
    """
    values = np.concatenate([np.expand_dims(res["ols"], 2),
                             np.expand_dims(res["DecoR"], 2)], axis=2).ravel()

    time = np.repeat(num_data, m * 2)
    method = np.tile(["OLS", "DecoR"], len(values) // 2)

    df = pd.DataFrame({"value": values.astype(float),
                       "n": time.astype(float),
                       "method": method})

    sns.lineplot(data=df, x="n", y="value", hue="method", style="method",
                 markers=["o", "X"], dashes=False, errorbar=("ci", 95), err_style="band",
                 palette=[colors[0], colors[1]], legend=True)


def get_conf(x:NDArray, estimate:NDArray, inliers: list, transformed: NDArray, alpha=0.95, L=0, basis_type="cosine_cont") -> NDArray:
    """
        Returns a confidence interval for the estimated f evaluated at x.
        Caution: We use all points to estimate the variance (not only the inliers) to avoid a underestimation and 
                to countersteer the fact we only get an interval for \hat{f}. The returned estimation is not a valid confidence interval
        Arguements:
            x: Points where confidence interval should be evaluated
            estimate: estimated coefficients
            inliers: estimated inliers from DecoR
            alpha: level for the confidence interval
        Output:
            ci=[ci_l, ci_u]: the lower and upper bound for the confidence interval
    """

    xn=transformed["xn"]
    yn=transformed["yn"]

    if isinstance(L, (int, np.int64)):
        n=xn.shape[0] 
        basis=get_funcbasis(x=x, L=L, type=basis_type)
        L_tot=L
    else:
        basis=get_funcbasis_multivariate(x=x, L=L, type=basis_type)
        n=xn.shape[0]
        L_tot=np.sum(L)+1

    #Estimate the variance
    r=yn- xn@estimate.T
    n=xn.shape[0]
    df=n-L_tot
    sigma_2=np.sum(np.square(r), axis=0)/df 

    #Compute the linear estimator
    xn=xn[list(inliers)]
    yn=yn[list(inliers)]
    H_help=np.linalg.solve(xn.T @ xn, xn.T)
    H=basis @ H_help
    sigma=np.sqrt(sigma_2 * np.diag(H @ H.T))

    #Compute the confidence interval
    qt=sp.stats.t.ppf((1-alpha)/2, df)
    ci_u=basis@estimate.T - qt*sigma
    ci_l=basis@estimate.T + qt*sigma
    ci=np.stack((ci_l, ci_u), axis=-1)

    return ci


def conf_help(estimate:NDArray, inliers: list, transformed: NDArray, alpha=0.95, L=0)->dict:
    """
        Returns a estimation of the variance sigma, the hat matrix H and quantile q
        Caution: We use all points to estimate the variance (not only the inliers) to avoid a underestimation and 
                to countersteer the fact we only get an interval for \hat{f}.
        Arguements:
            x: Points where confidence interval should be evaluated
            estimate: estimated coefficients
            inliers: estimated inliers from DecoR
            alpha: level for the confidence interval
        Output:
            H: Hat matrix for the coefficients beta
            sigma: estimated variance
            qt: (1-alpha)/2- quantile of the student-t distributions 
    """

    xn=transformed["xn"]
    yn=transformed["yn"]

    if isinstance(L, int):
        n=xn.shape[0]
        L_tot=xn.shape[1]-1
    else:
        n=xn.shape[0]
        L_tot=np.sum(L)+1

    #Estimate the variance
    r=yn- xn@estimate.T
    n=xn.shape[0]
    df=n-L_tot
    xn=transformed["xn"][list(inliers)]

    #Compute results
    qt=sp.stats.t.ppf((1-alpha)/2, df)
    sigma=np.sqrt(np.sum(np.square(r), axis=0)/df)
    H=np.linalg.solve(xn.T @ xn, xn.T)

    return{'sigma': sigma, 'H':H , 'qt': qt}
