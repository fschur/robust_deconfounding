from typing import Set, Optional, Self
from numpy.typing import NDArray
import numpy as np
import statsmodels.api as sm
import itertools


class BaseRobustRegression:
    """Base class for robust regression models.

    Attributes:
        fit_intercept (bool): Whether to include an intercept in the model.
        model: Holds the regression model after fitting.
        inliers (list): Indices of inliers used in the final model.
    """

    def __init__(self, fit_intercept: bool = False):
        self.fit_intercept = fit_intercept
        self.model = None
        self.inliers = []

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fits the regression model to the data.

        Must be implemented by subclasses.

        Args:
            x (NDArray): Design matrix.
            y (NDArray): Response vector.

        Returns:
            Self: The instance itself.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    @staticmethod
    def _validate_inputs(x, y):
        """Validates the input data and basis."""
        if len(y) != len(x) or len(y) == 0:
            raise ValueError("Data and basis must have the same length and be non-empty.")

    @staticmethod
    def _add_intercept(x):
        """Adds an intercept column to the design matrix."""
        n = len(x)
        return np.hstack((np.ones((n, 1)), x))

    @property
    def coef_(self) -> NDArray:
        """Returns the coefficients of the regression model, excluding the intercept."""
        return self.model.params

    @property
    def intercept_(self) -> Optional[float]:
        """Returns the intercept of the regression model, if applicable."""
        return self.model.params[0] if self.fit_intercept else None

    @property
    def inliers_(self) -> Set[int]:
        """Returns a set of indices of inliers used in the final model."""
        return set(self.inliers)


class Torrent(BaseRobustRegression):
    """Torrent algorithm for regression with robustness to outliers.

    Extends the base regression to implement an iterative process of fitting and refining inliers.

    Attributes:
        a (float): Proportion of data considered as inliers.
        max_iter (int): Maximum number of iterations.
        predicted_inliers (list): List to track inliers over iterations.

    Reference:
    Robust Regression via Hard Thresholding, Kush Bhatia, Prateek Jain, Purushottam Kar,
    https://arxiv.org/abs/1506.02428
    """

    def __init__(self, a: float, fit_intercept: bool = True, max_iter: int = 100):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("'a' must be in the range (0, 1).")
        self.a = a
        self.max_iter = max_iter
        self.predicted_inliers = []

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model using an iterative process to determine inliers and refit the model."""
        n = len(y)
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points .")

        self.inliers = list(range(n))
        self.predicted_inliers.append(self.inliers)

        for _ in range(self.max_iter):
            self.model = sm.OLS(y[self.inliers], x[self.inliers]).fit()

            err = np.linalg.norm(y - self.model.predict(x).reshape(n, -1), axis=1)

            old_inliers = self.inliers
            self.inliers = np.argpartition(err, an)[:an]
            self.predicted_inliers.append(self.inliers)

            if set(self.inliers) == set(old_inliers):
                break
        return self


class BFS(BaseRobustRegression):
    """Brute Force Search (BFS) algorithm for regression to find the best subset of inliers.

    Attributes:
        a (float): Proportion of data to be considered for each potential subset of inliers.
    """

    def __init__(self, a: float, fit_intercept: bool = True):
        super().__init__(fit_intercept)
        if not 0 < a < 1:
            raise ValueError("a must be in the range (0, 1).")

        self.a = a

    def fit(self, x: NDArray, y: NDArray) -> Self:
        """Fit model by exhaustively searching over possible combinations of inliers."""
        n = y.shape[0]
        y = y.reshape(n, -1)

        self._validate_inputs(x, y)
        if self.fit_intercept:
            x = self._add_intercept(x)

        an = int(self.a * n)
        if an == 0:
            raise ValueError("'a' is too small. Increase 'a' or the number of data points.")

        permu = itertools.combinations(range(n), an)
        err_min = np.inf

        for p in permu:
            model = sm.OLS(y[list(p)], x[list(p)]).fit()

            err = np.linalg.norm(y[list(p)] - model.predict(x[list(p)]).reshape(an, -1))

            if err < err_min:
                self.inliers = list(p)
                self.model = model
                err_min = err

        return self

