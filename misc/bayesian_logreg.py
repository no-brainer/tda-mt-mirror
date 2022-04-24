import numpy as np
from sklearn.base import BaseEstimator
from scipy.linalg import solve_triangular
from sklearn.linear_model._base import LinearClassifierMixin
from scipy.special import expit, exprel
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets


def lam(eps):
    """
    Calculates lambda eps (used for Jaakola & Jordan local bound)
    """
    eps = -abs(eps)
    return 0.25 * exprel(eps) / (np.exp(eps) + 1)


class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):

    def __init__(self, n_iter: int, tol: float, fit_intercept: bool, verbose: bool):
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_ = [0] * n_classes
            self.sigma_ = [0] * n_classes
            self.intercept_ = [0] * n_classes
        else:
            self.coef_ = [0]
            self.sigma_ = [0]
            self.intercept_ = [0]

        # make classifier for each class (one-vs-the rest)
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask] = self._mask_val
            coef_, sigma_ = self._fit(X, y_bin)
            if self.fit_intercept:
                self.intercept_[i], self.coef_[i] = self._get_intercept(coef_)
            else:
                self.coef_[i] = coef_
            self.sigma_[i] = sigma_

        self.coef_ = np.asarray(self.coef_)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # construct separating hyperplane
        scores = self.decision_function(X)
        if self.fit_intercept:
            X = self._add_intercept(X)

        # probit approximation to predictive distribution
        sigma = self._get_sigma(X)
        ks = 1. / ( 1. + np.pi*sigma / 8)**0.5
        probs = expit(scores.T*ks).T

        # handle several class cases
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_intercept(self, coef: np.ndarray):
        raise NotImplementedError

    def _get_sigma(self, X):
        raise NotImplementedError


class VBLogisticRegression(BayesianLogisticRegression):
    """
    Variational Bayesian Logistic Regression with local variational approximation.

    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 50 )
       Maximum number of iterations

    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated

    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting

    a: float, optional (DEFAULT = 1e-6)
       Rate parameter for Gamma prior on precision parameter of coefficients

    b: float, optional (DEFAULT = 1e-6)
       Shape parameter for Gamma prior on precision parameter of coefficients

    verbose: bool, optional (DEFAULT = False)
       Verbose mode


    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    intercept_: array, shape = (n_features)
        intercepts
    """
    def __init__(self,  n_iter: int = 50, tol: float = 1e-3, fit_intercept: bool = True,
                 a: float = 1e-4, b: float = 1e-4, verbose: bool = True):
        super().__init__(n_iter, tol, fit_intercept, verbose)
        self.a = a
        self.b = b
        self._mask_val = 0.

    def _fit(self, X, y):
        eps = 1
        n_samples, n_features = X.shape
        XY = np.dot(X.T, (y-0.5))
        w0 = np.zeros(n_features)

        # hyperparameters of q(alpha) (approximate distribution of precision
        # parameter of weights)
        a = self.a + 0.5 * n_features
        b = self.b

        for i in range(self.n_iter):
            # In the E-step we update approximation of
            # posterior distribution q(w,alpha) = q(w)*q(alpha)

            # --------- update q(w) ------------------
            l = lam(eps)
            w, Ri = self._posterior_dist(X, l, a, b, XY)

            # -------- update q(alpha) ---------------
            if self.fit_intercept:
                b = self.b + 0.5 * (np.sum(w[1:] ** 2) + np.sum(Ri[1:, :] ** 2))
            else:
                b = self.b + 0.5 * (np.sum(w ** 2) + np.sum(Ri ** 2))

            # -------- update eps  ------------
            # In the M-step we update parameter eps which controls
            # accuracy of local variational approximation to lower bound
            XMX = np.dot(X, w) ** 2
            XSX = np.sum(np.dot(X, Ri.T) ** 2, axis=1)
            eps = np.sqrt(XMX + XSX)

            # convergence
            if not np.any(abs(w - w0) > self.tol) or i + 1 == self.n_iter:
                break
            w0 = w

        l = lam(eps)
        coef_, sigma_ = self._posterior_dist(X, l, a, b, XY, True)
        return coef_, sigma_

    def _add_intercept(self, X):
        return np.hstack((np.ones([X.shape[0], 1]), X))

    def _get_intercept(self, coef):
        return coef[0], coef[1:]

    def _get_sigma(self, X):
        return np.asarray([np.sum(np.dot(X, s) * X, axis=1) for s in self.sigma_])

    def _posterior_dist(self, X, l, a, b, XY, full_covar=False):
        sigma_inv = 2 * np.dot(X.T * l, X)
        alpha_vec = np.ones(X.shape[1]) * float(a) / b
        if self.fit_intercept:
            alpha_vec[0] = np.finfo(np.float16).eps

        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        R = np.linalg.cholesky(sigma_inv)
        Z = solve_triangular(R, XY, lower=True)
        mean = solve_triangular(R.T, Z, lower=False)

        Ri = solve_triangular(R, np.eye(X.shape[1]), lower=True)
        if full_covar:
            sigma = np.dot(Ri.T, Ri)
            return mean, sigma
        else:
            return mean, Ri
