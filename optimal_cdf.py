from collections import namedtuple

import numpy as np
import scipy
import scipy.interpolate
import scipy.optimize

from likelihood_funcs import *


def init_parameters(X):
    n = len(X)
    n_mid = n // 2

    # straight line from first point to last point (when a^2 == 0)
    b_mid = ((n-1) / n) / (X[-1] - X[0])
    c = (.5 / n) + b_mid * (X - X[0])

    return np.concatenate(([b_mid], c))

    '''
    # interpolation through all points (when e^2 == 0)
    b_mid = (2/n) / (X[n_mid+1] - X[n_mid-1])
    c = np.arange(1, 2*n, 2) / (2*n)

    return np.concatenate(([b_mid], c))
    '''


class MatrixBuilder:
    def __init__(self, X):
        self._X = X
        self._b_mid, self.c = np.split(np.eye(self.n+1), [1], axis=0)

        self._b = None
        self._aDX = None

    @property
    def n(self):
        return len(self._X)

    @property
    def b(self):
        if self._b is None:
            n_mid = self.n // 2

            alt_sign = (-1) ** np.arange(self.n)[:, np.newaxis]
            Dc_div_DX = np.diff(self.c, axis=0) / np.diff(self._X)[:, np.newaxis]

            b_lower, b_upper = np.array_split(
                -2 * alt_sign[:-1] * Dc_div_DX,
                (n_mid,),
                axis=0
            )
            b_cumdiff = np.concatenate([
                -b_lower[::-1].cumsum(axis=0)[::-1],
                np.zeros_like(self._b_mid),
                b_upper.cumsum(axis=0),
            ], axis=0)
            self._b = alt_sign * (b_cumdiff + alt_sign[n_mid]*self._b_mid)

        return self._b

    @property
    def aDX(self):
        if self._aDX is None:
            self._aDX = np.diff(self.b, axis=0, prepend=0, append=0) / 2

        return self._aDX

    def __iter__(self):
        return iter((self.aDX, self.b, self.c))

    def pull_values_from(self, params):
        return (
            self.aDX.dot(params),
            self.b.dot(params),
            self.c.dot(params),
        )


def extend_samples(X, aDX, b, c):
    # Add leading/trailing endpoint regions. I.e., adds:
    #    1) knots X0, Xnp1 that smoothly joins curve to the constant-value regions
    #        P(x) = 0 as x -> -inf,
    #        P(x) = 1 as x ->  inf
    #    2) 'dead knots' Xm1, Xnp2 with zero-valued derivatives & P = {0,1},
    #        from which PPoly can extrapolate for x values outside of the
    #        sampled region
    X0m1 = X[0] - 2*c[0] / b[0]
    Xnp1 = X[-1] + 2*(1 - c[-1]) / b[-1]
    X0m2 = X0m1 - (Xnp1-X0m1)
    Xnp2 = Xnp1 + (Xnp1-X0m1)

    return np.concatenate(([X0m2, X0m1], X, [Xnp1, Xnp2]))


def make_spline(params, X):
    aDX, b, c = MatrixBuilder(X).pull_values_from(params)

    X = extend_samples(X, aDX, b, c)
    a = np.concatenate(([0], aDX, [0])) / np.diff(X)
    b = np.concatenate(([0, 0], b, [0]))
    c = np.concatenate(([0, 0], c, [1]))

    return scipy.interpolate.PPoly(
        np.stack((a, b, c), axis=0), X, extrapolate=True,
    )


def lhood_quad_coeffs(n):
    return -d2dp2_rlhood(n, np.arange(n))


def make_obj_func(X):
    p_exp = MatrixBuilder(X)
    est_c = np.linspace(0, 1, 2*len(X)+1)[1::2]

    def min_obj_func(params):
        aDX, b, c = p_exp.pull_values_from(params)

        return (
            np.log2(np.sum(aDX**2 / np.diff(extend_samples(X, aDX, b, c))[1:-1]))
            - np.sum(np.log2(1 + lhood_quad_coeffs(p_exp.n) * (c - est_c)**2))
        )

    return min_obj_func


def make_bi_constraints(X):
    return scipy.optimize.LinearConstraint(
        MatrixBuilder(X).b,
        0, np.inf, keep_feasible=True
    )


def make_delta_ci_constraints(X):
    return scipy.optimize.LinearConstraint(
        np.diff(
            MatrixBuilder(X).c,
            axis=0, prepend=0, append=1,
        ),
        0, np.inf, keep_feasible=True
    )


def clean_samples(X):
    X = np.asarray(X)

    if not np.all(X[1:] >= X[:-1]):
        X.sort()
    assert np.all(X[1:] > X[:-1])  # avoids case of duplicate X-values

    return X


def cdf_approx(X):
    """
    Generates a ppoly spline to approximate the cdf of a random variable,
    from a 1-D array of i.i.d. samples thereof.

    Args:
        X: a collection of i.i.d. samples from a random variable.
        args, kwargs: any options to forward to the cvxopt qp solver
    Returns:
        scipy.interpolate.PPoly object, estimating the cdf of the random variable.
    Raises:
        TODO
    """
    # Pre-format input as ordered numpy array
    X = clean_samples(X)

    results = scipy.optimize.minimize(
        make_obj_func(X),
        init_parameters(X),
        constraints=[make_bi_constraints(X), make_delta_ci_constraints(X)],
    )

    if not results.success:
        raise RuntimeError("failed optimization: " + results.message)

    return make_spline(results.x, X)
