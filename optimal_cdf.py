from collections import namedtuple

import numpy as np
import scipy
import scipy.interpolate

from likelihood_funcs import *


def init_parameters(X):
    n = len(X)
    n_mid = n // 2

    '''
    # straight line from first point to last point (when a^2 == 0)
    b_mid = ((n-1) / n) / (X[-1] - X[0])
    c = (.5 / n) + b_mid * (X - X[0])

    return np.concatenate(([b_mid], c))
    '''

    # interpolation through all points (when e^2 == 0)
    b_mid = (2/n) / (X[n_mid+1] - X[n_mid-1])
    c = np.arange(1, 2*n, 2) / (2*n)

    return np.concatenate(([b_mid], c))


class OptimizerVariableExpander:
    def __init__(self, params, X):
        self._b_mid, self.c = np.split(params, [1])
        self._X = X

        self._b = None
        self._a_DX = None

    @property
    def n(self):
        return len(self.c)

    @property
    def b(self):
        if self._b is None:
            n_mid = self.n // 2

            alt_sign = (-1) ** np.arange(self.n)
            Dc_div_DX = np.diff(self.c, axis=-1) / np.diff(self._X, axis=-1)

            b_lower, b_upper = np.array_split(
                -2 * alt_sign[:-1] * Dc_div_DX,
                (n_mid,),
                axis=-1
            )
            b_cumdiff = np.concatenate([
                -b_lower[..., ::-1].cumsum(-1)[..., ::-1],
                np.zeros_like(self._b_mid),
                np.cumsum(b_upper, -1),
            ], axis=-1)
            self._b = alt_sign * (b_cumdiff + alt_sign[n_mid]*self._b_mid)

        return self._b

    @property
    def a_DX(self):
        if self._a_DX is None:
            self._a_DX = np.diff(self.b, axis=-1, prepend=0, append=0) / 2

        return self._a_DX

    def __iter__(self):
        return iter((self.a_DX, self.b, self.c))


def make_spline(params, X):
    a_DX, b, c = OptimizerVariableExpander(params, X)

    # Add leading/trailing endpoint regions. I.e., adds:
    #    1) knots X0, Xnp1 that smoothly joins curve to the constant-value regions
    #        P(x) = 0 as x -> -inf,
    #        P(x) = 1 as x ->  inf
    #    2) 'dead knots' Xm1, Xnp2 with zero-valued derivatives & P = {0,1},
    #        from which PPoly can extrapolate for x values outside of the
    #        sampled region
    X0m1 = X[0] - 2*c[0] / b[0]
    Xnp1 = X[-1] + 2*(1 - c[-1]) / b[-1]

    a = np.concatenate(([0], a_DX, [0]))
    b = np.concatenate(([0, 0], b, [0]))
    c = np.concatenate(([0, 0], c, [1]))

    X = np.concatenate(([2*X0m1-X[0], X0m1], X, [Xnp1, 2*Xnp1-X[-1]]))

    return scipy.interpolate.PPoly(
        np.stack((a, b, c), axis=0), X, extrapolate=True,
    )


def lhood_quad_coeffs(n):
    return -d2dp2_rlhood(n, np.arange(n))


def min_obj_function(params, X):
    p_exp = OptimizerVariableExpander(params, X)
    a_DX, b, c = p_exp
    est_c = np.linspace(0, 1, 2*len(c)+1)[1::2]

    return (
        np.sum(a_DX**2)
        / np.prod(1 + lhood_quad_coeffs(p_exp.n) * (c - est_c)**2)
    )


def make_bi_constraint(i, X):
    def fun(params):
        return OptimizerVariableExpander(params, X).b[i]

    return {'type': 'ineq', 'fun': fun}


def make_delta_ci_constraint(i, X):
    def fun(params):
        pexp = OptimizerVariableExpander(params, X)
        return pexp.c[i+1] - pexp.c[i]

    return {'type': 'ineq', 'fun': fun}


def clean_samples(X):
    X = np.asarray(X)

    if not np.all(X[1:] >= X[:-1]):
        X.sort()
    assert np.all(X[1:] > X[:-1])  # avoids case of duplicate X-values


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
    clean_samples(X)
    n = len(X)

    results = scipy.optimize.minimize(
        min_obj_function,
        init_parameters(X),
        args=(X,),
        constraints=[
            make_bi_constraint(i, X) for i in range(n)
        ] + [
            make_delta_ci_constraint(i, X) for i in range(n-1)
        ] + [
            {
                'type': 'ineq',
                'fun': lambda params: OptimizerVariableExpander(params, None).c[0]
            },
            {
                'type': 'ineq',
                'fun': lambda params: 1-OptimizerVariableExpander(params, None).c[-1]
            },
        ],
    )

    return make_spline(results.x, X)
