from collections import namedtuple

import numpy as np
from scipy.interpolate import PPoly
import cvxopt
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['maxiters'] = 500

from likelihood_funcs import *


'''
TODO

- improve "smoothness" input parameter
    - change to operate on a [0,1] scale (0 = exact interpolation, 1 = uniform distribution)
- format tests more professionally
'''

def expand_vars(b0_c, X):
    n = X.shape[-1]
    b0_c = np.asarray(b0_c)
    b_0, c = b0_c[..., 0:1], b0_c[..., 1:]

    alt_sign = (-1) ** np.arange(n)
    diffc_diffX = np.diff(c, axis=-1) / np.diff(X, axis=-1)

    b = alt_sign * np.concatenate((
        b_0,
        b_0 + np.cumsum(-2 * alt_sign[:-1] * diffc_diffX, axis=-1)
        ), axis=-1)
    #a_diffX = diffc_diffX - b[..., :-1]
    a_diffX = np.diff(b, axis=-1) / 2

    return namedtuple("SplineVars", "a_diffX b c")(a_diffX, b, c)

def expand_vars_lc(X):
    n = X.shape[-1]

    b0_c = np.diagflat(np.ones(n+1, dtype=np.int64))
    b_0, c = b0_c[0:1], b0_c[1:]

    diffc_diffX = np.diff(c, axis=0) / np.diff(X)[:, np.newaxis]
    alt_sign = (-1) ** np.arange(n)[:, np.newaxis]

    b = alt_sign * np.cumsum(
        alt_sign * np.concatenate((b_0, 2*diffc_diffX), axis=0),
        axis=0)

    a_diffX = np.diff(b, axis=0) / 2

    return namedtuple("SplineVars", "a_diffX b c")(a_diffX, b, c)






def make_obj_scale(X, smoothness_factor=1):
    n = X.shape[-1]
    scale_a = smoothness_factor / (X[-1] - X[0])
    scale_c = -d2dp2_rlhood(n, np.arange(n)) * 10 / (n.bit_length() * n)

    return namedtuple("ObjectiveScales", "scale_a scale_c")(scale_a, scale_c)


def make_P_q(X, scale_a=np.ones(1), scale_e=np.ones(1), autoscale=True):
    n = X.shape[-1]
    a_diffX, b, c = expand_vars_lc(X)
    outer_prod = lambda col: col * col[:,np.newaxis]

    P_a = np.sum(
        np.apply_along_axis(outer_prod, -1, a_diffX)
            * ((scale_a / np.diff(X))[:, np.newaxis, np.newaxis]),
        axis=0)
    q_a = np.zeros(None)

    P_c = np.sum(
        np.apply_along_axis(outer_prod, -1, c)
            * (scale_e[:, np.newaxis, np.newaxis]),
        axis=0)
    q_c = np.sum(
        -(scale_e * np.arange(1, 2*n, 2) / n)[:, np.newaxis] * c,
        axis=0)
    
    P = 2*(P_a + P_c)
    q = q_a + q_c
    
    if autoscale:
        min_val = min(
            np.min(np.abs(P[P != 0])), 
            np.min(np.abs(q[q != 0]))
            )
        max_val = max(
            np.max(np.abs(P)),
            np.max(np.abs(q))
            )
        
        scale = 2 ** -(
            # centers exponent range on zero
            np.mean((np.frexp(min_val)[1], np.frexp(max_val)[1]))
            # biases range to account for sums of n values
            #+ n.bit_length() / 2
            )
        
        P = P * scale
        q = q * scale
        
    res = namedtuple("QuadProgramObj", "P q")(
        np.asmatrix(P),
        np.asmatrix(q).T
        )
    return res


def make_G_h(X):
    n = X.shape[-1]
    a_diffX, b, c = expand_vars_lc(X)

    G_b = -b
    h_b = np.zeros(b.shape[0])

    G_c0 = -c[:1]
    h_c0 = np.zeros(1)

    G_cnm1 = c[-1:]
    h_cnm1 = np.ones(1)

    return namedtuple("QuadProgramBounds", "G h")(
        np.asmatrix(np.concatenate((G_b, G_c0, G_cnm1), axis=0)),
        np.asmatrix(np.concatenate((h_b, h_c0, h_cnm1), axis=0)).T,
        )

def make_A_b(X):
    return namedtuple("QuadProgramBounds", "A b")(
        np.zeros((0, X.shape[-1]+1)),
        np.zeros(0),
        )



def b0_c_init_state(X):
    n = len(X)

    '''
    # straight line from first point to last point (when a^2 == 0)
    b_0 = ((n-1) / n) / (X[-1] - X[0])
    c = (.5 / n) + b_0 * (X - X[0])

    return np.concatenate(([b_0], c))
    '''

    # interpolation through all points (when e^2 == 0)
    b_0 = (1/n) / (X[1]-X[0])
    c = np.arange(1,2*n,2) / (2*n)
    return np.concatenate(([b_0], c))
    #'''



def clean_optimizer_results(b0_c_opt, X):
    n = len(X)
    b0_c_opt = np.squeeze(np.array(b0_c_opt))
    
    d2P_X, dP_X, P_X = expand_vars(b0_c_opt, X)
    d2P_X = d2P_X / np.diff(X)
    # Add leading/trailing endpoint regions. I.e., adds:
    #    1) knots X0, Xnp1 that smoothly joins curve to the constant-value regions
    #        P(x) = 0 as x -> -inf,
    #        P(x) = 1 as x ->  inf
    #    2) 'dead knots' Xm1, Xnp2 with zero-valued derivatives & P = {0,1},
    #        from which PPoly can extrapolate for x values outside of the
    #        sampled region
    d2P_X = np.concatenate((
        np.zeros(1),
        dP_X[:1]**2 / (4*P_X[0]),
        d2P_X,
        -dP_X[-1:]**2 / (4*(1-P_X[-1])),
        np.zeros(1)
        ))

    X0 = X[:1] - 2 * P_X[0] / dP_X[0]
    Xnp1 = X[-1:] + 2 * (1-P_X[-1]) / dP_X[-1]
    X = np.concatenate((
        X0 - (X[0] - X0),    # dead knot - included for extrapolation to -inf
        X0,
        X,
        Xnp1,
        Xnp1 + (Xnp1 - X[-1]),    # dead knot - included for extrapolation to inf
        ))

    P_X = np.concatenate((np.zeros(2), P_X, np.ones(1)))
    dP_X = np.concatenate((np.zeros(2), dP_X, np.zeros(1)))

    return X, P_X, dP_X, d2P_X















def cdf_approx(X): #, smoothness_factor=1):
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
    X = np.asarray(X)
    diff_X = np.diff(X)
    if not (diff_X > 0).all():
        X.sort()
        diff_X = np.diff(X)
    assert(diff_X.all())    # avoids case of duplicate X-values
    n = len(X)

    scale_axi, scale_ei = make_obj_scale(X)#, smoothness_factor)

    P, q = make_P_q(X, scale_a=scale_axi, scale_e=scale_ei)
    G, h = make_G_h(X)
    #A, b = make_A_b(X) # simply unnecessary
    b0_c_init = b0_c_init_state(X)
    
    qp_res = cvxopt.solvers.qp(
        cvxopt.matrix(P),
        cvxopt.matrix(q),
        cvxopt.matrix(G),
        cvxopt.matrix(h),
        #cvxopt.matrix(A),
        #cvxopt.matrix(b),
        #*args, **kwargs
        )

    X, P_X, dP_X, d2P_X = clean_optimizer_results(np.array(qp_res['x']), X)

    return PPoly.construct_fast(np.stack((d2P_X, dP_X, P_X)), X, extrapolate=True)


if __name__ == "__main__":
    from cdf_est_CVXOPT_tests import run_all_tests
    run_all_tests()
