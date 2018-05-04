import pytest

from cdf_est_CVXOPT import *

import numpy as np
import sympy as sp


'''
Author: Becker Awqatty

The below functions are meant to execute a set of tests for the
QP-enabled cdf estimate function code.

For each function to test, the input parameters (formatted as numpy
arrays) are stubbed with sympy symbols (np.object arrays filled with
sympy expressions). The test the checks if the symbolic result returned
by the function is mathematically equivalent to an expected result.

This allows for a simple approach to ensuring precise mathematical
correctness with 100% confidence, while avoiding the effort required
for designing comprehensive test cases. The downside in this method
is that rounding errors are not considered when comparing calculated
and expected results; however this is acceptable for my purposes.
'''

min_test_size = 3

test_size = 12
small_test_size = 5

def sp_stub_bmid_c_bmidc_X(n):
    '''
    This function generates some commonly used sympy symbol arrays
    that are utilized in the testing functions.
    '''
    b_mid = sp.symbols('b_mid', nonnegative=True)
    
    c = np.array(sp.symbols(
        ['_'.join(('c', str(i))) for i in range(n)],
        nonnegative=True))
    
    X = np.array(sp.symbols(
        ['_'.join(('x', str(i))) for i in range(n)]))
    
    bmid_c = np.concatenate(([b_mid], c))

    return b_mid, c, bmid_c, X

def sp_stub_s_t(n):
    scale_a = np.array(sp.symbols(
        ['_'.join(('s', str(i))) for i in range(n-1)],
        positive=True))
    
    scale_c = np.array(sp.symbols(
        ['_'.join(('t', str(i))) for i in range(n)],
        positive=True))
    
    return scale_a, scale_c







@pytest.mark.parametrize("n", [i for i in range(min_test_size, test_size+1)])
def test_sp_expand_vars(n):
    n_mid = n//2
    b_mid, c, bmid_c, X = sp_stub_bmid_c_bmidc_X(n)

    vars = expand_vars(bmid_c, X)
    
    assert vars.b[n//2] == b_mid
    assert (vars.c == c).all()
    
    tmp = np.vectorize(sp.expand)(
        (vars.a_diffX + vars.b[:-1]) * np.diff(X) - np.diff(vars.c)
        )
    np.vectorize(sp.pretty_print)(tmp)
    assert (0 == np.vectorize(sp.expand)(
        (vars.a_diffX + vars.b[:-1]) * np.diff(X) - np.diff(vars.c)
        )).all()

    assert (0 == np.vectorize(sp.expand)(
        (vars.b[1:] + vars.b[:-1]) * np.diff(X) - 2 * np.diff(vars.c)
        )).all()

    assert (0 == np.vectorize(sp.expand)(
        2 * vars.a_diffX - np.diff(vars.b)
        )).all()

@pytest.mark.parametrize("n", [i for i in range(min_test_size, test_size+1)])
def test_sp_expand_vars_lc(n):
    b_mid, c, bmid_c, X = sp_stub_bmid_c_bmidc_X(n)
    
    vars = expand_vars(bmid_c, X)
    vars_lc = expand_vars_lc(X)
    
    for var, var_lc, var_name in zip(vars, vars_lc, 'a_diffX b c'.split(' ')):
        assert (0 == np.vectorize(sp.expand)(
            var - np.sum(var_lc * bmid_c, axis=-1)
            )).all()

@pytest.mark.parametrize("n", [i for i in range(min_test_size, small_test_size+1)])
def test_sp_make_P_q(n):
    b_mid, c, bmid_c, X = sp_stub_bmid_c_bmidc_X(n)
    scale_a, scale_c = sp_stub_s_t(n)
    
    bmid_c_mat = np.asmatrix(bmid_c).T
    
    a_diffX_lc, b_lc, c_lc = expand_vars_lc(X)
    a_diffX, b, c = tuple(
        np.sum(var * bmid_c, axis=-1)
        for var in (a_diffX_lc, b_lc, c_lc)
        )
    
    P, q = make_P_q(X, scale_a, scale_c, autoscale=False)
    
    # from:     http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    res_calc = ((bmid_c_mat.T * P * bmid_c_mat)/2 + q.T * bmid_c_mat)[0,0]
    res_expected = (np.sum(scale_a * a_diffX * a_diffX / np.diff(X))
        + np.sum(scale_c * (c - sp.Rational(1, 2*n) * np.arange(1,2*n,2)) ** 2))    
    res_cancelled = sp.expand(res_calc - res_expected)
    
    assert res_cancelled == res_cancelled.subs(zip(bmid_c, np.zeros_like(bmid_c)))


@pytest.mark.parametrize("n", (test_size,))
def test_sp_make_G_h(n):
    '''
    This function tests that the results from "make_G_h" are accurate.
    
    Currently, this test is unimplemented.
    
    The code in the "make_G_h" function is simplistic, to the point
    that any test routine would simply copy the original function's
    logic and test the respective results for equality. And such a
    test would not serve to validate the function's results.
    
    Further, finding an appropriate way to test such a function is
    a relatively low priority.
    '''
    pass

@pytest.mark.parametrize("n", (test_size,))
def test_bmid_c_init_state(n):
    X = np.random.rand(n)
    X.sort()
    
    pass
    

@pytest.mark.parametrize("n", (test_size,))
def test_clean_optimizer_results(n):
    
    '''
    status = True
    fail_message = ""
    
    if not (P_X >= 0).all():
        status = False
        fail_message += "bound 0 (c >= 0) fails\n"
    
    if not (P_X <= 1).all():
        status = False
        fail_message += "bound 1 (c <= 1) fails\n"
    
    if not (dP_X >= 0).all():
        status = False
        fail_message += "bound 2+ (b >= 1) fails\n"
    
    if not (np.diff(X) >= 0).all():
        status = False
        fail_message += "X extension fails\n"
    
    if not status:
        print(fail_message)
        print(X)
        print(np.arange(X.shape[0]-1)[np.diff(X) < 0])
        print(P_X)
        print(dP_X)
        print(d2P_X)
        assert False
    '''
    pass
    

if __name__ == "__main__":
    pytest.main()
