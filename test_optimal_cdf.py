import pytest

from optimal_cdf import *

import numpy as np
import sympy as sp

np.random.seed(0)

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
    b_mid = sp.symbols('b_{}'.format(n // 2), nonnegative=True)
    c = sp.symbols('c_:{}'.format(n), nonnegative=True)
    params = np.concatenate(([b_mid], c))

    X = np.array(sp.symbols('x_:{}'.format(n)), dtype=np.object)

    return b_mid, c, params, X


@pytest.mark.parametrize("n", [i for i in range(min_test_size, test_size+1)])
def test_sp_expand_vars(n):
    n_mid = n//2
    b_mid, c, params, X = sp_stub_bmid_c_bmidc_X(n)

    sp_aDX, sp_b, sp_c = MatrixBuilder(X)

    np_sp_simplify = np.vectorize(sp.expand)

    def sp_eq(expr1, expr2):
        return np.all(np_sp_simplify(expr1 - expr2) == 0)

    assert sp_eq(sp_b[..., n_mid].dot(params), b_mid)
    assert sp_eq(sp_c.T.dot(params), c)

    assert sp_eq(
        np.diff(sp_c, axis=-1),
        (sp_aDX[..., 1:-1] + sp_b[..., :-1]) * np.diff(X)
    )
    assert sp_eq(
        2*np.diff(sp_c, axis=-1),
        (sp_b[..., 1:] + sp_b[..., :-1]) * np.diff(X)
    )
    assert sp_eq(2*sp_aDX[..., 1:-1], np.diff(sp_b, axis=-1))


@pytest.mark.parametrize("n", (test_size,))
def test_bmid_c_init_state(n):
    X = np.random.rand(n)
    X.sort()

    params_0 = init_parameters(X)

    a_DX, b, c = MatrixBuilder(X).pull_values_from(params_0)

    assert np.all(b >= 0)
    assert np.all(np.diff(c, prepend=0, append=1) >= 0)


@pytest.mark.parametrize("n", (test_size,))
def test_clean_optimizer_results(n):
    pass

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


if __name__ == "__main__":
    pytest.main()
