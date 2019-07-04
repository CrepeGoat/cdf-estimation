import pytest

from likelihood import *

import numpy as np
import sympy as sp

np.random.seed(0)


def sp_nCk(n, k):
    return sp.gamma(n) / (sp.gamma(k) * sp.gamma(n-k))


def test_likelihood(monkeypatch):
    n = sp.symbols('n', integer=True, positive=True)
    k = sp.symbols('k', integer=True, nonnegative=True)
    p = sp.symbols('p', nonnegative=True)

    monkeypatch.setattr('likelihood.scipy.misc.comb', sp_nCk)

    assert sp.simplify(lhood(n, k, p) - lhood(n, n-k, p)) == 0


def test_log2_likelihood(monkeypatch):
    n = sp.symbols('n', integer=True, positive=True)
    k = sp.symbols('k', integer=True, nonnegative=True)
    p = sp.symbols('p', nonnegative=True)

    monkeypatch.setattr('likelihood.np.log2', sp.log)
    monkeypatch.setattr('likelihood.scipy.misc.comb', sp_nCk)

    assert sp.expand(log2_lhood(n, k, p) - sp.log(lhood(n, k, p))) == 0
