from itertools import chain
import functools

import numpy as np


def nCkarray(k_array):
    result = 1
    for i, j in enumerate(chain(*(range(1, k+1) for k in k_array)), 1):
        result = (result * i) // j
    return result


@functools.lru_cache(maxsize=None)
def log2_nPk(n, k):
    return np.sum(np.log2(np.arange(n, n-k, -1)))


def log2_factorial(n):
    return log2_nPk(n, n)


def log2_nCk(n, k):
    k = min(k, n-k)
    return log2_nPk(n, k) - log2_factorial(k)


def log2_nCkarray(k_array):
    n = np.sum(k_array)

    return log2_factorial(n) - sum(
        log2_factorial(k_i) for k_i in k_array
    )

# k indexed in [0,n)
def p_max_likelihood(n, k):
    """
    Calculates the maximum likelihood estimator for the kth sample
    of n total samples.

    Args:
        n: the number of samples from the random variable, X
        k: the hierarchical position of the sample when ordered
            k in [0,n)
    Returns:
        the maximum likelihood estimator, p_hat, such that
        Lhood(Prob(X < Xk) = p_hat) is maximized
    Raises:
        N/A
    """
    return (k+.5) / n


def lhood(n, k, p=None):
    """
    Calculates the likelihood that Prob(x < x_k) == p for the kth sample
    of n total samples.

    Args:
        n: the number of samples from the random variable, X
        k: the hierarchical position of the sample when ordered
            (k in [0,n))
        p: the probability that X < Xk. Set to max. likelihood est. by default.
    Returns:
        the likelihood that Prob(x < x_k) == p
    Raises:
        N/A
    """
    if p is None:
        p = p_max_likelihood(n, k)

    return nCkarray([k, n-k]) * p**(k+.5) * (1-p)**(n-(k+.5))


def log2_lhood(n, k, p=None):
    """
    Calculates the logarithm (base-2) of the result from the lhood function.
    """
    if p is None:
        p = p_max_likelihood(n, k)

    return log2_nCk(n, k) + np.log2(p)*(k+.5) + np.log2(1-p)*(n-(k+.5))


def d2dp2_lhood(n, k, p=None):
    """
    Calculates the 2nd derivative of the likelihood function with respect to p.

    Args:
        n: the number of samples from the random variable, X
        k: the hierarchical position of the sample when ordered
            (k in [1,n])
        p: the probability that X < Xk. Set to max. likelihood est. by default.
    Returns:
        (d2L/dp2)(p)
    Raises:
        N/A
    """
    if p is None:
        p = p_max_likelihood(n, k)

    return (
        nCkarray([k, n-k])
        * p**(k-1.5)
        * (1-p)**(n-k-2.5)
        * ((n-1)*p*(n*p - 2*k - 1) + k**2 - .25)
    )


def d2dp2_rlhood(n, k, p=None):
    """
    Calculates the 2nd derivative of the max-unity-scaled likelihood function
    with respect to p.

    Args:
        n: the number of samples from the random variable, X
        k: the hierarchical position of the sample when ordered
            (k in [1,n])
        p: the probability that X < Xk. Set to max. likelihood est. by default.
    Returns:
        (d2L/dp2)(p) / L(p_hat)
    Raises:
        N/A
    """
    phat = p_max_likelihood(n, k)
    if p is None:
        p = phat

    return (
        (p/phat)**(k-1.5)
        * ((1-p)/(1-phat))**(n-k-2.5) / (phat*(1-phat))**2.
        * ((n-1)*p*(n*p - 2*k - 1) + k**2 - .25)
    )
