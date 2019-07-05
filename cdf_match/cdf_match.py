import numpy as np
import scipy.stats


def cdf_match(X_cdf, x, presorted=False):
    if presorted:
        x = np.asarray(x)
    else:
        x = np.sort(x)
    n = len(x)
    i = np.arange(n)

    confidence_levels = scipy.stats.binom.cdf(i, n, X_cdf(x))
    confidence_levels = 2 * np.minimum(confidence_levels, 1-confidence_levels)

    if np.any(confidence_levels == 0):
        return 0

    return np.power(2, np.sum(np.log2(confidence_levels)) / n)


if __name__ == '__main__':
    rv = scipy.stats.norm()
    x = rv.rvs(100000)
    print(cdf_match(rv.cdf, x))
