import cProfile

import numpy as np

from optimal_cdf import *


def main():
    X = np.arange(0, 60, 3) + np.random.random(20)

    prof = cProfile.Profile()
    prof.enable()

    cdf = cdf_approx(X)

    prof.disable()
    prof.print_stats(sort='tottime')


if __name__ == '__main__':
    main()
