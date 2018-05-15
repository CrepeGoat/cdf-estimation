# CDF Estimator
## Description
This is a small Python exercise to produce an estimate of a random variable's cumulative distribution function (cdf), given a set of independent and identically-distributed (I.I.D.) samples from said random variable. The estimator makes a calculated tradeoff between maximizing the *likelihood* of the CDF estimate, and maximizing the estimate's 2nd-degree *smoothness*.

## Installation
This project runs on Python 3, and requires the following packages to run:
- [NumPy](http://www.numpy.org/) - Base N-dimensional array package
- [SciPy](https://www.scipy.org/) - Fundamental library for scientific computing
- [CVXOPT](http://cvxopt.org/) - a quadratic programming solver

Additionally, the validity tests require the following packages:
- [SymPy](http://www.sympy.org/) - a symbolic logic package

With these packages installed, one need only download the main program file, *"cdf_est_CVXOPT.py"*, and include it in the desired project path.

## Usage
To use, just include *"cdf_est_CVXOPT.py"* in your project. The functionality is contained solely in the function, *cdf_est*, which returns a spline representing the CDF estimate.

## License
This poject is made available under the MIT License.
