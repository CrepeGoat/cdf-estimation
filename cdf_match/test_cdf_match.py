import pytest
from cdf_match import *

import numpy as np
import scipy.stats


np.random.seed(0)


def check_cdf_match(rv, cdf, sample_count):
    confidence_level = 0.05
    repetitions = 100

    false_negatives = sum(
        cdf_match(cdf, rv.rvs(sample_count)) < confidence_level
        for _ in range(repetitions)
    )

    return false_negatives / repetitions < confidence_level


@pytest.mark.parametrize('rv', [
    scipy.stats.norm(),
    scipy.stats.expon(),
    scipy.stats.cauchy(),
])
def test_cdf_match_true_positives(rv):
    assert check_cdf_match(rv, rv.cdf, sample_count=10**3)


@pytest.mark.parametrize('rv_true, rv_guess', [
    (scipy.stats.expon(), scipy.stats.norm(1, 1)),
    (scipy.stats.norm(0, 1), scipy.stats.norm(0, 0.9)),
    (scipy.stats.norm(0, 0.9), scipy.stats.norm(0, 1)),
])
def test_cdf_match_false_positives(rv_true, rv_guess):
    assert check_cdf_match(rv_true, rv_guess.cdf, sample_count=10)


@pytest.mark.parametrize('rv_true, rv_guess', [
    (scipy.stats.norm(1, 1), scipy.stats.expon()),
    (scipy.stats.expon(), scipy.stats.norm(1, 1)),
    (scipy.stats.norm(0, 1), scipy.stats.norm(0, 0.9)),
    (scipy.stats.norm(0, 0.9), scipy.stats.norm(0, 1)),
])
def test_cdf_match_true_negatives(rv_true, rv_guess):
    assert not check_cdf_match(rv_true, rv_guess.cdf, sample_count=10**3)
