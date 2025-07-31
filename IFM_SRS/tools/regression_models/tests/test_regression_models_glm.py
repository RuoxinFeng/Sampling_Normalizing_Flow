#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit as logistic
from scipy.stats import bernoulli
from tools.regression_models.regression_models import ConvenientGlmFit
from tools.remove_invalid_sessions.remove_invalid_sessions import TotalFailureInfo


def bernoulli_vec(p: np.ndarray) -> List[int]:
    samples = []
    for pi in p:
        dist = bernoulli(pi)
        samples.append(dist.rvs())
    return samples


@pytest.fixture
def simple_binary_data() -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    beta = (-2.0, 3.0, -4.0)
    np.random.seed(3425343)
    n = 10000
    X = np.random.randn(n, 2)
    X_with_intercept = np.insert(X, 0, 1, axis=1)
    linear_predictor = np.dot(X_with_intercept, beta)
    p = logistic(linear_predictor)  # link function
    y = bernoulli_vec(p)
    df = pd.DataFrame({"y_1234567a": y, "x1_1234567": X[:, 0], "x2_12345678": X[:, 1]})
    return df, beta


def srs_setup(df: pd.DataFrame, formula: str, family: sm.genmod.families.family.Family) -> ConvenientGlmFit:
    df["session_id"] = range(1, len(df) + 1)
    return ConvenientGlmFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        family=family,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )


@pytest.fixture
def simple_logistic_regression_setup(
    simple_binary_data: Tuple[pd.DataFrame, Tuple[float, float, float]]
) -> ConvenientGlmFit:
    df, _ = simple_binary_data
    formula = "y_1234567a ~ x1_1234567 + x2_12345678"
    return srs_setup(df=df, formula=formula, family=sm.families.Binomial())


def test_logistic_regression(
    simple_binary_data: Tuple[pd.DataFrame, Tuple[float, float, float]],
    simple_logistic_regression_setup: ConvenientGlmFit,
):
    _, expected_params = simple_binary_data
    actual_params = simple_logistic_regression_setup.result.params
    assert np.allclose(actual_params, expected_params, atol=0.1)


def test_logistic_regression_summary(simple_logistic_regression_setup: ConvenientGlmFit):
    simple_logistic_regression_setup.print_summary()


def test__logistic_regression_plots__smoke(simple_logistic_regression_setup: ConvenientGlmFit, mock_plots: None):
    simple_logistic_regression_setup.resid_qqplot()
    simple_logistic_regression_setup.plot_residuals_hist()
    simple_logistic_regression_setup.plot_residuals()
    simple_logistic_regression_setup.plot_partregress_grid()
    simple_logistic_regression_setup.plot_ccpr_grid()
