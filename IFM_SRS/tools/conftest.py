#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest
import scipy.stats as st
from matplotlib import pyplot as plt
from tools.regression_models.regression_models import ConvenientMleFit, ConvenientOlsFit
from tools.remove_invalid_sessions.remove_invalid_sessions import TotalFailureInfo

TEST_TIMESTAMPS = {"timestamp_input_data": "30/11/2023 09:07:37", "timestamp_export": "29/11/2023 10:07:37"}


def simple_data_setup(seed: int = 9876789, endog_name: str = "y_1234567a") -> Tuple[pd.DataFrame, pd.Series]:
    np.random.seed(seed)

    # True coefficients for the linear model
    beta0 = 2.0
    beta1 = 3.0
    beta2 = -1.5
    beta3 = 0.05
    beta4 = 0.00000001

    # Randomly generated features
    n = 1000
    x1_1234567, x2_12345678, x3_1234567a, x4_12345678b = np.random.randn(4, n)
    U = np.random.normal(size=n)
    # Generate the target variable with some random noise
    y_1234567a = beta0 + beta1 * x1_1234567 + beta2 * x2_12345678 + beta3 * x3_1234567a + beta4 * x4_12345678b + U

    df = pd.DataFrame(
        {
            endog_name: y_1234567a,
            "x1_1234567": x1_1234567,
            "x2_12345678": x2_12345678,
            "x3_1234567a": x3_1234567a,
            "x4_12345678b": x4_12345678b,
            "U": U,
        }
    )
    params = {
        "Intercept": beta0,
        "x1_1234567": beta1,
        "x2_12345678": beta2,
        "x3_1234567a": beta3,
        "x4_12345678b": beta4,
    }
    return df, pd.Series(params)


@pytest.fixture
def simple_data() -> Tuple[pd.DataFrame, pd.Series]:
    return simple_data_setup()


@pytest.fixture
def alternative_simple_data() -> Tuple[pd.DataFrame, pd.Series]:
    return simple_data_setup(seed=12345678, endog_name="y1_12345678b")


@pytest.fixture
def data_with_transformation_quadratic_x() -> Tuple[pd.DataFrame, pd.Series]:
    # True coefficients for the linear model
    beta0 = 2.0
    beta1 = 3.0
    beta2 = -0.5

    # Randomly generated features
    np.random.seed(9876789)
    n = 10000
    x1_1234567, x2_12345678 = np.random.randn(2, n)
    # Generate the target variable with some random noise
    y_1234567a = (
        beta0 + beta1 * x1_1234567 + beta2 * np.power(x2_12345678, 2) + np.random.normal(size=n)
    )  # Add quadratic term

    df = pd.DataFrame({"y_1234567a": y_1234567a, "x1_1234567": x1_1234567, "x2_12345678": x2_12345678})
    params = {
        "Intercept": beta0,
        "x1_1234567": beta1,
        "x2_12345678": beta2,
    }
    return df, pd.Series(params)


@pytest.fixture
def data_with_transformation_log_y() -> Tuple[pd.DataFrame, pd.Series]:
    # True coefficients for the linear model
    beta0 = 2.0
    beta1 = 3.0
    beta2 = -0.5

    # Randomly generated features
    np.random.seed(4534333)
    n = 10000
    x1_1234567, x2_12345678 = np.random.randn(2, n)
    # Generate the target variable with some random noise
    y_1234567a = np.exp(
        beta0 + beta1 * x1_1234567 + beta2 * x2_12345678 + np.random.normal(size=n)
    )  # inverse of log trafo

    df = pd.DataFrame({"y_1234567a": y_1234567a, "x1_1234567": x1_1234567, "x2_12345678": x2_12345678})
    params = {
        "Intercept": beta0,
        "x1_1234567": beta1,
        "x2_12345678": beta2,
    }
    return df, pd.Series(params)


@pytest.fixture
def data_with_categories() -> Tuple[pd.DataFrame, pd.Series]:
    # True coefficients for the linear model
    beta0 = 2.0
    beta1 = 3.0
    beta2 = -1.5
    beta_A = 1.0
    beta_B = 5.0
    beta_C = 10.0
    beta_D = 0.1
    beta_E = 100.0

    # Randomly generated features
    np.random.seed(9876789)
    n = 10000
    x1_1234567, x2_12345678 = np.random.randn(2, n)
    x3_1234567a = np.random.choice(["A", "B"], size=n)
    x4_12345678b = np.random.choice(["C", "D", "E"], size=n)
    # Generate the target variable with some random noise
    y_1234567a = (
        beta0
        + beta1 * x1_1234567
        + beta2 * x2_12345678
        + beta_A * (x3_1234567a == "A")
        + beta_B * (x3_1234567a == "B")
        + beta_C * (x4_12345678b == "C")
        + beta_D * (x4_12345678b == "D")
        + beta_E * (x4_12345678b == "E")
        + np.random.normal(size=n)
    )

    df = pd.DataFrame(
        {
            "y_1234567a": y_1234567a,
            "x1_1234567": x1_1234567,
            "x2_12345678": x2_12345678,
            "x3_1234567a": x3_1234567a,
            "x4_12345678b": x4_12345678b,
        }
    )
    params = {
        "Intercept": beta0,
        "x1_1234567": beta1,
        "x2_12345678": beta2,
        "x3_1234567a[T.A]": beta_A,
        "x3_1234567a[T.B]": beta_B,
        "x4_12345678b[T.C]": beta_C,
        "x4_12345678b[T.D]": beta_D,
        "x4_12345678b[T.E]": beta_E,
    }
    return df, pd.Series(params)


@pytest.fixture
def data_mixed() -> Tuple[pd.DataFrame, pd.Series]:
    params = {
        "Intercept": 2.0,
        "x3_1234567a[T.B]:x4_12345678b[T.D]": -3.0,
        "x1_1234567:x3_1234567a[T.B]": 3.0,
        "x2_12345678:x3_1234567a[T.B]:x4_12345678b[T.D]": 10.0,
        "x1_1234567:x2_12345678:x4_12345678b[T.E]": -50,
    }
    np.random.seed(9876789)
    n = 10000
    x1_1234567, x2_12345678 = np.random.randn(2, n)
    x3_1234567a = np.random.choice(["A", "B"], size=n)
    x4_12345678b = np.random.choice(["C", "D", "E"], size=n)

    y_1234567a = (
        params["Intercept"]
        + params["x3_1234567a[T.B]:x4_12345678b[T.D]"] * (x3_1234567a == "B") * (x4_12345678b == "D")
        + params["x1_1234567:x3_1234567a[T.B]"] * x1_1234567 * (x3_1234567a == "B")
        + params["x2_12345678:x3_1234567a[T.B]:x4_12345678b[T.D]"]
        * x2_12345678
        * (x3_1234567a == "B")
        * (x4_12345678b == "D")
        + params["x1_1234567:x2_12345678:x4_12345678b[T.E]"] * x1_1234567 * x2_12345678 * (x4_12345678b == "E")
        + np.random.normal(size=n)
    )

    df = pd.DataFrame(
        {
            "y_1234567a": y_1234567a,
            "x1_1234567": x1_1234567,
            "x2_12345678": x2_12345678,
            "x3_1234567a": x3_1234567a,
            "x4_12345678b": x4_12345678b,
        }
    )

    return df, params


def srs_setup_ols_fit(
    df: pd.DataFrame,
    formula: str,
) -> ConvenientOlsFit:
    df["session_id"] = range(1, len(df) + 1)
    return ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
        **TEST_TIMESTAMPS
    )


def srs_setup_no_timestamps_and_total_failure_rate(
    df: pd.DataFrame,
    formula: str,
) -> ConvenientOlsFit:
    df["session_id"] = range(1, len(df) + 1)
    return ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
    )


@pytest.fixture
def simple_setup(simple_data: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = simple_data
    formula = "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b"
    return srs_setup_ols_fit(df=df, formula=formula)


@pytest.fixture
def alternative_simple_setup(alternative_simple_data: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = alternative_simple_data
    formula = "y1_12345678b ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b"
    return srs_setup_ols_fit(df=df, formula=formula)


@pytest.fixture
def simple_setup_no_timestamps_and_total_failure_rate(simple_data: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = simple_data
    formula = "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b"
    return srs_setup_no_timestamps_and_total_failure_rate(df=df, formula=formula)


@pytest.fixture
def setup_transformation_quadratic_x(
    data_with_transformation_quadratic_x: Tuple[pd.DataFrame, pd.Series]
) -> ConvenientOlsFit:
    df, _ = data_with_transformation_quadratic_x
    formula = "y_1234567a ~ x1_1234567 + np.power(x2_12345678,2)"
    return srs_setup_ols_fit(df=df, formula=formula)


@pytest.fixture
def setup_transformation_log_y(data_with_transformation_log_y: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = data_with_transformation_log_y
    formula = "np.log(y_1234567a) ~ x1_1234567 + x2_12345678"
    df["session_id"] = range(1, len(df) + 1)
    return ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        inverse_output_trafo=np.exp,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
        **TEST_TIMESTAMPS
    )


@pytest.fixture
def setup_with_categories(data_with_categories: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = data_with_categories
    formula = "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b"
    return srs_setup_ols_fit(df=df, formula=formula)


@pytest.fixture
def setup_mixed(data_mixed: Tuple[pd.DataFrame, pd.Series]) -> ConvenientOlsFit:
    df, _ = data_mixed
    formula = "y_1234567a ~ x1_1234567*x3_1234567a + x2_12345678*x3_1234567a*x4_12345678b + x1_1234567*x2_12345678*x4_12345678b"
    return srs_setup_ols_fit(df=df, formula=formula)


@pytest.fixture
def simple_ml_data_no_shape_par() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    beta = (2.0, 3.0, -1.5)
    np.random.seed(9876789)
    n = 10000
    X = np.random.randn(n, 2)
    X_with_intercept = np.insert(arr=X, obj=0, values=1, axis=1)
    par = 0.1
    mean = st.gumbel_l(scale=par).mean()  # Does not have a shape parameter
    dist = st.gumbel_l(-mean, par)
    y_1234567a = np.dot(X_with_intercept, beta) + dist.rvs(n)
    df = pd.DataFrame({"y_1234567a": y_1234567a, "x1_1234567": X[:, 0], "x2_12345678": X[:, 1]})
    params = {"linear_params": beta, "noise_params": dist.args}
    return df, params


@pytest.fixture
def simple_ml_data_with_shape_par() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    beta = (2.0, 3.0, -1.5)
    np.random.seed(9876789)
    n = 10000
    X = np.random.randn(n, 2)
    X_with_intercept = np.insert(arr=X, obj=0, values=1, axis=1)
    par = (5.0, 0.1)
    mean = st.exponnorm(K=par[0], loc=0.0, scale=par[1]).mean()
    dist = st.exponnorm(par[0], -mean, par[1])  # Does have a shape parameter
    y_1234567a = np.dot(X_with_intercept, beta) + dist.rvs(n)
    df = pd.DataFrame({"y_1234567a": y_1234567a, "x1_1234567": X[:, 0], "x2_12345678": X[:, 1]})
    params = {"linear_params": beta, "noise_params": dist.args}
    return df, params


def srs_setup_mle_fit(df: pd.DataFrame, formula: str, noise_dist: st.rv_continuous) -> ConvenientMleFit:
    df["session_id"] = range(1, len(df) + 1)
    return ConvenientMleFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        noise_dist=noise_dist,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
        **TEST_TIMESTAMPS
    )


@pytest.fixture
def simple_ml_setup_no_shape_par(simple_ml_data_no_shape_par: Tuple[pd.DataFrame, pd.Series]) -> ConvenientMleFit:
    df, _ = simple_ml_data_no_shape_par
    return srs_setup_mle_fit(df=df, formula="y_1234567a ~ x1_1234567 + x2_12345678", noise_dist=st.gumbel_l)


@pytest.fixture
def simple_ml_setup_with_shape_par(simple_ml_data_with_shape_par: Tuple[pd.DataFrame, pd.Series]) -> ConvenientMleFit:
    df, _ = simple_ml_data_with_shape_par
    return srs_setup_mle_fit(df=df, formula="y_1234567a ~ x1_1234567 + x2_12345678", noise_dist=st.exponnorm)


@pytest.fixture
def mock_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_plt_and_fig_show(*args, **kwargs) -> None:
        pass

    monkeypatch.setattr(plt, "show", mock_plt_and_fig_show)
    monkeypatch.setattr(go.Figure, "show", mock_plt_and_fig_show)
