#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from tools.conftest import TEST_TIMESTAMPS
from tools.general_utility.helpers import NR_OF_DIGITS_AFTER_COMMA
from tools.regression_models.regression_models import ConvenientMleFit, GenericLinearLikelihoodModel
from tools.remove_invalid_sessions.remove_invalid_sessions import TotalFailureInfo
from tools.test_helpers.file_creation_helpers import assert_file_content


def test_simple_ml_fit_no_shape_par(
    simple_ml_data_no_shape_par: Tuple[pd.DataFrame, Dict[str, Any]], simple_ml_setup_no_shape_par: ConvenientMleFit
):
    _, expected_params = simple_ml_data_no_shape_par
    actual_params = simple_ml_setup_no_shape_par.result.params
    actual_params_noise = simple_ml_setup_no_shape_par.noise_dist.args
    assert np.allclose(actual_params, expected_params["linear_params"], rtol=0.01)
    assert np.allclose(actual_params_noise, expected_params["noise_params"], rtol=0.1)


def test_simple_ml_fit_with_shape_par(
    simple_ml_data_with_shape_par: Tuple[pd.DataFrame, Dict[str, Any]], simple_ml_setup_with_shape_par: ConvenientMleFit
):
    _, expected_params = simple_ml_data_with_shape_par
    actual_params = simple_ml_setup_with_shape_par.result.params
    actual_params_noise = simple_ml_setup_with_shape_par.noise_dist.args
    assert np.allclose(actual_params, expected_params["linear_params"], rtol=0.01)
    assert np.allclose(actual_params_noise, expected_params["noise_params"], rtol=0.1)


def test__simple_ml_summary__smoke(simple_ml_setup_no_shape_par: ConvenientMleFit):
    simple_ml_setup_no_shape_par.print_summary()


def test__all_simple_plots__smoke(simple_ml_setup_no_shape_par: ConvenientMleFit, mock_plots: None):
    simple_ml_setup_no_shape_par.plot_residuals_hist()
    simple_ml_setup_no_shape_par.plot_residuals()
    simple_ml_setup_no_shape_par.resid_qqplot()
    simple_ml_setup_no_shape_par.plot_partregress_grid()
    simple_ml_setup_no_shape_par.plot_ccpr_grid()


def test__plot_model_vs_original__smoke(simple_ml_setup_no_shape_par: ConvenientMleFit, mock_plots: None):
    simple_ml_setup_no_shape_par.plot_model_vs_original()


def test__fit_normalized__smoke(simple_ml_setup_no_shape_par: ConvenientMleFit, mock_plots: None):
    simple_ml_setup_no_shape_par.plot_standardized_coefficients()


@pytest.mark.parametrize(
    "formula, expected_exog",
    [
        ("y_1234567a ~ x1_1234567 * x2_12345678", ["Intercept", "x1_1234567", "x2_12345678"]),
        ("y_1234567a ~ x1_1234567", ["Intercept", "x1_1234567"]),
    ],
)
def test_backward_elimination(
    formula: str, expected_exog: List[str], simple_ml_data_no_shape_par: Tuple[pd.DataFrame, Dict[str, Any]]
):
    df, _ = simple_ml_data_no_shape_par
    fit = ConvenientMleFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        noise_dist=stats.gumbel_l,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )
    fit.backward_elimination()
    assert all(fit.result.pvalues.index.isin(expected_exog))


def read_file(file: str) -> Optional[Any]:
    file_path = file
    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while parsing JSON: {e}")


def assert_right_model_meta_parameter(expected_model: Dict[str, Any], actual_model: Dict[str, Any]) -> None:
    assert actual_model["name"] == expected_model["name"]
    assert actual_model["model_class"] == expected_model["model_class"]
    assert actual_model["dependent_variable"] == expected_model["dependent_variable"]


def test_export_simple_model(tmpdir, simple_ml_setup_no_shape_par: ConvenientMleFit):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    simple_ml_setup_no_shape_par.model_write_file = file_path
    simple_ml_setup_no_shape_par.export_model()
    expected_model = {
        **TEST_TIMESTAMPS,
        "name": "y_1234567a",
        "model_class": "<class 'tools.regression_models.regression_models.GenericLinearLikelihoodModel'>",
        "params": {
            "Intercept": pytest.approx(2.0, abs=0.1),
            "x1_1234567": pytest.approx(3.0, abs=0.1),
            "x2_12345678": pytest.approx(-1.5, abs=0.1),
        },
        "dependent_variable": "y_1234567a",
        "mse_residuals": 0.016588149,
        "custom_noise_distribution": {
            "distributions": [
                {
                    "name": "gumbel_l",
                    "parameters": [pytest.approx(0.05772156649015329, abs=1e-3), pytest.approx(0.1, abs=1e-3)],
                }
            ],
        },  # Params (loc and scale) take from data generation
        "total_failure_rate": 0.5,
    }
    assert_file_content(
        expected_content=expected_model,
        file_path=file_path,
        nr_of_digits_after_comma=NR_OF_DIGITS_AFTER_COMMA,
    )


def test__mean_and_scale_set_correctly_for_noise_model(simple_ml_setup_no_shape_par: ConvenientMleFit):
    mle_model = GenericLinearLikelihoodModel(
        endog=simple_ml_setup_no_shape_par.y,
        exog=simple_ml_setup_no_shape_par.X,
        noise_dist=simple_ml_setup_no_shape_par.noise_dist_family,
    )
    scale = 0.1
    noise_distr = mle_model.noise_zero_mean(arg=[], scale=scale)
    assert np.isclose(noise_distr.mean(), 0.0)
    assert np.isclose(noise_distr.args[1], scale)


def test__predict_as_expected(
    simple_ml_setup_no_shape_par: ConvenientMleFit, simple_ml_data_no_shape_par: Tuple[pd.DataFrame, Dict[str, Any]]
):
    mle_model = GenericLinearLikelihoodModel(
        endog=simple_ml_setup_no_shape_par.y,
        exog=simple_ml_setup_no_shape_par.X,
        noise_dist=simple_ml_setup_no_shape_par.noise_dist_family,
    )
    _, model_params = simple_ml_data_no_shape_par
    mle_model.exog = pd.DataFrame({"a": [1, 1, 3], "b": [1, 2, 2], "c": [1, 3, 1]})
    expected_predict = [3.5, 3.5, 10.5]
    actual_predict = mle_model.predict(
        params=model_params["linear_params"] + model_params["noise_params"], exog="unused_value"
    )
    assert np.array_equal(actual_predict, expected_predict)
