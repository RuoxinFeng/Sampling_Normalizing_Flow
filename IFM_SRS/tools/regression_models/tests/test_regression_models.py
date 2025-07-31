#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import copy
import json
import os
import re
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
from pandas.testing import assert_frame_equal
from scipy.stats import johnsonsu
from tools.conftest import TEST_TIMESTAMPS
from tools.regression_models.regression_models import (
    LINES_SEPARATOR,
    ConvenientGlmFit,
    ConvenientMleFit,
    ConvenientOlsFit,
    ConvenientRegressionFit,
    _create_log_dataframe,
    _get_best_model,
    backward_stepwise_selection,
    full_run,
    get_polynomial_of_parameters_and_order,
)
from tools.remove_invalid_sessions.remove_invalid_sessions import TotalFailureInfo
from tools.test_helpers.file_creation_helpers import assert_file_content

REGRESSION_MODELS_PATH = "tools.regression_models.regression_models"


@pytest.mark.parametrize("regression_models_class", [ConvenientOlsFit, ConvenientGlmFit, ConvenientMleFit])
@pytest.mark.parametrize(
    "formula_without_cb_id, error_msg",
    [
        ("y ~ x1_1234567 + x2_12345678", "Codebeamer ID missing in SPV y."),
        ("y_1234567a ~ x1_1234567 + x2", "Codebeamer ID missing in IFV x2"),
    ],
)
def test_regression_models_no_cb_id_in_formula_raise_error(
    formula_without_cb_id: str, error_msg: str, regression_models_class: Type[ConvenientRegressionFit]
):
    input_params = {
        "df": pd.DataFrame({"column1": [7, 9, 2], "column2": [3, 5, 8]}),
        "model_name": "xxx_12345678",
        "scenario_name": "SRS-XX",
        "formula": formula_without_cb_id,
        **TEST_TIMESTAMPS,
        "total_failure_info": TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    }
    if regression_models_class == ConvenientGlmFit:
        input_params["family"] = sm.genmod.families.family.Family
    if regression_models_class == ConvenientMleFit:
        input_params["noise_dist"] = st.rv_continuous
    with pytest.raises(ValueError, match=error_msg):
        regression_models_class(**input_params)


@pytest.mark.parametrize("regression_models_class", [ConvenientOlsFit, ConvenientGlmFit, ConvenientMleFit])
def test_regression_models_nan_values_raise_error(regression_models_class: Type[ConvenientRegressionFit]):
    nan_df = pd.DataFrame({"column1": [7, None, 9, 2], "column2": [3, 5, None, 8]})
    input_params = {
        "model_name": "xxx_12345678",
        "scenario_name": "SRS-XX",
        "formula": "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b",
        **TEST_TIMESTAMPS,
        "total_failure_info": TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    }
    input_params["df"] = nan_df
    if regression_models_class == ConvenientGlmFit:
        input_params["family"] = sm.genmod.families.family.Family
    if regression_models_class == ConvenientMleFit:
        input_params["noise_dist"] = st.rv_continuous
    with pytest.raises(ValueError, match="There are nan values in the data frame!"):
        regression_models_class(**input_params)


def test_simple_fit(simple_data: Tuple[pd.DataFrame, pd.Series], simple_setup: ConvenientOlsFit):
    _, expected_params = simple_data
    actual_params = simple_setup.result.params
    expected_y = simple_data[0]["y_1234567a"]
    actual_y = simple_setup.df["resid"] + simple_setup.df["y_predict"]
    assert np.allclose(actual_params, expected_params, atol=0.1)
    assert np.allclose(actual_y, expected_y)


def test_setup_transformation_quadratic_x(
    data_with_transformation_quadratic_x: Tuple[pd.DataFrame, pd.Series],
    setup_transformation_quadratic_x: ConvenientOlsFit,
):
    _, expected_params = data_with_transformation_quadratic_x
    actual_params = setup_transformation_quadratic_x.result.params
    expected_y = data_with_transformation_quadratic_x[0]["y_1234567a"]
    actual_y = setup_transformation_quadratic_x.df["resid"] + setup_transformation_quadratic_x.df["y_predict"]
    assert np.allclose(actual_params, expected_params, atol=0.1)
    assert np.allclose(actual_y, expected_y)


def test_setup_transformation_log_y(
    data_with_transformation_log_y: Tuple[pd.DataFrame, pd.Series], setup_transformation_log_y: ConvenientOlsFit
):
    _, expected_params = data_with_transformation_log_y
    actual_params = setup_transformation_log_y.result.params
    expected_y = np.log(data_with_transformation_log_y[0]["y_1234567a"])
    actual_y = setup_transformation_log_y.df["resid"] + setup_transformation_log_y.df["y_predict"]
    assert np.allclose(actual_params, expected_params, atol=0.1)
    assert np.allclose(actual_y, expected_y)


def test__correctly_detects_endog_part_of_formula(
    simple_setup: ConvenientOlsFit, setup_transformation_log_y: ConvenientOlsFit
):
    assert simple_setup._endog_part_of_formula == "y_1234567a"
    assert setup_transformation_log_y._endog_part_of_formula == "np.log(y_1234567a)"


def test_noise_dist_stays_after_drop_term_but_gets_fitted_again(simple_setup: ConvenientOlsFit):
    simple_setup.nongaussian_resid_fit(johnsonsu)
    assert simple_setup.noise_dist_name == "johnsonsu"

    args_before_drop = simple_setup.noise_dist.args
    simple_setup.drop_term("x4_12345678b")
    args_after_drop = simple_setup.noise_dist.args

    assert simple_setup.noise_dist_name == "johnsonsu"
    for after, before in zip(args_after_drop, args_before_drop):
        assert after != before


def test_nongaussian_resid_fit__warning_if_inf(simple_setup: ConvenientOlsFit):
    dist = MagicMock()
    dist.return_value.ppf.return_value = np.array([np.inf] + 99_998 * [1] + [np.inf])
    test_quantiles = np.linspace(1e-10, 1 - 1e-10, 100_000)

    with pytest.warns(
        Warning, match=f"Inf values occurred for the quantiles:\n{test_quantiles[0]}\n{test_quantiles[-1]}"
    ):
        simple_setup.nongaussian_resid_fit(dist)


def test_nongaussian_resid_fit__warning_if_nan(simple_setup: ConvenientOlsFit):
    dist = MagicMock()
    dist.return_value.ppf.return_value = np.array([np.nan] + 99_998 * [1] + [np.nan])
    test_quantiles = np.linspace(1e-10, 1 - 1e-10, 100_000)

    with pytest.warns(
        Warning, match=f"NaN values occurred for the quantiles:\n{test_quantiles[0]}\n{test_quantiles[-1]}"
    ):
        simple_setup.nongaussian_resid_fit(dist)


def test_nongaussian_resid_fit__check_print_if_ok(simple_setup: ConvenientOlsFit, capfd: pytest.CaptureFixture[str]):
    simple_setup.nongaussian_resid_fit(johnsonsu)
    noise_dist = simple_setup.noise_dist
    out, err = capfd.readouterr()
    assert (
        out
        == f"1e-10 percentile of the fitted residual distribution: {noise_dist.ppf(1e-10)}\n1-1e-10 percentile of the fitted residual distribution: {noise_dist.ppf(1-1e-10)}\n"
    )
    assert err == ""


def test_nongaussian_resid_fit__warning_if_raises_runtimeerror(simple_setup: ConvenientOlsFit):
    dist = MagicMock()
    dist.return_value.ppf.side_effect = RuntimeError("Failed to converge after 100 iterations.")
    with pytest.warns(
        Warning, match="Some example percentile threw the following error:\nFailed to converge after 100 iterations."
    ):
        simple_setup.nongaussian_resid_fit(dist)


def test_simple_fit_with_categories(
    data_with_categories: Tuple[pd.DataFrame, pd.Series], setup_with_categories: ConvenientOlsFit
):
    _, expected_params = data_with_categories
    actual_params = setup_with_categories.result.params
    assert actual_params["Intercept"] == pytest.approx(
        expected_params["Intercept"] + expected_params["x3_1234567a[T.A]"] + expected_params["x4_12345678b[T.C]"],
        rel=0.01,
    )
    assert actual_params["x1_1234567"] == pytest.approx(expected_params["x1_1234567"], rel=0.01)
    assert actual_params["x2_12345678"] == pytest.approx(expected_params["x2_12345678"], rel=0.01)
    assert actual_params["x3_1234567a[T.B]"] == pytest.approx(
        expected_params["x3_1234567a[T.B]"] - expected_params["x3_1234567a[T.A]"], rel=0.01
    )
    assert actual_params["x4_12345678b[T.D]"] == pytest.approx(
        expected_params["x4_12345678b[T.D]"] - expected_params["x4_12345678b[T.C]"], rel=0.01
    )
    assert actual_params["x4_12345678b[T.E]"] == pytest.approx(
        expected_params["x4_12345678b[T.E]"] - expected_params["x4_12345678b[T.C]"], rel=0.01
    )


def test_simple_fit_mixed(data_mixed: Tuple[pd.DataFrame, pd.Series], setup_mixed: ConvenientOlsFit):
    _, expected_params = data_mixed
    actual_params = setup_mixed.result.params
    for par_name, expected_value in expected_params.items():
        assert actual_params[par_name] == pytest.approx(expected_value, rel=0.1)


@patch.object(statsmodels.regression.linear_model.RegressionResults, "summary")
def test_summary_called(summary_mock: MagicMock, simple_setup: ConvenientOlsFit):
    simple_setup.print_summary()
    summary_mock.assert_called_once()


def test_print_summary(simple_setup: ConvenientOlsFit):
    simple_setup.print_summary()


def assert_data_is_standardized(data: pd.DataFrame, response_name: str) -> None:
    data = data.drop(response_name, axis=1).select_dtypes(include=[int, float])
    np.testing.assert_allclose(data.mean(axis=0), np.zeros(len(data.columns)), atol=1e-10)
    np.testing.assert_allclose(data.std(axis=0), np.ones(len(data.columns)), atol=1e-3)


def remove_category_from_string(string: str) -> str:
    return re.sub(r"\[.*?\]", "", string)


def assert_all_coefficients_contained(model: ConvenientRegressionFit, coefficients: Iterable) -> None:
    assert all(
        [
            remove_category_from_string(param) in coefficients
            for param in model.result.params.index.difference(["Intercept"])
        ]
    )


@patch(REGRESSION_MODELS_PATH + ".ConvenientOlsFit._return_fit", return_value=MagicMock())
def test_use_standardized_data_for_fit(mocked_fit: MagicMock, simple_setup: ConvenientOlsFit):
    mocked_fit.return_value = [MagicMock()] * 3
    simple_setup._get_fitted_standardized_data_coefficients_and_confidence_intervals()
    assert_data_is_standardized(data=mocked_fit.call_args.kwargs["data"], response_name=simple_setup.response_name)


def test_data_correctly_standardized_for_simple_setup(
    simple_setup: ConvenientOlsFit,
    setup_with_categories: ConvenientOlsFit,
    simple_ml_setup_with_shape_par: ConvenientMleFit,
):
    for model in [simple_setup, setup_with_categories, simple_ml_setup_with_shape_par]:
        data = model._standardized_data
        assert_data_is_standardized(data=data, response_name=model.response_name)


def test_all_coefficients_contained_in_standardized_data(
    simple_setup: ConvenientOlsFit,
    setup_with_categories: ConvenientOlsFit,
    simple_ml_setup_with_shape_par: ConvenientMleFit,
):
    for model in [simple_setup, setup_with_categories, simple_ml_setup_with_shape_par]:
        data = model._standardized_data
        assert_all_coefficients_contained(model=model, coefficients=data.columns)


def test_data_standardized_in_the_correct_way(
    simple_setup: ConvenientOlsFit,
    setup_with_categories: ConvenientOlsFit,
    simple_ml_setup_with_shape_par: ConvenientMleFit,
):
    for model in [simple_setup, setup_with_categories, simple_ml_setup_with_shape_par]:
        data = model._standardized_data.select_dtypes(include=[int, float])
        number_of_rows_to_check = 20
        cols_not_to_check = [model.response_name, "session_id"]
        model_df = model.df[data.columns]
        unstandardized_df = data * model_df.std(axis=0) + model_df.mean(axis=0)
        assert np.allclose(
            unstandardized_df.drop(cols_not_to_check, axis=1).head(number_of_rows_to_check),
            model_df.drop(cols_not_to_check, axis=1).head(number_of_rows_to_check),
            atol=1e-2,
        )


def test_fitting_standardized_data_outputs_expected_coefficients(
    simple_setup: ConvenientOlsFit,
    setup_with_categories: ConvenientOlsFit,
    simple_ml_setup_with_shape_par: ConvenientMleFit,
):
    for model in [simple_setup, setup_with_categories, simple_ml_setup_with_shape_par]:
        coefficients, _ = model._get_fitted_standardized_data_coefficients_and_confidence_intervals()
        assert_all_coefficients_contained(
            model, coefficients=[remove_category_from_string(c) for c in coefficients.index]
        )


def test_sanity_check_on_confidence_intervals(
    simple_setup: ConvenientOlsFit,
    setup_with_categories: ConvenientOlsFit,
    simple_ml_setup_with_shape_par: ConvenientMleFit,
):
    for model in [simple_setup, setup_with_categories, simple_ml_setup_with_shape_par]:
        (
            coefficients,
            confidence_intervals,
        ) = model._get_fitted_standardized_data_coefficients_and_confidence_intervals()
        for coefficient in coefficients.index:
            assert (
                confidence_intervals.T[coefficient][0]
                <= coefficients[coefficient]
                <= confidence_intervals.T[coefficient][1]
            )


def test__fit_standardized__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.plot_standardized_coefficients()


def test__print_p_sorted_terms_smoke(simple_setup: ConvenientOlsFit):
    simple_setup.print_p_sorted_terms()


def test_drop_term(simple_data: Tuple[pd.DataFrame, pd.Series], simple_setup: ConvenientOlsFit):
    resid_before_drop = simple_setup.df["resid"]
    y_predict_before_drop = simple_setup.df["y_predict"]
    simple_setup.drop_term("x2_12345678")
    resid_after_drop = simple_setup.df["resid"]
    y_predict_after_drop = simple_setup.df["y_predict"]
    expected_y = simple_data[0]["y_1234567a"]
    actual_y = resid_after_drop + y_predict_after_drop
    assert "x2_12345678" not in simple_setup.result.model.exog_names
    assert simple_setup.formula == "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b - x2_12345678"
    assert not np.allclose(resid_before_drop, resid_after_drop)
    assert not np.allclose(y_predict_before_drop, y_predict_after_drop)
    assert np.allclose(actual_y, expected_y)


def test_combine_categories(setup_with_categories: ConvenientOlsFit):
    setup_with_categories.combine_categories(
        column_name="x4_12345678b", categories=("C", "D"), combined_category="C or D"
    )
    assert setup_with_categories.df["x4_12345678b"].unique().tolist() == ["E", "C or D"]
    assert "x4_12345678b[T.C]" not in setup_with_categories.result.model.exog_names
    assert "x4_12345678b[T.D]" not in setup_with_categories.result.model.exog_names
    assert "x4_12345678b[T.E]" in setup_with_categories.result.model.exog_names


def test_print_outlier_session_ids(simple_setup: ConvenientOlsFit):
    simple_setup.print_outlier_session_ids()


def test__all_simple_plots__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.plot_residuals_hist()
    simple_setup.plot_residuals()
    simple_setup.resid_qqplot()
    simple_setup.plot_partregress_grid()
    simple_setup.plot_ccpr_grid()


def test__plot_model_vs_original__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.plot_model_vs_original()


def test__plot_model_vs_original_y_log__smoke(setup_transformation_log_y: ConvenientOlsFit, mock_plots: None):
    setup_transformation_log_y.plot_model_vs_original()


def test_plot_model_vs_original_raises_for_too_many_points(simple_setup: ConvenientOlsFit, mock_plots: None):
    with pytest.raises(ValueError, match="Can only plot a maximum of 1000 points. Requested number of points: 9999"):
        simple_setup.plot_model_vs_original(n=9999)


def read_file(file_path: str) -> Optional[Dict[str, Any]]:
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
    assert actual_model["timestamp_input_data"] == expected_model["timestamp_input_data"]
    assert actual_model["timestamp_export"] == expected_model["timestamp_export"]


@pytest.mark.parametrize("rounding_strategy", [{"nr_of_digits_after_comma": 4}, {"nr_of_digits_after_comma": None}, {}])
def test_export_simple_model(
    tmpdir, simple_setup: ConvenientOlsFit, capsys: pytest.CaptureFixture, rounding_strategy: Dict[str, Optional[int]]
):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    simple_setup.model_write_file = file_path
    simple_setup.export_model(**rounding_strategy)
    expected_model = {
        "name": "y_1234567a",
        "model_class": "<class 'statsmodels.regression.linear_model.OLS'>",
        "params": {
            "Intercept": pytest.approx(2.0, abs=0.1),
            "x1_1234567": pytest.approx(3.0, abs=0.1),
            "x2_12345678": pytest.approx(-1.5, abs=0.1),
            "x3_1234567a": pytest.approx(0.05, abs=0.1),
            "x4_12345678b": pytest.approx(0.0, abs=0.1),
        },
        "dependent_variable": "y_1234567a",
        "mse_residuals": pytest.approx(1.0, abs=1e-2),
        "custom_noise_distribution": {
            "distributions": [
                {"name": "norm", "parameters": [pytest.approx(0.0, abs=1e-3), pytest.approx(1.0, abs=1e-3)]}
            ],
        },
        "total_failure_rate": 0.5,
        **TEST_TIMESTAMPS,
    }

    assert_file_content(expected_content=expected_model, file_path=file_path, **rounding_strategy)

    captured = capsys.readouterr()
    assert f"Model written to {file_path} on 29/11/2023 10:07:37" in captured.out


@patch("tools.models_export.statsmodels_exporter.round_output", return_value={})
def test_export_model_no_rounding(mock_round_output: MagicMock, tmpdir, simple_setup: ConvenientOlsFit):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    simple_setup.model_write_file = file_path
    simple_setup.export_model(None)
    assert mock_round_output.call_count == 1
    assert mock_round_output.call_args_list[0].args[1] == None  # Called with None as nr_of_digits_after_comma


def test_export_transformation_quadratic_x(tmpdir, setup_transformation_quadratic_x: ConvenientOlsFit):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    setup_transformation_quadratic_x.model_write_file = file_path
    setup_transformation_quadratic_x.export_model()
    actual_model = read_file(file_path)
    np.testing.assert_almost_equal(actual_model["params"]["np.power(x2_12345678, 2)"], -0.50, decimal=2)


def test_export_model_with_categories(tmpdir, setup_with_categories: ConvenientOlsFit):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    setup_with_categories.model_write_file = file_path
    setup_with_categories.export_model()
    expected_model = {
        "name": "y_1234567a",
        "model_class": "<class 'statsmodels.regression.linear_model.OLS'>",
        "params": {
            "Intercept": pytest.approx(13.0, abs=0.1),
            "x1_1234567": pytest.approx(3.0, abs=0.1),
            "x2_12345678": pytest.approx(-1.5, abs=0.1),
            "x3_1234567a[T.B]": pytest.approx(4.0, abs=0.1),
            "x4_12345678b[T.D]": pytest.approx(-9.9, abs=0.1),
            "x4_12345678b[T.E]": pytest.approx(90, abs=0.1),
        },
        "dependent_variable": "y_1234567a",
        "mse_residuals": pytest.approx(1.0, abs=1e-2),
        "custom_noise_distribution": {
            "distributions": [
                {"name": "norm", "parameters": [pytest.approx(0.0, abs=0.2), pytest.approx(1.0, abs=1e-2)]}
            ]
        },
        "total_failure_rate": 0.5,
        **TEST_TIMESTAMPS,
    }
    assert_file_content(expected_content=expected_model, file_path=file_path)


def test_export_model_works_without_timestamp(
    tmpdir, simple_setup_no_timestamps_and_total_failure_rate: ConvenientOlsFit
):
    file_path = os.path.join(tmpdir, "xxx_12345678_model.json")
    simple_setup_no_timestamps_and_total_failure_rate.model_write_file = file_path
    simple_setup_no_timestamps_and_total_failure_rate.export_model()
    actual_model = read_file(file_path)
    expected_model = {
        "name": "y_1234567a",
        "model_class": "<class 'statsmodels.regression.linear_model.OLS'>",
        "params": {
            "Intercept": pytest.approx(2.0, abs=0.2),
            "x1_1234567": pytest.approx(3.0, abs=0.2),
            "x2_12345678": pytest.approx(-1.5, abs=0.2),
            "x3_1234567a": pytest.approx(0.05, abs=0.2),
            "x4_12345678b": pytest.approx(0.0, abs=0.2),
        },
        "dependent_variable": "y_1234567a",
        "mse_residuals": pytest.approx(1.0, abs=1e-2),
        "custom_noise_distribution": {
            "distributions": [{"name": "norm", "parameters": [pytest.approx(0.0, abs=1e-2), pytest.approx(1.0, 1e-2)]}],
        },
        "timestamp_input_data": None,
        "timestamp_export": None,
    }
    assert_file_content(expected_content=expected_model, file_path=file_path)


def test_export_tests(tmpdir, simple_setup: ConvenientOlsFit, capsys: pytest.CaptureFixture):
    file_path = os.path.join(tmpdir, "xxx_12345678_tests.json")
    simple_setup.tests_write_file = file_path
    input_data = {
        "x1_1234567": [0.0, 5.0],
        "x2_12345678": [1.0, -2.0],
        "x3_1234567a": [2.0, 3.0],
        "x4_12345678b": [3.0, -10.0],
    }
    simple_setup.export_tests(input_data=input_data, seed=34332)
    expected_tests = {
        "X_input": input_data,
        "U_noise": [0.5414476870643913, 0.12696189260539836],
        "Y_predict": [0.820310251763, 18.579337119273],
        **TEST_TIMESTAMPS,
    }
    assert_file_content(expected_content=expected_tests, file_path=file_path)

    captured = capsys.readouterr()
    assert f"Tests written to {file_path} on 29/11/2023 10:07:37" in captured.out


@pytest.mark.parametrize(
    ["timestamp_input_data", "timestamp_export"],
    [(None, TEST_TIMESTAMPS["timestamp_export"]), (TEST_TIMESTAMPS["timestamp_input_data"], None), (None, None)],
)
def test__raises_warning_if_timestamp_missing_during_tests_export_simple_setup_and_mixed_setup(
    tmpdir,
    simple_setup: ConvenientOlsFit,
    timestamp_input_data: Optional[str],
    timestamp_export: Optional[str],
):
    file_path = os.path.join(tmpdir, "xxx_12345678_tests.json")
    simple_setup.tests_write_file = file_path
    input_data = {
        "x1_1234567": [0.0, 5.0],
        "x2_12345678": [1.0, -2.0],
        "x3_1234567a": [2.0, 3.0],
        "x4_12345678b": [3.0, -10.0],
    }
    simple_setup.timestamp_input_data = timestamp_input_data
    simple_setup.timestamp_export = timestamp_export
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simple_setup.export_tests(input_data=input_data, seed=34332)
        simple_setup.export_tests_from_input_data()
        assert len(w) == 2
        assert issubclass(w[0].category, UserWarning)
        assert issubclass(w[1].category, UserWarning)
        assert "Either input data timestamp or export timestamp was not provided while exporting tests" == str(
            w[0].message
        )
        assert (
            "Either input data timestamp or export timestamp was not provided while exporting tests from input data"
            == str(w[1].message)
        )


def test_export_tests_one_point(simple_setup: ConvenientOlsFit):
    input_data = {"x1_1234567": [0.0], "x2_12345678": [1.0], "x3_1234567a": [2.0], "x4_12345678b": [3.0]}
    with pytest.raises(ValueError, match="Define at least two test vectors!"):
        simple_setup.export_tests(input_data=input_data, seed=34332)


@pytest.mark.parametrize(
    "term, variable",
    [
        ("x1_1234567", "x1_1234567"),
        ("x1_1234567[a.b]", "x1_1234567"),
        (
            "x1_1234567[c.d]:x2_12345678[a.b]+x1_1234567[c.d]+x2_12345678[a.b]",
            "x1_1234567:x2_12345678+x1_1234567+x2_12345678",
        ),
    ],
)
def test__category_removed_from_string(term: str, variable: str):
    assert variable == ConvenientRegressionFit._ConvenientRegressionFit__remove_category_from_string(term)


def test__convenient_regression_fit__predictors(simple_setup: ConvenientOlsFit):
    assert simple_setup.predictors == ["x1_1234567", "x2_12345678", "x3_1234567a", "x4_12345678b"]


@pytest.mark.parametrize(
    "X, expected_h",
    [
        (
            pd.DataFrame({"Intercept": [1, 1, 1, 1], "x1": [1, 2, 0, 4], "x2": [5, 6, 7, 0]}),
            np.array([[0.42528735632], [0.99425287356], [0.63218390804], [0.94827586206]]),
        ),
        (
            pd.DataFrame({"Intercept": [1, 1, 1, 1, 1], "x1": [0, 1, 0, 1, 1], "x2": [1.2, 3.4, 0.1, 0.01, 4]}),
            np.array([[0.53067204737], [0.42103004001], [0.53067204737], [0.94693651308], [0.57068935215]]),
        ),
    ],
)
def test__convenient_regression_fit__leverage(X: pd.DataFrame, expected_h: np.array, simple_setup: ConvenientOlsFit):
    simple_setup.X = X

    np.testing.assert_almost_equal(expected_h, simple_setup.leverage)


@pytest.mark.parametrize(
    "X, y, df, expected_loocv",
    [
        (
            pd.DataFrame({"Intercept": [1, 1, 1, 1], "x1": [1, 2, 0, 4], "x2": [5, 6, 7, 0]}),
            pd.DataFrame({"y": [1, 2, 3, 4]}),
            pd.DataFrame({"y_predict": [1.1, 2.2, 3.3, 4.4]}),
            317.8849908,
        ),
        (
            pd.DataFrame({"Intercept": [1, 1, 1, 1, 1], "x1": [0, 1, 0, 1, 1], "x2": [1.2, 3.4, 0.1, 0.01, 4]}),
            pd.DataFrame({"y": [-1, 271, 42, 314, 404]}),
            pd.DataFrame({"y_predict": [1, 300, 35, 315, 400]}),
            638.2954236,
        ),
    ],
)
def test__convenient_regression_fit__loocv(
    X: pd.DataFrame, y: pd.DataFrame, df: pd.DataFrame, expected_loocv: float, simple_setup: ConvenientOlsFit
):
    simple_setup.X = X
    simple_setup.y = y
    simple_setup.df = df

    assert simple_setup.loocv == pytest.approx(expected_loocv)


@pytest.mark.parametrize(
    "df, formula, linear_combinations",
    [
        (
            pd.DataFrame(
                {
                    "y_1234567a": [1, 1, 1, 1],
                    "x1_1234567": [1, 2, 0, 4],
                    "x2_12345678": [2, 4, 0, 8],
                    "x3_1234567a": [3, 5, 1, 2],
                }
            ),
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a",
            ["-0.894*x1_1234567 + 0.447*x2_12345678 = 0"],
        ),
        (
            pd.DataFrame(
                {
                    "y_1234567a": [1, 1, 1, 1, 1],
                    "x1_1234567": [1, 2, 0, 4, 3],
                    "x2_12345678": [2, 4, 0, 8, 6],
                    "x3_1234567a": [3, 5, 9, 2, 6],
                    "x4_1234567a": [6, 10, 18, 4, 12],
                }
            ),
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_1234567a",
            ["-0.894*x1_1234567 + 0.447*x2_12345678 = 0", "0.894*x3_1234567a - 0.447*x4_1234567a = 0"],
        ),
    ],
)
def test_backward_stepwise_selection_raises_correctly_for_combinations_of_linearly_dependent_predictors(
    df, formula, linear_combinations
):
    model = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Some of the predictors are linearly dependent, which is not allowed in OLS regression. See:\n {LINES_SEPARATOR.join(linear_combinations)}"
        ),
    ):
        backward_stepwise_selection(model)


@pytest.mark.parametrize(
    "term1, term2, is_subterm",
    [
        ("x1_1234567", "x1_1234567", True),
        ("x1_1234567[abc]:x2_12345678", "x1_1234567[def]:x2_12345678:x3_1234567a", True),
        ("x1_1234567[abc]:x2_12345678", "x1_1234567[def]:x3_1234567a", False),
    ],
)
def test__single_subterms_are_detected_correctly(term1: str, term2: str, is_subterm: bool):
    assert is_subterm == ConvenientRegressionFit._ConvenientRegressionFit__is_subterm(term1, term2)


@pytest.mark.parametrize(
    "formula, inverse_output_trafo",
    [
        ("np.log(y_1234567a) ~ x1_1234567 + x2_12345678", np.exp),
        ("np.sqrt(y_1234567a) ~ x1_1234567 + x2_12345678", np.square),
        ("np.sqrt(y_1234567a) ~ x1_1234567 + x2_12345678", lambda x: np.power(x, 2)),
        ("y_1234567a ~ x1_1234567 + x2_12345678", None),
    ],
)
def test__convenient_regression_fit__parse_output_trafo(
    simple_data: Tuple[pd.DataFrame, pd.Series], formula: str, inverse_output_trafo: Callable
):
    unit = ConvenientOlsFit(
        model_name="test",
        scenario_name="test",
        df=simple_data[0],
        formula=formula,
        timestamp_input_data=None,
        timestamp_export=None,
        total_failure_info=None,
        inverse_output_trafo=inverse_output_trafo,
    )

    assert unit.inverse_output_trafo == inverse_output_trafo
    assert unit.response_name == "y_1234567a"


@pytest.mark.parametrize(
    "formula, inverse_output_trafo, error_msg",
    [
        ("log(y_1234567a) ~ x1_1234567 + x2_12345678", np.exp, "Only numpy functions are supported"),
        ("np.log123(y_1234567a) ~ x1_1234567 + x2_12345678", np.exp, "No log123 found in numpy"),
        (
            "np.log(y_1234567a) ~ x1_1234567 + x2_12345678",
            np.log,
            f"The given function {np.log} is not the inverse of the transformation function {np.log}",
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678",
            np.exp,
            "Invalid input format for working with an inverse output trafo. Valid format np.log(var_name)",
        ),
    ],
)
def test__convenient_regression_fit__parse_output_trafo_raises(
    simple_data: Tuple[pd.DataFrame, pd.Series], formula: str, inverse_output_trafo: Callable, error_msg: str
):
    with pytest.raises(ValueError, match=re.escape(error_msg)):
        ConvenientOlsFit(
            model_name="test",
            scenario_name="test",
            df=simple_data[0],
            formula=formula,
            timestamp_input_data=None,
            timestamp_export=None,
            total_failure_info=None,
            inverse_output_trafo=inverse_output_trafo,
        )


@pytest.mark.parametrize(
    "predictor, predictors, expected_result",
    (
        ["x1", ["x1", "x2", "x3"], False],
        ["x1", ["x1", "x1:x2", "x3"], True],
        ["x1:x2", ["x1", "x2", "x3", "x1:x2", "x1:x2:x2"], True],
        ["x1:x2", ["x1", "x2", "x3", "x1:x2", "x1:x2:x3"], True],
        ["x1:x2", ["x1", "x2", "x3", "x1:x2", "x1:x3:x2"], True],
        ["x1", ["x1", "x1:x2", "x3", "np.power(x1,2)"], True],
        ["x1", ["x1", "x3", "np.power(x1,2)"], False],
        ["np.log(x1)", ["x1", "x1:x2", "x3", "np.log(x1)"], False],
    ),
)
def test__convenient_regression_fit__is_predictor_hierarchically_dependent(
    predictor: str, predictors: List[str], expected_result: bool
):
    assert ConvenientRegressionFit.is_predictor_hierarchically_dependent(predictor, predictors) == expected_result


def test__convenient_regression_fit__is_predictor_hierarchically_dependent_doesnt_change_predictors(
    simple_setup: ConvenientRegressionFit,
):
    predictors_before = copy.copy(simple_setup.predictors)
    ConvenientRegressionFit.is_predictor_hierarchically_dependent("x1_1234567", simple_setup.predictors)
    assert predictors_before == simple_setup.predictors


@pytest.mark.parametrize(
    "term, val, other_terms, is_subterm_with_lower_p_value",
    [
        ("x1_1234567", 0.1, {"x1_1234567": 0.1}, False),
        ("x1_1234567", 0.1, {"x1_1234567": 0.1, "x1_1234567:x2_12345678": 0.2, "x2_12345678": 0.01}, False),
        ("x1_1234567", 0.2, {"x1_1234567": 0.2, "x1_1234567:x2_12345678": 0.1, "x2_12345678": 0.2}, True),
    ],
)
def test__finds_subterm_with_lower_pvalue(
    term: str, val: float, other_terms: Dict[str, float], is_subterm_with_lower_p_value: bool
):
    assert (
        is_subterm_with_lower_p_value
        == ConvenientRegressionFit._ConvenientRegressionFit__is_subterm_of_other_terms_with_lower_value(
            term, val, other_terms
        )
    )


def test_least_significant_dummy_result():
    pvalues = pd.Series(
        {
            "x1_1234567": 0.01,
            "x2_12345678": 0.03,
            "x4_12345678b": 0.04,
            "x2_12345678:x4_12345678b": 0.05,
            "x3_1234567a:x4_12345678b": 0.07,  # must be removed first
            "x4_12345678b": 0.1,
        }
    )  # must not be removed first, since there is 'x3:x4' term with lower p-value
    actual_term, actual_p_value = ConvenientOlsFit.least_significant(pvalues)
    assert actual_term == "x3_1234567a:x4_12345678b"
    assert actual_p_value == 0.07


def test_least_significant_dummy_result_with_categories():
    # Notice it is good practice to remove categorical factors entirely, not just for single levels.
    # Cf. https://stats.stackexchange.com/questions/6050/how-should-i-handle-categorical-variables-with-multiple-levels-when-doing-backward-elimination
    # In other words: Ignore category infos in []
    pvalues = pd.Series(
        {
            "x1_1234567": 0.01,
            "x2_12345678[T.A]": 0.03,
            "x2_12345678[T.A]:x4_12345678b[T.B]": 0.05,
            "x3_1234567a:x4_12345678b[T.B]": 0.07,  # must be removed first
            "x4_12345678b[T.B]": 0.1,  # cannot be removed because of the interaction term with p=0.05
        }
    )  # must not be removed first, since there is 'x3_1234567a:x4_12345678b' term with lower p-value
    actual_term, actual_p_value = ConvenientOlsFit.least_significant(pvalues)
    assert actual_term == "x3_1234567a:x4_12345678b"
    assert actual_p_value == 0.07


def test_least_significant_dummy_result_with_categories_difference_only_in_category():
    pvalues = pd.Series(
        {
            "x1_1234567": 0.01,
            "x2_12345678": 0.03,
            "x4_12345678b[T.B]": 0.04,
            "x2_12345678:x4_12345678b[T.B]": 0.05,
            "x3_1234567a:x4_12345678b[T.B]": 0.07,  # must be removed first
            "x2_12345678:x4_12345678b[T.C]": 0.1,  # must not be removed because of x2_12345678:x4_12345678b[T.B]
        }
    )  # must not be removed first, since there is 'x3_1234567a:x4_12345678b' term with lower p-value
    actual_term, actual_p_value = ConvenientOlsFit.least_significant(pvalues)
    assert actual_term == "x3_1234567a:x4_12345678b"
    assert actual_p_value == 0.07


def test_least_significant_dummy_result_with_higher_order_terms():
    pvalues = pd.Series(
        {
            "x1_1234567": 0.0,
            "x2_12345678": 0.0,
            "x3_1234567a": 0.0,
            "x2_12345678:x4_12345678b": 0.0,
            "x4_12345678b:x4_12345678b": 0.7,  # must not be removed first
            "x4_12345678b:x3_1234567a:x4_12345678b": 0.5,
        }
    )  # since this term contains x4:x4
    actual_term, actual_p_value = ConvenientOlsFit.least_significant(pvalues)
    assert actual_term == "x4_12345678b:x3_1234567a:x4_12345678b"
    assert actual_p_value == 0.5


def test_least_significant_dummy_result_wrong_order():
    pvalues = pd.Series(
        {
            "cut_in_velocity_lateral_7914911": 0.003,
            "cut_in_velocity_lateral_7914911:cut_in_direction_9770634": 0.006,
            "cut_in_distance_10601062:cut_in_velocity_lateral_7914911:cut_in_direction_9770634": 0.110,
            "cut_in_distance_10601062:cut_in_velocity_lateral_7914911": 0.113,
            "cut_in_direction_9770634": 0.124,
            "cut_in_relative_velocity_9770614:cut_in_velocity_lateral_7914911:cut_in_direction_9770634": 0.189,  # dont remove this since it is a sub-factor
            "cut_in_relative_velocity_9770614:cut_in_velocity_lateral_7914911": 0.223,
            "cut_in_distance_10601062:cut_in_direction_9770634": 0.257,
            "cut_in_distance_10601062": 0.282,
            "cut_in_relative_velocity_9770614:cut_in_distance_10601062:cut_in_velocity_lateral_7914911": 0.343,
            "cut_in_relative_velocity_9770614:cut_in_distance_10601062:cut_in_velocity_lateral_7914911:cut_in_direction_9770634": 0.402,  # ... of this term (not on string but on element level)
            "cut_in_relative_velocity_9770614:cut_in_distance_10601062:cut_in_direction_9770634": 0.460,
            "cut_in_relative_velocity_9770614:cut_in_direction_9770634": 0.587,
            "cut_in_relative_velocity_9770614:cut_in_distance_10601062": 0.761,
            "cut_in_relative_velocity_9770614": 0.961,
        }
    )
    actual_term, actual_p_value = ConvenientOlsFit.least_significant(pvalues)
    assert (
        actual_term
        == "cut_in_relative_velocity_9770614:cut_in_distance_10601062:cut_in_velocity_lateral_7914911:cut_in_direction_9770634"
    )
    assert actual_p_value == 0.402


def test_least_significant_with_real_problem(simple_data: Tuple[pd.DataFrame, pd.Series]):
    df, _ = simple_data
    formula = "y_1234567a ~ x1_1234567 + x2_12345678*x4_12345678b + x3_1234567a"
    fit = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )

    least_significant, _ = ConvenientOlsFit.least_significant(p_values=fit.result.pvalues)
    assert least_significant == "x2_12345678:x4_12345678b"


def test_least_significant_with_no_removable_p_values():
    assert (None, -np.inf) == ConvenientOlsFit.least_significant(p_values={})


@pytest.mark.parametrize(
    "formula, expected_exog",
    [
        (
            "y_1234567a ~ x1_1234567 + x2_12345678*x4_12345678b + x1_1234567*x3_1234567a * x1_1234567*x2_12345678",
            ["Intercept", "x1_1234567", "x2_12345678"],
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b",
            ["Intercept", "x1_1234567", "x2_12345678"],
        ),
        ("y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a", ["Intercept", "x1_1234567", "x2_12345678"]),
    ],
)
def test_backward_elimination(formula: str, expected_exog: List[str], simple_data: Tuple[pd.DataFrame, pd.Series]):
    df, _ = simple_data
    fit = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )
    fit.backward_elimination()
    assert sorted(list(fit.result.pvalues.index)) == sorted(expected_exog)


@pytest.mark.parametrize(
    "df, formula, linear_combinations",
    [
        (
            pd.DataFrame(
                {
                    "y_1234567a": [1, 1, 1, 1],
                    "x1_1234567": [1, 2, 0, 4],
                    "x2_12345678": [2, 4, 0, 8],
                    "x3_1234567a": [3, 5, 1, 2],
                }
            ),
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a",
            ["-0.894*x1_1234567 + 0.447*x2_12345678 = 0"],
        ),
        (
            pd.DataFrame(
                {
                    "y_1234567a": [1, 1, 1, 1, 1],
                    "x1_1234567": [1, 2, 0, 4, 3],
                    "x2_12345678": [2, 4, 0, 8, 6],
                    "x3_1234567a": [3, 5, 9, 2, 6],
                    "x4_1234567a": [6, 10, 18, 4, 12],
                }
            ),
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_1234567a",
            ["-0.894*x1_1234567 + 0.447*x2_12345678 = 0", "0.894*x3_1234567a - 0.447*x4_1234567a = 0"],
        ),
    ],
)
def test_backward_elimination_warns_correctly_for_combinations_of_linearly_dependent_predictors(
    df, formula, linear_combinations
):
    model = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
    )
    with pytest.warns(
        match=re.escape(
            f"Some of the predictors are linearly dependent, which is not allowed in OLS regression. See:\n {LINES_SEPARATOR.join(linear_combinations)}"
        ),
    ):
        model.backward_elimination()


@pytest.mark.parametrize(
    "formula, expected_exog",
    [
        (
            "y_1234567a ~ x1_1234567 + x2_12345678*x4_12345678b + x1_1234567*x3_1234567a * x1_1234567*x2_12345678",
            ["Intercept", "x3_1234567a[T.B]", "x4_12345678b[T.D]", "x4_12345678b[T.E]", "x1_1234567", "x2_12345678"],
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b",
            ["Intercept", "x3_1234567a[T.B]", "x4_12345678b[T.D]", "x4_12345678b[T.E]", "x1_1234567", "x2_12345678"],
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a",
            ["Intercept", "x3_1234567a[T.B]", "x1_1234567", "x2_12345678"],
        ),
    ],
)
def test_backward_elimination_with_categories(
    formula: str, expected_exog: List[str], data_with_categories: Tuple[pd.DataFrame, pd.Series]
):
    df, _ = data_with_categories
    fit = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )
    fit.backward_elimination()
    assert sorted(list(fit.result.pvalues.index)) == sorted(expected_exog)


@pytest.mark.parametrize(
    ["model_formula", "expected_regressors_list", "expected_dependent_variable"],
    [
        ("y_1234567a~x1_1234567+x2_12345678", ["x1_1234567", "x2_12345678"], ["y_1234567a"]),
        (
            "y_1234567a~x1_1234567*x2_12345678 + x3_1234567a",
            ["x1_1234567", "x2_12345678", "x3_1234567a"],
            ["y_1234567a"],
        ),
    ],
)
def test__get_params_and_kpi_for_custom_regression(
    model_formula: str,
    expected_regressors_list: List[str],
    expected_dependent_variable: str,
    simple_setup: ConvenientOlsFit,
):
    simple_setup.formula = model_formula
    actual_dependent_variable, actual_regressors_list = simple_setup.factor_names_from_formula()
    assert actual_dependent_variable == expected_dependent_variable
    assert expected_regressors_list == actual_regressors_list


@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit._ConvenientRegressionFit__create_file")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit._ConvenientRegressionFit__create_folder")
def test__export_tests_from_input_data(mock_folder: MagicMock, mock_file: MagicMock, setup_mixed: ConvenientOlsFit):
    setup_mixed.export_tests_from_input_data()
    expected_tests = {
        "X_input": {
            "x1_1234567": [-0.8405158941408978, 1.597495258395815, 0.4655420409568447],
            "x2_12345678": [-1.6321500873005803, -0.9464574094148719, -0.4745570237174968],
            "x3_1234567a": ["B", "A", "B"],
            "x4_12345678b": ["E", "C", "D"],
        },
        "U_noise": [0.1915194503788923, 0.6221087710398319, 0.4377277390071145],
        "Y_predict": [-69.99190316478969, 2.3874544960504034, -4.541143927086224],
    }
    mock_file.assert_called_once()
    mock_folder.assert_called_once()
    actual_tests = mock_file.call_args[0][0]
    N = 3  # Only tests the content of the first 3 test vectors
    for key in ["x1_1234567", "x2_12345678"]:
        np.testing.assert_allclose(expected_tests["X_input"][key], actual_tests["X_input"][key][:N], rtol=1e-5)
    for key in ["x3_1234567a", "x4_12345678b"]:
        assert expected_tests["X_input"][key] == actual_tests["X_input"][key][:N]
    for key in ["U_noise", "Y_predict"]:
        np.testing.assert_allclose(expected_tests[key], actual_tests[key][:N], rtol=1e-5)


def test__export_tests_with_substitution(tmpdir, setup_transformation_quadratic_x: ConvenientOlsFit):
    file_path = os.path.join(tmpdir, "xxx_12345678_tests.json")
    setup_transformation_quadratic_x.tests_write_file = file_path

    with warnings.catch_warnings(record=True) as w:
        setup_transformation_quadratic_x.export_tests_from_input_data(
            substitute_columns=(
                {"np.power(x2_12345678, 2)": "x2_12345678", "unknown_column": "new_name_of_unknown_column"}
            )
        )
        assert f"unknown_column was not found in the regression formula!" == str(w[0].message)
    # This output is just copy and pasted from the actual file to detect regressions.
    expected_tests = {
        "X_input": {
            "x1_1234567": [-0.8405158941408978, 1.597495258395815, 0.4655420409568447],
            "x2_12345678": [-1.6321500873005803, -0.9464574094148719, -0.4745570237174968],
        },
        "U_noise": [0.1915194503788923, 0.6221087710398319, 0.4377277390071145],
        "Y_predict": [-2.7218811833432843, 6.623406839026341, 3.1205011697463143],
        **TEST_TIMESTAMPS,
    }
    assert_file_content(expected_content=expected_tests, file_path=file_path)


def test__raises_when_using_complex_transformations_without_substitutions(setup_mixed: ConvenientOlsFit):
    setup_mixed.formula = "y~np.power(x2_12345678, 2)"
    with pytest.raises(
        KeyError, match=re.escape("None of [Index(['np.power(x2_12345678, 2)'], dtype='object')] are in the [columns]")
    ):
        setup_mixed.export_tests_from_input_data()


def test_export_tests_from_input_data_and_create_file(
    tmpdir, setup_mixed: ConvenientOlsFit, capsys: pytest.CaptureFixture
):
    file_path = os.path.join(tmpdir, "xxx_12345678_tests.json")
    setup_mixed.tests_write_file = file_path
    setup_mixed.export_tests_from_input_data()
    actual_tests = read_file(file_path)

    # This output is just copy and pasted from the actual file to detect regressions.
    expected_tests = {
        "X_input": {
            "x1_1234567": [-0.8405158941408978, 1.597495258395815, 0.4655420409568447],
            "x2_12345678": [-1.6321500873005803, -0.9464574094148719, -0.4745570237174968],
            "x3_1234567a": ["B", "A", "B"],
            "x4_12345678b": ["E", "C", "D"],
        },
        "U_noise": [0.1915194503788923, 0.6221087710398319, 0.4377277390071145],
        "Y_predict": [-69.99190316478969, 2.3874544960504034, -4.541143927086224],
    }

    expected_number_of_tests = 19
    for key in ["x1_1234567", "x2_12345678", "x3_1234567a", "x4_12345678b"]:
        assert len(actual_tests["X_input"][key]) == expected_number_of_tests
    for key in ["U_noise", "Y_predict"]:
        assert len(actual_tests[key]) == expected_number_of_tests

    N = 3  # Only tests the content of the first 3 test vectors
    for key in ["x1_1234567", "x2_12345678"]:
        np.testing.assert_allclose(expected_tests["X_input"][key], actual_tests["X_input"][key][:N], rtol=1e-5)
    for key in ["x3_1234567a", "x4_12345678b"]:
        assert expected_tests["X_input"][key] == actual_tests["X_input"][key][:N]
    for key in ["U_noise", "Y_predict"]:
        np.testing.assert_allclose(expected_tests[key], actual_tests[key][:N], rtol=1e-5)

    captured = capsys.readouterr()
    assert f"Tests written to {file_path} on 29/11/2023 10:07:37" in captured.out


def test_export_tests_from_input_data_transformation_quadratic_x(
    tmpdir, setup_transformation_quadratic_x: ConvenientOlsFit, capsys: pytest.CaptureFixture
):
    file_path = os.path.join(tmpdir, "xxx_12345678_tests.json")
    setup_transformation_quadratic_x.tests_write_file = file_path
    setup = setup_transformation_quadratic_x
    setup.export_tests_from_input_data(substitute_columns=({"np.power(x2_12345678, 2)": "x2_12345678"}))

    # This output is just copy and pasted from the actual file to detect regressions.
    expected_tests = {
        "X_input": {
            "x1_1234567": [-0.8405158941408978, 1.597495258395815, 0.4655420409568447],
            "x2_12345678": [-1.6321500873005803, -0.9464574094148719, -0.4745570237174968],
        },
        "U_noise": [0.1915194503788923, 0.6221087710398319, 0.4377277390071145],
        "Y_predict": [-2.7218811833432843, 6.623406839026341, 3.1205011697463143],
        **TEST_TIMESTAMPS,
    }

    assert_file_content(expected_content=expected_tests, file_path=file_path)

    captured = capsys.readouterr()
    assert f"Tests written to {file_path} on 29/11/2023 10:07:37" in captured.out


def test__print_model_vs_original_histogram_directly__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.plot_model_vs_original_histogram()


def test__print_model_vs_original_histogram_via_summary__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.print_model_summary()


@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.plot_model_vs_original_histogram")
@patch(REGRESSION_MODELS_PATH + ".display_dataframe")
def test_correct_info_is_printed_when_printing_model_summary(
    display_dataframe_mock: MagicMock, _: MagicMock, simple_setup: ConvenientOlsFit
):
    simple_setup.print_model_summary()
    df_pval_exp = pd.DataFrame(
        {
            "Exog name": ["Intercept", "x1_1234567", "x2_12345678", "x3_1234567a", "x4_12345678b"],
            "p-value": [0, 0, 2.546552e-276, 7.67307e-02, 1.082033e-01],
            "coefficient": [1.994070, 2.992030, -1.546965, 0.058815, 0.050524],
        }
    )
    df_gen_inf_exp = pd.DataFrame(
        {
            "Property": [
                "Number of observations",
                "R-Squared",
                "R-Squared adj.",
                "Noise distribution",
                "Posterior total failure probability with uniform prior",
                "Observed total failures",
            ],
            "Value": [1000.0, 0.922679, 0.922368, "norm", 0.5, 10],
        }
    )

    assert_frame_equal(display_dataframe_mock.call_args_list[0][0][0], df_gen_inf_exp, check_exact=False, atol=1e-5)
    assert_frame_equal(display_dataframe_mock.call_args_list[1][0][0], df_pval_exp, check_exact=False, atol=1e-5)


@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.plot_model_vs_original_histogram")
@patch(REGRESSION_MODELS_PATH + ".display_markdown")
def test_correct_inverse_transformation_printed_when_printing_model_summary(
    display_markdown_mock: MagicMock, _: MagicMock, setup_transformation_log_y: ConvenientOlsFit
):
    setup_transformation_log_y.print_model_summary()
    expected_info = "This model predicts np.log(y_1234567a). To get the value for y_1234567a, the output needs to be transformed with np.exp()."
    assert expected_info in display_markdown_mock.call_args_list[2][0]


def perform_full_run_using_model_and_basic_asserts(
    data: pd.DataFrame, total_failure_info: TotalFailureInfo = TotalFailureInfo(0.1, 1), **kwargs
) -> ConvenientOlsFit:
    full_run_test_inputs = {
        "df": data,
        "formula": "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b",
        "model_name": "xxx_12345678",
        "scenario_name": "SRS-XX",
        "noise_dist": st.norm,
        "total_failure_info": TotalFailureInfo(0.1, 1),
        **TEST_TIMESTAMPS,
    }
    fit = full_run(
        **full_run_test_inputs,
        **kwargs,
    )
    assert fit.model_name == full_run_test_inputs["model_name"]
    assert fit.observed_total_failures == full_run_test_inputs["total_failure_info"].observed_total_failures
    assert (
        fit.posterior_total_failure_probability
        == full_run_test_inputs["total_failure_info"].posterior_total_failure_probability
    )
    assert fit.noise_dist_name == full_run_test_inputs["noise_dist"].name
    assert fit.timestamp_export == TEST_TIMESTAMPS["timestamp_export"]
    assert fit.timestamp_input_data == TEST_TIMESTAMPS["timestamp_input_data"]
    assert (
        fit.formula == full_run_test_inputs["formula"]
        if not kwargs.get("drop_terms")
        else fit.formula != full_run_test_inputs["formula"]
    )
    return fit


@patch(REGRESSION_MODELS_PATH + ".backward_stepwise_selection")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.print_summary")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.plot_residuals")
@patch(REGRESSION_MODELS_PATH + ".ConvenientOlsFit.nongaussian_resid_fit")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.resid_qqplot")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.plot_partregress_grid")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.plot_ccpr_grid")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.print_model_summary")
@patch(REGRESSION_MODELS_PATH + ".ConvenientRegressionFit.print_outlier_session_ids")
@patch(REGRESSION_MODELS_PATH + ".find_best_fit")
def test_full_run_calls_expected_functions(
    best_fit_mock: MagicMock,
    outlier_session_ids_mock: MagicMock,
    model_summary_mock: MagicMock,
    ccpr_grid_mock: MagicMock,
    partregress_grid_mock: MagicMock,
    qqplot_mock: MagicMock,
    resid_fit_mock: MagicMock,
    residual_plot_mock: MagicMock,
    regression_summary_mock: MagicMock,
    backward_stepwise_selection_mock: MagicMock,
    simple_data: ConvenientOlsFit,
):
    backward_stepwise_selection_mock.side_effect = lambda model: model
    perform_full_run_using_model_and_basic_asserts(
        simple_data[0], print_outlier_session_ids=True, run_find_best_fit=True
    )
    for mock in [
        model_summary_mock,
        ccpr_grid_mock,
        partregress_grid_mock,
        qqplot_mock,
        resid_fit_mock,
        residual_plot_mock,
        regression_summary_mock,
        backward_stepwise_selection_mock,
        outlier_session_ids_mock,
        best_fit_mock,
    ]:
        mock.assert_called_once()


def test_backward_stepwise_selection_raises_for_n_smaller_p():
    model = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=pd.DataFrame(
            {
                "y_1234567a": [1],
                "x1_1234567": [2],
                "x2_12345678": [3],
                "x3_1234567a": [4],
                "x4_12345678b": [5],
            }
        ),
        formula="y_1234567a ~ x1_1234567 + x2_12345678",
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )
    with pytest.raises(
        ValueError,
        match="Backward stepwise selection: The number of samples has to be greater than the number of predictors.",
    ):
        backward_stepwise_selection(model)


@pytest.mark.parametrize(
    "formula, expected_exog",
    [
        (
            "y_1234567a ~ x1_1234567 + x2_12345678*x4_12345678b + x1_1234567*x3_1234567a * x1_1234567*x2_12345678",
            ["Intercept", "x3_1234567a[T.B]", "x4_12345678b[T.D]", "x4_12345678b[T.E]", "x1_1234567", "x2_12345678"],
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b",
            ["Intercept", "x3_1234567a[T.B]", "x4_12345678b[T.D]", "x4_12345678b[T.E]", "x1_1234567", "x2_12345678"],
        ),
        (
            "y_1234567a ~ x1_1234567 + x2_12345678 + x3_1234567a",
            ["Intercept", "x3_1234567a[T.B]", "x1_1234567", "x2_12345678"],
        ),
    ],
)
def test_backward_stepwise_selection(
    formula: str, expected_exog: List[str], data_with_categories: Tuple[pd.DataFrame, pd.Series]
):
    df, _ = data_with_categories
    fit = ConvenientOlsFit(
        scenario_name="SRS-XX",
        model_name="xxx_12345678",
        df=df,
        formula=formula,
        total_failure_info=TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10),
    )
    final_model = backward_stepwise_selection(fit)
    assert sorted(list(final_model.result.pvalues.index)) == sorted(expected_exog)


def test_get_polynomial_of_parameters_and_order_raises_for_wrong_order():
    with pytest.raises(ValueError, match=r"The order of the polynomial as to be greater than 1. Given 0"):
        get_polynomial_of_parameters_and_order(parameters=["x1", "x2"], order=0)


@pytest.mark.parametrize(
    "parameters, order, expected_polynomial",
    [
        (["x1", "x2", "x3"], 1, "x1 + x2 + x3"),
        (["x1", "x2", "x3"], 2, "x1 + x2 + x3 + x1*x1 + x1*x2 + x1*x3 + x2*x2 + x2*x3 + x3*x3"),
        (["x1", "x2"], 3, "x1 + x2 + x1*x1 + x1*x2 + x2*x2 + x1*x1*x1 + x1*x1*x2 + x1*x2*x2 + x2*x2*x2"),
    ],
)
def test_get_polynomial_of_parameters_and_order(parameters: List[str], order: int, expected_polynomial: str):
    assert get_polynomial_of_parameters_and_order(parameters, order) == expected_polynomial


def test_create_log_dataframe(simple_setup: ConvenientRegressionFit):
    log_data = {
        "x1_1234567": ["removed", "removed", "removed", "removed", "removed"],
        "x2_12345678": ["removed", "removed", "removed", "removed", "removed"],
        "x3_1234567a": ["removed", "removed", "removed", "removed", "removed"],
        "x4_12345678b": ["removed", "removed", "removed", "removed", "removed"],
        "LOOCV mean error": [0, 0, 0, 0, 0],
    }
    expected_df = pd.DataFrame(
        log_data,
        index=pd.Index(reversed(range(0, len(simple_setup.predictors) + 1)), name="k"),
        columns=simple_setup.predictors,
    )
    pd.testing.assert_frame_equal(expected_df, _create_log_dataframe(simple_setup, "LOOCV mean error"))


def test_get_best_model(simple_setup: ConvenientRegressionFit):
    actual_model = _get_best_model(simple_setup)
    expected_predictors = ["x1_1234567", "x2_12345678", "x3_1234567a"]
    assert expected_predictors == actual_model.predictors
