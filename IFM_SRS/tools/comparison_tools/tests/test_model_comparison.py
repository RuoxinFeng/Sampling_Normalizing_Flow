#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
import warnings
from os.path import dirname, join
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, uniform
from tools.comparison_tools.divergence_metrics import emd, empirical_js_divergence, ks_test, mse
from tools.comparison_tools.model_comparison import ModelComparison
from tools.conftest import TEST_TIMESTAMPS
from tools.regression_models.regression_models import ConvenientOlsFit

BASE_PATH = join(dirname(__file__), "test_data", "model_comparison")
PATH_OLD_MODEL = join(BASE_PATH, "example_position_model_model_comparison.json")
PATH_OLD_MODEL_WITHOUT_TRAFO = join(BASE_PATH, "example_position_model_model_comparison_without_trafo.json")


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spv_1234567a": [3.28, 1.56, 1.94, 2.26],
            "ifv_23456789": [0.1, 0.2, 0.3, 0.4],
            "categorical_ifv_12345678": ["category 1", "some other category", "category_2", "something else"],
        }
    )


@pytest.fixture
def expected_dependent_variable_estimate(df: pd.DataFrame) -> pd.Series:
    return (
        1
        + (df["categorical_ifv_12345678"] == "category 1") * 2
        + np.power((df["categorical_ifv_12345678"] == "category 1"), 2) * 0.1
        + df["ifv_23456789"] * 3
        + df["ifv_23456789"] * np.power(df["ifv_23456789"], 3) * (df["categorical_ifv_12345678"] == "category_2") * 4
        + np.exp(df["ifv_23456789"]) * 0.1
        + np.log(df["ifv_23456789"]) * 0.1
        + np.power(df["ifv_23456789"], 5) * 0.1
    )  # from input file


@pytest.fixture
def new_model(df: pd.DataFrame) -> ConvenientOlsFit:
    new_model = ConvenientOlsFit(
        scenario_name="test",
        model_name="test",
        df=df,
        formula="np.log(spv_1234567a) ~ ifv_23456789 + categorical_ifv_12345678",
        total_failure_info=None,
        inverse_output_trafo=np.exp,
        **TEST_TIMESTAMPS,
    )
    new_model.nongaussian_resid_fit(uniform)
    return new_model


@pytest.fixture
def new_model_with_modified_df(df: pd.DataFrame) -> ConvenientOlsFit:
    modified_df = df.rename(
        columns={
            "ifv_23456789": "new_ifv_23456789",
            "categorical_ifv_12345678": "new_categorical_ifv_12345678",
            "spv_1234567a": "new_spv_1234567a",
        }
    )
    return ConvenientOlsFit(
        scenario_name="test",
        model_name="test",
        df=modified_df,
        formula="new_spv_1234567a ~ new_ifv_23456789 + new_categorical_ifv_12345678",
        total_failure_info=None,
        **TEST_TIMESTAMPS,
    )


@pytest.fixture
def comparison_model(new_model: ConvenientOlsFit) -> ModelComparison:
    return ModelComparison(
        new_model=new_model,
        path_old_model=PATH_OLD_MODEL,
        name_new_model="New model name",
        name_old_model="Old model name",
        inverse_output_trafo=np.exp,
    )


def test__model_comparison__dependent_variable(
    new_model: ConvenientOlsFit,
):
    unit = ModelComparison(
        new_model=new_model,
        path_old_model=PATH_OLD_MODEL_WITHOUT_TRAFO,
    )
    assert unit.dependent_variable == "spv_1234567a"


def test__model_comparison_instantiation__raises_warning_when_dependent_variable_transformed_and_no_inverse_trafo_given(
    new_model: ConvenientOlsFit,
):
    with pytest.warns(
        UserWarning,
        match="The dependent variable has been transformed and no inverse transformation has been given. The Regression output will correspond to the transformed variable!",
    ):
        ModelComparison(
            new_model=new_model,
            path_old_model=PATH_OLD_MODEL,
        )


def test__model_comparison_instantiation__raises_warning_when_dependent_variable_not_transformed_and_inverse_trafo_given(
    new_model: ConvenientOlsFit,
):
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "An inverse output transformation was given but no transformation of the dependent variable spv_1234567a was detected. At the moment, only the transformations of the type np.(...) are recognized. In case the dependent variable was transformed by a different function and the correct inverse transformation has been given, this warning can be ignored."
        ),
    ):
        ModelComparison(
            new_model=new_model,
            path_old_model=PATH_OLD_MODEL_WITHOUT_TRAFO,
            inverse_output_trafo=np.exp,
        )


def test__model_comparison_instantiation__does_not_raise_warning_when_inverse_trafo_given(new_model: ConvenientOlsFit):
    with warnings.catch_warnings(record=True) as record:
        ModelComparison(
            new_model=new_model,
            path_old_model=PATH_OLD_MODEL,
            inverse_output_trafo=np.exp,
        )
    assert len(record) == 0


def test__old_model_result__expected_formula_parsed_from_input_file_when_old_model_result_requested(
    new_model: ConvenientOlsFit,
):
    expected_formula = "1.0 + (df['categorical_ifv_12345678']=='category 1') * 2.0 + np.power((df['categorical_ifv_12345678']=='category 1'), 2) * 0.1 + df['ifv_23456789'] * 3.0 + df['ifv_23456789'] * np.power(df['ifv_23456789'], 3) * (df['categorical_ifv_12345678']=='category_2') * 4.0 + abs(np.exp(df['ifv_23456789'])) * 0.1 + np.log(df['ifv_23456789']) * 0.1 + np.power(df['ifv_23456789'], 5) * 0.1"
    compare = ModelComparison(
        new_model=new_model,
        path_old_model=PATH_OLD_MODEL,
        inverse_output_trafo=np.exp,
    )
    compare.old_model_result
    assert compare._regression_formula == expected_formula


def test__old_model_result__correctly_estimates_with_transformation_of_dependent_variable(
    comparison_model: ModelComparison, expected_dependent_variable_estimate: pd.Series
):
    uniform_noise = uniform.rvs(size=comparison_model.df.shape[0], random_state=123)
    assert np.allclose(
        comparison_model.old_model_result,
        np.exp(expected_dependent_variable_estimate + norm(1, 1).ppf(uniform_noise)),
    )


def test__old_model_result__correctly_estimates_dependent_variable_without_transformation_of_dependent_variable(
    expected_dependent_variable_estimate: pd.Series, new_model: ConvenientOlsFit
):
    compare = ModelComparison(
        new_model=new_model,
        path_old_model=PATH_OLD_MODEL_WITHOUT_TRAFO,
    )
    assert np.allclose(
        compare.old_model_result,
        expected_dependent_variable_estimate + norm(1, 1).ppf(compare._uniform_noise),
    )


def test__old_model_result__raises_when_formula_entries_not_contained_in_dataframe(
    new_model_with_modified_df: ConvenientOlsFit,
):
    comparison_model = ModelComparison(
        new_model=new_model_with_modified_df,
        path_old_model=PATH_OLD_MODEL_WITHOUT_TRAFO,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No matching column found in the data for old model parameter categorical_ifv_12345678[T.category 1]"
        ),
    ):
        comparison_model.old_model_result


def test__new_model_result__correctly_estimates_dependent_variable(
    df: pd.DataFrame, new_model: ConvenientOlsFit, comparison_model: ModelComparison
):
    new_model_params = new_model.result.params
    pd.testing.assert_series_equal(
        comparison_model.new_model_result,
        np.exp(
            new_model_params["Intercept"]
            + df["ifv_23456789"] * new_model_params["ifv_23456789"]
            + (df["categorical_ifv_12345678"] == "category_2")
            * new_model_params["categorical_ifv_12345678[T.category_2]"]
            + (df["categorical_ifv_12345678"] == "some other category")
            * new_model_params["categorical_ifv_12345678[T.some other category]"]
            + (df["categorical_ifv_12345678"] == "something else")
            * new_model_params["categorical_ifv_12345678[T.something else]"]
        ),
    )


@patch("tools.comparison_tools.model_comparison.ff.create_distplot")
def test__compare_predictions__distplot_created_from_expected_data(
    ff_mock: MagicMock, comparison_model: ModelComparison
):
    comparison_model.compare_predictions()
    for act, exp in zip(
        ff_mock.call_args.kwargs["hist_data"],
        [
            comparison_model.new_model.df["spv_1234567a"],
            comparison_model.new_model_result,
            comparison_model.old_model_result,
        ],
    ):
        assert act.equals(exp)


@patch("tools.comparison_tools.model_comparison.ff.create_distplot")
def test__compare_predictions__prints_expected_output(
    ff_mock: MagicMock, comparison_model: ModelComparison, capsys: pytest.CaptureFixture
):
    comparison_model.compare_predictions()
    captured = capsys.readouterr()
    assert (
        f"Used regression formula parsed from the input file (not including inverse output transformation if given): \n {comparison_model._regression_formula}"
        in captured.out
    )


@patch("tools.comparison_tools.model_comparison.display_dataframe")
def test__compare_params__expected_parameters_are_displayed(
    display_dataframe_mock: MagicMock, comparison_model: ModelComparison
):
    comparison_model.compare_params()
    displayed_df = display_dataframe_mock.call_args.args[0]
    assert set(displayed_df["Parameter"]) == set(comparison_model.old_model.params.keys()) | set(
        comparison_model.new_model.result.params.keys()
    )


@patch("tools.comparison_tools.model_comparison.display_dataframe")
def test__compare_params__expected_parameter_values_are_displayed(
    display_dataframe_mock: MagicMock, comparison_model: ModelComparison
):
    comparison_model.compare_params()
    displayed_df = display_dataframe_mock.call_args.args[0]
    model_param_dicts = {
        "New model name": comparison_model.new_model.result.params,
        "Old model name": comparison_model.old_model.params,
    }
    for model, model_dict in model_param_dicts.items():
        for param, value in displayed_df[["Parameter", model]].dropna().itertuples(index=False):
            assert model_dict[param] == value


@patch("tools.comparison_tools.model_comparison.display_dataframe")
def test__compare_params__abs_difference_of_displayed_parameters_correctly_calculated(
    display_dataframe_mock: MagicMock, comparison_model: ModelComparison
):
    comparison_model.compare_params()
    displayed_df = display_dataframe_mock.call_args.args[0]
    df = displayed_df[
        displayed_df["Parameter"].isin(
            set(comparison_model.old_model.params.keys()) & set(comparison_model.new_model.result.params.keys())
        )
    ]
    assert abs(df["New model name"] - df["Old model name"]).equals(df["Absolute Difference"])


def test__create_comparison_information_list__information_data_contains_expected_comparisons(
    comparison_model: ModelComparison,
):
    info_list = comparison_model._create_comparison_information_list()
    expected_comparisons = [
        ("New model name", "Old model name"),
        ("Actual observations", "New model name"),
        ("Actual observations", "Old model name"),
    ]
    assert expected_comparisons == [comparison.model_names for comparison in info_list]


def test__create_comparison_information_list__information_data_contains_expected_data(
    comparison_model: ModelComparison,
):
    info_list = comparison_model._create_comparison_information_list()
    expected_data = [
        (comparison_model.new_model_result, comparison_model.old_model_result),
        (comparison_model.y_observed, comparison_model.new_model_result),
        (comparison_model.y_observed, comparison_model.old_model_result),
    ]
    for data_set_exp, comparison in zip(expected_data, info_list):
        for data_exp, data_act in zip(data_set_exp, comparison.data):
            pd.testing.assert_series_equal(data_exp, data_act)


def test__create_comparison_information_list__information_data_contains_expected_titles(
    comparison_model: ModelComparison,
):
    info_list = comparison_model._create_comparison_information_list()
    expected_titles = [
        "New model name vs Old model name",
        "Actual observations vs New model name",
        "Actual observations vs Old model name",
    ]
    assert expected_titles == [comparison.title for comparison in info_list]


@patch("tools.comparison_tools.model_comparison.display_dataframe")
def test__model_similarity_quantities__displays_correct_data(
    display_dataframe_mock: MagicMock, mock_plots: None, comparison_model: ModelComparison
):
    comparison_model.model_similarity_quantities()
    data_to_insert = [
        [comparison_model.new_model_result, comparison_model.old_model_result],
        [comparison_model.y_observed, comparison_model.new_model_result],
        [comparison_model.y_observed, comparison_model.old_model_result],
    ]
    expected_df = pd.DataFrame(
        data={
            "MSE": [mse(*data) for data in data_to_insert],
            "EMD": [emd(*data) for data in data_to_insert],
            "JS-Divergence": [empirical_js_divergence(*data, nbins=50) for data in data_to_insert],
            "KS test statistic": [ks_test(*data).statistic for data in data_to_insert],
        },
        index=[
            "New model name vs Old model name",
            "Actual observations vs New model name",
            "Actual observations vs Old model name",
        ],
    )
    displayed_df = display_dataframe_mock.call_args.args[0]
    pd.testing.assert_frame_equal(displayed_df, expected_df)


def test__model_similarity_quantities__notes_printed_correctly(
    mock_plots: None, comparison_model: ModelComparison, capsys: pytest.CaptureFixture
):
    comparison_model.model_similarity_quantities()
    captured = capsys.readouterr()
    assert (
        "Notes: \n 1) JS divergence is calculated from a discretized distribution \n 2) The KS test column shows the \033[1mtest statistic\033[0m of the Kolmogorov-Smirnov hypothesis test. The KS test statistic can be interpreted as the highest possible deviation between the two empirical cumulative distribution functions of the provided data, see figures below. As such, it can take values ranging from 0 (models are identical) to 1 (not similar at all)."
        in captured.out
    )
