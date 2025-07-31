#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import warnings
from itertools import count
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
from scipy.stats import norm
from statsmodels.distributions.copula.api import GaussianCopula
from tools.conftest import TEST_TIMESTAMPS
from tools.general_utility.helpers import NR_OF_DIGITS_AFTER_COMMA
from tools.marginal_distributions.marginal_distributions import (
    ConvenientMarginalDistributionFit,
    ConvenientMarginalDistributionFitKDEUnivariate,
    ConvenientMarginalDistributionFitSciPy,
    ConvenientMarginalDistributionFitSciPyFrozen,
    ConvenientNataf,
)
from tools.regression_models.regression_models import ConvenientOlsFit
from tools.test_helpers.file_creation_helpers import assert_file_content


@pytest.fixture
def common_situation_model_params() -> Dict[str, Any]:
    return {
        "seed": 1234,
        "n_test_cases": 2,
        "scenario_occurrence_rate_per_hour": 0.001,
    }


@pytest.fixture
def expected_correlation_matrix() -> Dict[str, List[Any]]:
    return {
        "parameter_order": ["resid_y1_12345678b", "resid_y_1234567a"],
        "matrix_rows": [[1.0, -0.066926630510], [-0.066926630510, 1.0]],
    }


@pytest.fixture
def expected_correlation_matrix_json(expected_correlation_matrix: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {**TEST_TIMESTAMPS, **expected_correlation_matrix}


@pytest.fixture
def expected_model_file_content(expected_correlation_matrix: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {
        **TEST_TIMESTAMPS,
        "parameters": [
            {
                "name": "resid_y1_12345678b",
                "transformation_function": "x",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [0.0, 1.00191347128],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.191519450379, 0.622108771004, 0.437727739007],
                    "y_predict": [-0.873979843657, 0.311619028184, -0.157032650146],
                },
                "codebeamer_reference": "abc",
            },
            {
                "name": "resid_y_1234567a",
                "transformation_function": "x",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [0.0, 0.999258072857],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.191519450379, 0.622108771004, 0.437727739007],
                    "y_predict": [-0.871663511194, 0.31079313583, -0.156616462258],
                },
                "codebeamer_reference": "abc",
            },
        ],
        "correlation_matrix": {"timestamp_input_data": None, "timestamp_export": None, **expected_correlation_matrix},
        "scenario_occurrence_rate_per_hour": 1000.0,
        "filtering_function": "None",
    }


def test__to_marginal_distribution_transfers_params_correctly(simple_setup: ConvenientOlsFit):
    resid_dist = simple_setup.to_marginal_distribution()
    assert isinstance(resid_dist, ConvenientMarginalDistributionFitSciPyFrozen)
    assert resid_dist.ds.equals(simple_setup.df["resid"])
    assert resid_dist.ds.name == "resid_y_1234567a"
    assert all(np.isclose(resid_dist.dist.args, (0, 1), rtol=0.01))


def test__frozen_marginal_distribution_uses_cdf_and_ppf_of_passed_distribution():
    dist_mock = Mock()
    resid_dist = ConvenientMarginalDistributionFitSciPyFrozen(dist=dist_mock, ds=Mock())
    resid_dist.cdf(123)
    resid_dist.icdf(456)
    dist_mock.cdf.assert_called_once_with(123)
    dist_mock.ppf.assert_called_once_with(456)


def test__params_in_frozen_marginal_distribution_passed_on_correctly():
    pars = ["test_par1", "test_par2"]
    dist_mock = Mock()
    dist_mock.args = pars
    resid_dist = ConvenientMarginalDistributionFitSciPyFrozen(dist=dist_mock, ds=Mock())
    assert resid_dist.pars == pars
    assert resid_dist.dist == dist_mock


def test__plot_marginal_distribution(simple_setup: ConvenientOlsFit, mock_plots: None):
    resid_dist = simple_setup.to_marginal_distribution()
    resid_dist.compare_fit_and_data(x_lim=[-4, 4])
    resid_dist.compare_fit_and_data(x_lim=[-4, 4], type="cdf")
    resid_dist.compare_fit_and_data(type="qq")


@pytest.mark.parametrize(
    "filtering_function, expected_sampling_filters",
    [
        (None, {}),
        ("dist_0>0", {"df['dist_0']>0": "dist_0>0"}),
        (
            "dist_0>0.01, abs(dist_0)>np.log(dist_1)",
            {
                "df['dist_0']>0.01": "dist_0>0.01",
                "abs(df['dist_0'])>np.log(df['dist_1'])": "abs(dist_0)>np.log(dist_1)",
            },
        ),
    ],
)
def test_sampling_filters(
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
    filtering_function: str,
    expected_sampling_filters: Dict[str, str],
):
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        **TEST_TIMESTAMPS,
        name="test_nataf_1",
        filtering_function=filtering_function,
    )
    assert convenient_nataf_setup._sampling_filters == expected_sampling_filters


def test__plot_model_vs_original_raises_for_too_many_points(
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
):
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        **TEST_TIMESTAMPS,
        name="test_nataf",
    )
    with pytest.raises(ValueError, match="Can only plot a maximum of 1000 points. Requested number of points: 9999"):
        convenient_nataf_setup.plot_model_vs_original(n=9999)


def test__creates_file_containing_correlation_matrix_without_backslash(
    tmpdir,
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
    expected_correlation_matrix_json: Dict[str, Any],
):
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        **TEST_TIMESTAMPS,
        name="test_nataf_1",
    )
    convenient_nataf_setup.save_correlation_matrix(folder=tmpdir)
    assert_file_content(
        expected_content=expected_correlation_matrix_json,
        tmpdir=tmpdir,
        filename="test_nataf_1_correlation_matrix.json",
        nr_of_digits_after_comma=NR_OF_DIGITS_AFTER_COMMA,
    )


def test__creates_file_containing_correlation_matrix_with_backslash(
    tmpdir,
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
    expected_correlation_matrix_json: Dict[str, Any],
):
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        **TEST_TIMESTAMPS,
        name="test_nataf_2",
    )
    convenient_nataf_setup.save_correlation_matrix(folder=tmpdir)
    assert_file_content(
        expected_content=expected_correlation_matrix_json,
        tmpdir=tmpdir,
        filename="test_nataf_2_correlation_matrix.json",
        nr_of_digits_after_comma=NR_OF_DIGITS_AFTER_COMMA,
    )


@pytest.mark.parametrize(
    ["timestamp_input_data", "timestamp_export"],
    [(None, TEST_TIMESTAMPS["timestamp_export"]), (TEST_TIMESTAMPS["timestamp_input_data"], None), (None, None)],
)
def test__raises_warning_one_of_the_timestamps_missing_while_exporting_model_and_correlation_matrix(
    tmpdir,
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
    timestamp_input_data: Optional[str],
    timestamp_export: Optional[str],
):
    file_name = "test_nataf_timestamp_missing"
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        timestamp_input_data=timestamp_input_data,
        timestamp_export=timestamp_export,
        name=file_name,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folder = tmpdir
        convenient_nataf_setup.save_correlation_matrix(folder=folder)
        convenient_nataf_setup.save_model(
            folder=folder, codebeamer_reference="abc", scenario_occurrence_rate_per_hour=1000.0
        )
        assert len(w) == 2
        for warning in w:
            assert issubclass(warning.category, UserWarning)
            assert "Either input data timestamp or export timestamp was not provided while exporting the model" == str(
                warning.message
            )


def test__save_model_creates_file_containing_correlation_matrix(
    tmpdir,
    simple_setup: ConvenientOlsFit,
    alternative_simple_setup: ConvenientOlsFit,
    expected_model_file_content: Dict[str, Any],
):
    convenient_nataf_setup = ConvenientNataf(
        dists=(alternative_simple_setup.to_marginal_distribution(), simple_setup.to_marginal_distribution()),
        **TEST_TIMESTAMPS,
        name="test_nataf_1",
    )
    convenient_nataf_setup.save_model(
        folder=tmpdir,
        codebeamer_reference="abc",
        scenario_occurrence_rate_per_hour=1000.0,
    )
    assert_file_content(
        expected_content=expected_model_file_content,
        tmpdir=tmpdir,
        filename="test_nataf_1.json",
        nr_of_digits_after_comma=NR_OF_DIGITS_AFTER_COMMA,
    )


def create_marginal_dist_info(seed: int) -> Dict[str, str]:
    return {
        "name": f"dist_{seed}",
        "codebeamer_reference": f"cb_ref{seed}",
        "transformation_function": f"some_trafo_{seed}",
    }


def create_frozen_marginal_distribution(
    seed: int,
    dist: st.distributions.rv_frozen,
    n: int,
) -> ConvenientMarginalDistributionFitSciPyFrozen:
    return ConvenientMarginalDistributionFitSciPyFrozen(
        dist=dist,
        ds=pd.Series(data=dist.rvs(n, random_state=seed), name=f"data_{seed}"),
        **create_marginal_dist_info(seed),
    )


def create_marginal_distribution(
    seed: int,
    dist: st.distributions.rv_continuous,
    n: int,
) -> ConvenientMarginalDistributionFitSciPy:
    np.random.seed(seed)
    return ConvenientMarginalDistributionFitSciPy(
        dist=dist,
        ds=pd.Series(data=dist(*np.random.rand(2)).rvs(n, random_state=seed), name=f"data_{seed}"),
        **create_marginal_dist_info(seed),
    )


def create_kde_univariate_marginal_distribution(
    seed: int,
    dist: st.distributions.rv_frozen,
    n: int,
) -> ConvenientMarginalDistributionFitKDEUnivariate:
    return ConvenientMarginalDistributionFitKDEUnivariate(
        ds=pd.Series(data=dist.rvs(n, random_state=seed), name=f"data_{seed}"), **create_marginal_dist_info(seed)
    )


def create_nataf_from_non_frozen_marginals(
    dists: List[st.distributions.rv_continuous], n: int = 1000, filtering_function: Optional[str] = None
) -> Tuple[ConvenientNataf, List[ConvenientMarginalDistributionFitSciPy]]:
    seed_gen = count(start=1)
    dists = [create_marginal_distribution(seed=next(seed_gen), dist=dist, n=n) for dist in dists]
    return (
        ConvenientNataf(
            name="nataf_uncorrelated",
            dists=dists,
            filtering_function=filtering_function,
            **TEST_TIMESTAMPS,
        ),
        dists,
    )


def create_nataf_from_kde_univariate_marginals(
    dists: List[st.distributions.rv_frozen], n: int = 1000, filtering_function: Optional[str] = None
) -> Tuple[ConvenientNataf, List[ConvenientMarginalDistributionFitKDEUnivariate]]:
    seed_gen = count(start=1)
    dists = [create_kde_univariate_marginal_distribution(seed=next(seed_gen), dist=dist, n=n) for dist in dists]
    return (
        ConvenientNataf(
            name="nataf_uncorrelated",
            dists=dists,
            filtering_function=filtering_function,
            **TEST_TIMESTAMPS,
        ),
        dists,
    )


def create_nataf_with_uncorrelated_frozen_marginals(
    dists: List[st.distributions.rv_frozen], n: int = 1000
) -> Tuple[ConvenientNataf, List[ConvenientMarginalDistributionFitSciPyFrozen]]:
    seed_gen = count(start=1)
    dists = [create_frozen_marginal_distribution(seed=next(seed_gen), dist=dist, n=n) for dist in dists]
    return (
        ConvenientNataf(
            name="nataf_uncorrelated",
            dists=dists,
        ),
        dists,
    )


def create_nataf_with_correlated_frozen_marginals(
    dists: List[st.distributions.rv_frozen], n: int = 1000, filtering_function: Optional[str] = None
) -> Tuple[ConvenientNataf, List[ConvenientMarginalDistributionFitSciPyFrozen]]:
    k_dim = len(dists)
    # first create a random symmetric positive semi-definite matrix to be used as correlation matrix
    np.random.seed(1)
    helper_matrix = np.random.rand(k_dim, k_dim)
    correlation_matrix = helper_matrix @ helper_matrix.T
    for i in range(k_dim):
        correlation_matrix[i][i] = 1

    copula = GaussianCopula(corr=correlation_matrix, k_dim=k_dim)
    uniform_samples = copula.rvs(n).T

    dists = [
        ConvenientMarginalDistributionFitSciPyFrozen(
            dist=dist,
            ds=pd.Series(dist.ppf(uniform_samples[i])),
            **create_marginal_dist_info(i),
        )
        for i, dist in enumerate(dists)
    ]

    return (
        ConvenientNataf(
            name="nataf_correlated",
            dists=dists,
            **TEST_TIMESTAMPS,
            filtering_function=filtering_function,
        ),
        dists,
    )


def perform_gof_tests(nataf_setup: ConvenientNataf, assumed_cdfs: st.rv_continuous.cdf) -> None:
    df = nataf_setup.sample(n=1000, seed=9012)
    for cdf, i in zip(assumed_cdfs, range(len(assumed_cdfs))):
        assert st.kstest(rvs=df.loc[i, :], cdf=cdf).pvalue > 0.05


def perform_correlation_matrix_checks(
    nataf_setup: ConvenientNataf, dists: List[ConvenientMarginalDistributionFitSciPyFrozen]
) -> None:
    corr = np.corrcoef(*[dist.ds for dist in dists])
    assert np.allclose(nataf_setup.corrcoef(), corr)


def calculate_correlations_diff_assertions(nataf_setup: ConvenientNataf) -> None:
    actual = nataf_setup._calculate_correlations_diff(max_sampling_iterations=100).to_numpy()
    sample = nataf_setup.sample(len(nataf_setup.dists[0].ds), seed=1234)
    data = np.column_stack([dist.ds for dist in nataf_setup._dists])
    corr_data = np.corrcoef(data, rowvar=False)
    corr_model = np.corrcoef(sample, rowvar=False)
    expected = abs(corr_model - corr_data)
    np.testing.assert_array_almost_equal(actual, expected)


def test_calculate_correlations_diff_creates_expected_dataframe_when_data_uncorrelated():
    nataf_setup, _ = create_nataf_with_uncorrelated_frozen_marginals(dists=[st.uniform(), st.norm(loc=5, scale=0.01)])
    calculate_correlations_diff_assertions(nataf_setup)


def test_calculate_correlations_diff_creates_expected_dataframe_when_data_correlated():
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.uniform(), st.norm(loc=5, scale=0.01)])
    calculate_correlations_diff_assertions(nataf_setup)


def check_samples_correctly_correlated(nataf_setup: ConvenientNataf) -> None:
    df = nataf_setup.sample(n=1000, seed=9012)
    np.testing.assert_allclose(nataf_setup.corrcoef(), np.corrcoef(*np.transpose(df.values)), atol=0.05)


def test_nataf_samples_have_correct_marginal_distributions_when_uncorrelated():
    nataf_setup, _ = create_nataf_with_uncorrelated_frozen_marginals(dists=[st.uniform(), st.norm(loc=5, scale=0.01)])
    perform_gof_tests(nataf_setup=nataf_setup, assumed_cdfs=[st.uniform.cdf, st.norm.cdf])


def test_nataf_samples_have_correct_marginal_distributions_when_correlated():
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm()] * 2)
    perform_gof_tests(nataf_setup=nataf_setup, assumed_cdfs=[st.norm.cdf] * 2)


def test_nataf_samples_have_correct_correlation_when_marginals_uncorrelated():
    nataf_setup, _ = create_nataf_with_uncorrelated_frozen_marginals(dists=[st.uniform(), st.norm(loc=5, scale=0.01)])
    check_samples_correctly_correlated(nataf_setup)


def test_nataf_samples_have_correct_correlation_when_marginals_correlated():
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm(loc=0, scale=0.01)] * 2)
    check_samples_correctly_correlated(nataf_setup)


def test_correlation_coefficient_does_not_change_using_nataf_when_marginals_correlated():
    perform_correlation_matrix_checks(
        *create_nataf_with_correlated_frozen_marginals(dists=[st.norm(loc=0, scale=0.01)] * 2)
    )


def test_correlation_coefficient_does_not_change_using_nataf_when_marginals_uncorrelated():
    perform_correlation_matrix_checks(
        *create_nataf_with_uncorrelated_frozen_marginals(dists=[st.norm(loc=0, scale=0.01)] * 2)
    )


def test_correlation_coefficient_small_when_samples_not_correlated():
    nataf_setup, _ = create_nataf_with_uncorrelated_frozen_marginals(dists=[st.uniform(), st.norm(loc=5, scale=0.1)])
    assert nataf_setup.corrcoef()[0][1] < 0.05


def test_correlation_coefficient_big_when_samples_are_correlated():
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm(loc=0, scale=0.01)] * 2)
    assert nataf_setup.corrcoef()[0][1] > 0.05


@pytest.mark.parametrize(
    "filtering_function, function_for_df, expected_warnings",
    [
        ("dist_1>0", "df['dist_1'] > 0", []),
        ("abs(dist_1)>0.01", "abs(df['dist_1'])>0.01", []),
        ("np.log(dist_1)>-4", "np.log(df['dist_1'])>-4", []),
        ("abs(dist_1)>0.01, dist_0>0", "(abs(df['dist_1'])>0.01) & (df['dist_0']>0)", []),
        (
            "dist_1<0, another_one>0, a>=b",
            "df['dist_1']<0",
            ["Failed to apply filtering another_one>0.", "Failed to apply filtering a>=b."],
        ),
    ],
)
def test_filtering_function_works_as_expected(
    filtering_function: str, function_for_df: str, expected_warnings: List[str]
):
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(loc=0, scale=0.1)] * 2, filtering_function=filtering_function
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(action="always")
        df = nataf_setup.sample(n=1000, seed=9012)
        assert all(eval(function_for_df))
        w = [
            str(warning.message) for warning in w if type(warning.message) == UserWarning
        ]  # There can be a runtime warning about invalid values in np.log
        for expected_warning in expected_warnings:
            assert expected_warning in w
        assert df.shape == (1000, 2)


def test__sample__warns_for_too_restrictive_filtering():
    filtering_function = "dist_1<0.5"
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(loc=0, scale=1)] * 2, filtering_function=filtering_function
    )
    with warnings.catch_warnings(record=True) as w:
        _ = nataf_setup.sample(n=1000, seed=9012, max_iterations=100)
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "In at least one iteration of sampling, more than 30% of the samples have been rejected by your filter function. This results in very inefficient sampling."
        )


def test__sample__warns_when_not_finding_valid_number_of_samples():
    filtering_function = "dist_1>0.5"
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(loc=0, scale=2)] * 2, filtering_function=filtering_function
    )
    with warnings.catch_warnings(record=True) as w:
        df = nataf_setup.sample(n=1000, seed=9012, max_iterations=5)
        assert "Could not find valid number of samples in 5 iterations!" in [str(warning.message) for warning in w]
        assert len(df.index) < 1000


def test__plot_model_vs_original__warns_when_not_finding_valid_number_of_samples(mock_plots: None):
    filtering_function = "dist_1>0.5"
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(loc=0, scale=2)] * 2, filtering_function=filtering_function
    )
    with warnings.catch_warnings(record=True) as w:
        nataf_setup.plot_model_vs_original(n=1000, max_sampling_iterations=5)
        assert "Could not find valid number of samples in 5 iterations!" in [str(warning.message) for warning in w]


def test__check_correlations__warns_when_not_finding_valid_number_of_samples():
    filtering_function = "dist_1>0.5"
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(loc=0, scale=2)] * 2, filtering_function=filtering_function
    )
    with warnings.catch_warnings(record=True) as w:
        nataf_setup.check_correlations(max_sampling_iterations=5)
        assert "Could not find valid number of samples in 5 iterations!" in [str(warning.message) for warning in w]


def test_no_filtering_function_works_as_expected():
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm(loc=0, scale=0.1)] * 2)
    df = nataf_setup.sample(n=1000, seed=9012)
    assert df.shape == (1000, 2)


def test_to_situation_model_creates_model_with_all_expected_entries_with_frozen_dists(
    common_situation_model_params: Dict[str, Any]
):
    pars = [(0, 0.01), (1, 0.01)]
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(
        dists=[st.norm(*param) for param in pars], filtering_function="dist_1<0"
    )
    situation_model = nataf_setup.to_situation_model(**common_situation_model_params)

    expected_situation_model_dict = {
        **TEST_TIMESTAMPS,
        "parameters": [
            {
                "name": "dist_0",
                "transformation_function": "some_trafo_0",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [0.0, 0.01],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [-0.00872310702180929, 0.0031102389289802542],
                },
                "codebeamer_reference": "cb_ref0",
            },
            {
                "name": "dist_1",
                "transformation_function": "some_trafo_1",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [1.0, 0.01],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [0.9912768929781907, 1.0031102389289803],
                },
                "codebeamer_reference": "cb_ref1",
            },
        ],
        "correlation_matrix": {
            "parameter_order": ["dist_0", "dist_1"],
            "matrix_rows": [[1.0, 0.22187555319346774], [0.22187555319346777, 1.0]],
        },
        "scenario_occurrence_rate_per_hour": 0.001,
        "filtering_function": "dist_1<0",
    }
    assert situation_model.dict(exclude_unset=True) == expected_situation_model_dict


def test_to_situation_model_creates_model_without_filtering_function(common_situation_model_params: Dict[str, Any]):
    pars = [(0, 0.01), (1, 0.01)]
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm(*param) for param in pars])
    situation_model = nataf_setup.to_situation_model(**common_situation_model_params)
    assert situation_model.dict(exclude_unset=True)["filtering_function"] == "None"


def test_to_situation_model_creates_model_with_all_expected_entries_non_frozen(
    common_situation_model_params: Dict[str, Any]
):
    nataf_setup, _ = create_nataf_from_non_frozen_marginals(dists=[st.norm, st.norm], filtering_function="dist_1<0")
    situation_model = nataf_setup.to_situation_model(**common_situation_model_params)

    expected_situation_model_dict = {
        **TEST_TIMESTAMPS,
        "parameters": [
            {
                "name": "dist_1",
                "transformation_function": "some_trafo_1",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [0.44497958193147497, 0.7066413058393833],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [-0.17143119175532612, 0.6647619117561842],
                },
                "codebeamer_reference": "cb_ref1",
            },
            {
                "name": "dist_2",
                "transformation_function": "some_trafo_2",
                "marginal_distribution": "norm",
                "marginal_distribution_parameters": [0.43475433362918403, 0.026039579860484833],
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [0.4120397294365881, 0.44285326512682105],
                },
                "codebeamer_reference": "cb_ref2",
            },
        ],
        "correlation_matrix": {
            "parameter_order": ["dist_1", "dist_2"],
            "matrix_rows": [[1.0, -0.010178402523321691], [-0.010178402523321691, 1.0]],
        },
        "scenario_occurrence_rate_per_hour": 0.001,
        "filtering_function": "dist_1<0",
    }
    assert situation_model.dict(exclude_unset=True) == expected_situation_model_dict


def test_to_situation_model_creates_model_with_all_expected_entries_kde_univariate(
    common_situation_model_params: Dict[str, Any]
):
    pars = [(0, 0.01), (1, 0.01)]
    nataf_setup, _ = create_nataf_from_kde_univariate_marginals(
        dists=[st.norm(*param) for param in pars], filtering_function="dist_1<0"
    )
    situation_model = nataf_setup.to_situation_model(**common_situation_model_params)
    d = situation_model.dict(exclude_unset=True)

    expected_situation_model_dict = {
        **TEST_TIMESTAMPS,
        "parameters": [
            {
                "name": "dist_1",
                "transformation_function": "some_trafo_1",
                "marginal_distribution": "Kernel Density",
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [-0.007858045822280757, 0.003643834318130222],
                },
                "codebeamer_reference": "cb_ref1",
            },
            {
                "name": "dist_2",
                "transformation_function": "some_trafo_2",
                "marginal_distribution": "Kernel Density",
                "marginal_distribution_test_cases": {
                    "u_noise": [0.1915194503788923, 0.6221087710398319],
                    "y_predict": [0.9906056949745929, 1.002837437922447],
                },
                "codebeamer_reference": "cb_ref2",
            },
        ],
        "correlation_matrix": {
            "parameter_order": ["dist_1", "dist_2"],
            "matrix_rows": [[1.0, -0.010821083518421107], [-0.010821083518421107, 0.9999999999999999]],
        },
        "scenario_occurrence_rate_per_hour": 0.001,
        "filtering_function": "dist_1<0",
    }
    d = situation_model.dict(exclude_unset=True)
    first_marginal_dist_params = [-0.030537643804263048, 0.9734055054361651]
    for param, dist_param in zip(d["parameters"], first_marginal_dist_params):
        assert dist_param in param.pop("marginal_distribution_parameters")
    assert d == expected_situation_model_dict


def test_correlation_matrix_properties_are_equal():
    pars = [(0, 0.01), (1, 0.01)]
    nataf_setup, _ = create_nataf_with_correlated_frozen_marginals(dists=[st.norm(*param) for param in pars])
    c1 = nataf_setup.correlation_matrix_without_timestamps
    c2 = nataf_setup.correlation_matrix
    assert c1.parameter_order == c2.parameter_order
    assert c1.matrix_rows == c2.matrix_rows


@pytest.mark.parametrize(
    "correlation_matrix",
    [np.array([[1.0, 0.8], [0.8, 1.0]]), np.array([[1.0, 0.1, 0.5], [0.1, 1.0, -0.7], [0.5, -0.7, 1.0]])],
)
def test_sample_gaussian_copula(correlation_matrix: np.ndarray):
    unit = ConvenientNataf(name="test", dists=[Mock()])
    unit.corrcoef = lambda: correlation_matrix
    # Generate samples from Gaussian Copula,...
    U = unit._sample_gaussian_copula(n=100_000, seed=4322)
    # ... transform to "Normal space",...
    X = norm.ppf(U)
    # ...compute correlation coefficient from normal samples,...
    correlation_matrix_estimated = np.corrcoef(X)
    # ...and compare to initial correlation matrix.
    assert correlation_matrix == pytest.approx(correlation_matrix_estimated, rel=0.04)


@pytest.mark.parametrize(
    "fit_class",
    [
        ConvenientMarginalDistributionFitSciPy,
        ConvenientMarginalDistributionFitSciPyFrozen,
        ConvenientMarginalDistributionFitKDEUnivariate,
    ],
)
def test_correlation_matrix_works_with_filtered_data_without_clean_index(
    fit_class: Type[ConvenientMarginalDistributionFit],
):
    """
    This test essentially checks if the index of a Series gets reset when passed to ConvenientMarginalDistributionFit
    If that is not the case, calculate_correlations_diff outputs incorrect data because of NaN values in the
    DataFrame that gets create inside that function.
    """
    ds = pd.Series([1, 0.5, 2, 3.1, 0.8, 0.3])
    ds_1 = ds[ds < 1]
    ds_2 = ds[ds >= 1]

    fit_1_input, fit_2_input = {"ds": ds_1, "name": "fit_1"}, {"ds": ds_2, "name": "fit_2"}
    if fit_class == ConvenientMarginalDistributionFitSciPy:
        fit_1_input["dist"] = fit_2_input["dist"] = norm
    elif fit_class == ConvenientMarginalDistributionFitSciPyFrozen:
        fit_1_input["dist"] = fit_2_input["dist"] = norm(0, 1)
    fit_1 = fit_class(**fit_1_input)
    fit_2 = fit_class(**fit_2_input)

    ds_1_expected = pd.Series([0.5, 0.8, 0.3])
    ds_2_expected = pd.Series([1, 2, 3.1])
    pd.testing.assert_series_equal(fit_1.ds, ds_1_expected)
    pd.testing.assert_series_equal(fit_2.ds, ds_2_expected)

    dists = [fit_1, fit_2]
    nataf = ConvenientNataf("test_nataf", dists=dists)
    correlations_diff = nataf._calculate_correlations_diff(max_sampling_iterations=100)
    assert correlations_diff.notnull().values.all()
