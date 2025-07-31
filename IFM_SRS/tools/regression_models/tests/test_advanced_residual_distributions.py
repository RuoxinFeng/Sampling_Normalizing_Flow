#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import List, Tuple

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, norm, rv_continuous
from scipy.stats._distn_infrastructure import rv_frozen
from tools.regression_models.advanced_residual_distributions import piecewise_distribution, truncated_distribution


@pytest.mark.parametrize("dist, lower_bound, upper_bound", [(gamma, 1, 3), (norm, 4, 5)])
def test__trunctated_distribution__initialized_correctly(dist: rv_continuous, lower_bound: float, upper_bound: float):
    td = truncated_distribution(dist=dist, lower_bound=lower_bound, upper_bound=upper_bound)

    assert td.dist.name == dist.name
    assert td.dist.xtol == 1e-10
    assert td.name == f"truncated_{dist.name}"
    assert td._lower_bound == lower_bound
    assert td._upper_bound == upper_bound
    assert td.shapes == td.dist.shapes


def test__trunctated_distribution__raises_for_initialization_with_frozen_dist():
    with pytest.raises(TypeError, match="'dist' cannot be frozen."):
        truncated_distribution(dist=gamma(1, 0, 1), lower_bound=1, upper_bound=2)


def test__truncated_distribution__raises_for_lower_bound_geq_upper_bound():
    with pytest.raises(ValueError, match="The lower bound 0 has to be lower than the upper bound 0."):
        truncated_distribution(dist=gamma, lower_bound=0, upper_bound=0)
    with pytest.raises(ValueError, match="The lower bound 0.1 has to be lower than the upper bound 0."):
        truncated_distribution(dist=gamma, lower_bound=0.1, upper_bound=0)


def test__truncated_distribution__frozen():
    td = truncated_distribution(dist=gamma, lower_bound=0, upper_bound=1)
    frozen_td = td(1, 0, scale=1)

    assert isinstance(frozen_td, rv_frozen)
    assert frozen_td.dist.name == td.name
    assert frozen_td.dist.dist.name == gamma.name
    assert frozen_td.dist._lower_bound == 0
    assert frozen_td.dist._upper_bound == 1
    assert frozen_td.args == (1, 0)
    assert frozen_td.kwds == {"scale": 1}


@pytest.mark.parametrize("lower_bound, upper_bound", [(-np.inf, np.inf), (0, np.inf), (-np.inf, 1), (0, 1)])
def test__truncated_distribution__pdf__integrates_to_one(lower_bound: float, upper_bound: float):
    a, loc, scale = 1, 0, 1
    td = truncated_distribution(dist=gamma, lower_bound=lower_bound, upper_bound=upper_bound)
    integral, _ = quad(func=td.pdf, a=lower_bound, b=upper_bound, args=(a, loc, scale))

    assert integral == pytest.approx(1)


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 1 required positional argument: 'a'",
        ),
        ((2, 3, 4, 5), "_parse_args() takes from 2 to 4 positional arguments but 5 were given"),
    ],
)
def test__truncated_distribution__pdf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    td = truncated_distribution(dist=gamma)
    with pytest.raises(TypeError, match=re.escape(err)):
        td.pdf(1, *args)


def test__truncated_distribution__cdf__between_zero_and_one():
    td = truncated_distribution(dist=norm, lower_bound=-1, upper_bound=1)
    x = np.linspace(-2, 2, 1000)
    y = td.cdf(x)

    assert np.all(y[x <= -1] == 0.0)
    assert np.all((0 < y[(-1 < x) & (x < 1)]) & (y[(-1 < x) & (x < 1)] < 1))
    assert np.all(y[x >= 1] == 1.0)


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 1 required positional argument: 'a'",
        ),
        ((2, 3, 4, 5), "_parse_args() takes from 2 to 4 positional arguments but 5 were given"),
    ],
)
def test__truncated_distribution__cdf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    td = truncated_distribution(dist=gamma)
    with pytest.raises(TypeError) as e:
        td.cdf(1, *args)
    assert err in str(e)


def test__truncated_distribution__ppf__inverse_of_cdf():
    td = truncated_distribution(dist=gamma, lower_bound=1, upper_bound=3)
    frozen_td = td(11.35, 0, 0.2)
    q = np.linspace(0, 1, 1000)
    cdf_ppf = frozen_td.cdf(frozen_td.ppf(q))

    assert np.all(np.isclose(q, cdf_ppf))


def test__truncated_distribution__ppf__between_lower_and_upper_bound():
    td = truncated_distribution(dist=norm, lower_bound=-1, upper_bound=1)
    q = np.linspace(0, 1, 1000)
    y = td.ppf(q)

    assert y[0] == -1.0
    assert y[-1] == 1.0
    assert np.all((-1 <= y) & (y <= 1))


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 1 required positional argument: 'a'",
        ),
        ((2, 3, 4, 5), "_parse_args() takes from 2 to 4 positional arguments but 5 were given"),
    ],
)
def test__truncated_distribution__ppf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    td = truncated_distribution(dist=gamma)
    with pytest.raises(TypeError) as e:
        td.ppf(1, *args)
    assert err in str(e)


def test__truncated_distribution__fit__amount_of_params_correct():
    td = truncated_distribution(dist=gamma, lower_bound=-1, upper_bound=1)
    data = np.linspace(-1, 1, 1000, endpoint=False)
    result = td.fit(data)

    assert len(result) == 3  # gamma.numargs + 2


def test__truncated_distribution__fit__raises_for_data_out_of_bounds():
    td = truncated_distribution(dist=gamma, lower_bound=-1, upper_bound=1)
    data = np.linspace(-2, 2, 1000)
    with pytest.raises(ValueError, match=re.escape("The provided data is not in the specified bounds [-1, 1).")):
        td.fit(data)


@pytest.mark.parametrize("value", (-np.inf, np.nan))
def test__truncated_distribution__fit__raises_for_nonfinite_data(value: float):
    td = truncated_distribution(dist=gamma, lower_bound=-np.inf, upper_bound=1)
    with pytest.raises(ValueError, match="The data contains non-finite values."):
        td.fit([value])


def test__piecewise_distribution__initialized_correctly():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)

    assert pd.threshold == 1
    assert pd.lower_dist.name == "truncated_gamma"
    assert pd.upper_dist.name == "truncated_norm"
    assert pd.shapes == "al, locl, scalel, locu, scaleu, ltmass"
    assert pd.name == "piecewise_gamma_norm"
    assert pd.numargs == 6


@pytest.mark.parametrize(
    "lower_dist, upper_dist, err",
    [
        (gamma(1, 0, 1), norm(3, 2), "'lower_dist' cannot be frozen."),
        (gamma, norm(3, 2), "'upper_dist' cannot be frozen."),
        (gamma(1, 0, 1), norm, "'lower_dist' cannot be frozen."),
    ],
)
def test__piecewise_distribution__raises_for_initialization_with_frozen_dist(lower_dist, upper_dist, err):
    with pytest.raises(TypeError, match=err):
        piecewise_distribution(lower_dist=lower_dist, upper_dist=upper_dist, threshold=1)


def test__piecewise_distribution__frozen():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    frozen_pd = pd(1, 0, scalel=1, locu=0.5, scaleu=2, ltmass=0.4)

    assert isinstance(frozen_pd, rv_frozen)
    assert frozen_pd.dist.name == pd.name
    assert frozen_pd.dist.threshold == 1
    assert frozen_pd.args == (1, 0)
    assert frozen_pd.kwds == {"scalel": 1, "locu": 0.5, "scaleu": 2, "ltmass": 0.4}


def test__piecewise_distribution__pdf__integrates_to_one():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    integral, _ = quad(pd.pdf, -np.inf, np.inf, args=(1, 0, 1, 4, 1, 0.3))

    assert integral == pytest.approx(1)


def test__piecewise_distribution__pdf__integrates_to_ltmass_until_threshold():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    integral, _ = quad(pd.pdf, -np.inf, 1, args=(1, 0, 1, 4, 1, 0.3))

    assert integral == pytest.approx(0.3)


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 6 required positional arguments: 'al', 'locl', 'scalel', 'locu', 'scaleu', and 'ltmass'",
        ),
        ((2, 3, 4, 5, 6, 0.1, 7, 8, 9), "_parse_args() takes from 7 to 9 positional arguments but 10 were given"),
    ],
)
def test__piecewise_distribution__pdf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(TypeError) as e:
        pd.pdf(1, *args)
    assert err in str(e)


def test__piecewise_distribution__pdf__raises_for_wrong_ltmass():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.pdf(1, 1, 2, 3, 4, 5, ltmass=1 + 1e-7)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.pdf(1, 1, 2, 3, 4, 5, ltmass=-1e-7)


def test__piecewise_distribution__cdf__between_zero_and_one():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    x = np.linspace(-10, 10, 1000)
    y = pd.cdf(x, 1, 0, 1, 4, 1, 0.3)

    assert np.all((0 <= y) & (y <= 1))


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 6 required positional arguments: 'al', 'locl', 'scalel', 'locu', 'scaleu', and 'ltmass'",
        ),
        ((2, 3, 4, 5, 6, 0.1, 7, 8, 9), "_parse_args() takes from 7 to 9 positional arguments but 10 were given"),
    ],
)
def test__piecewise_distribution__cdf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(TypeError) as e:
        pd.cdf(1, *args)
    assert err in str(e)


def test__piecewise_distribution__cdf__raises_for_wrong_ltmass():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.cdf(1, 1, 2, 3, 4, 5, ltmass=1 + 1e-7)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.cdf(1, 1, 2, 3, 4, 5, ltmass=-1e-7)


def test__piecewise_distribution__ppf__inverse_of_cdf():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    frozen_pd = pd(1.7, 0.03, 0.5, 3.0, 0.5, 0.5)
    q = np.linspace(0, 1, 1000)
    cdf_ppf = frozen_pd.cdf(frozen_pd.ppf(q))

    assert np.all(np.isclose(q, cdf_ppf))


@pytest.mark.parametrize(
    "args, err",
    [
        (
            (),
            "_parse_args() missing 6 required positional arguments: 'al', 'locl', 'scalel', 'locu', 'scaleu', and 'ltmass'",
        ),
        ((2, 3, 4, 5, 6, 0.1, 7, 8, 9), "_parse_args() takes from 7 to 9 positional arguments but 10 were given"),
    ],
)
def test__piecewise_distribution__ppf__raises_for_wrong_shape_params(args: Tuple[float], err: str):
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(TypeError, match=re.escape(err)):
        pd.ppf(1, *args)


def test__piecewise_distribution__ppf__raises_for_wrong_ltmass():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=1)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.ppf(1, 1, 2, 3, 4, 5, ltmass=1 + 1e-7)
    with pytest.raises(ValueError, match=re.escape("ltmass must be in [0, 1].")):
        pd.ppf(1, 1, 2, 3, 4, 5, ltmass=-1e-7)


def test__piecewise_distribution__fit__amount_of_params_correct():
    threshold = 2
    norm_data = norm(3, 0.5).rvs(1000)
    gamma_data = gamma(2, 0, 0.5).rvs(1000)
    data = np.sort(np.concatenate([gamma_data[gamma_data < threshold], norm_data[norm_data >= threshold]]))

    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=threshold)
    result = pd.fit(data)

    assert len(result) == 6  # lower_dist.numargs + upper_dist.numargs + 5


def test__piecewise_distribution__fit__raises_for_no_data_below_or_above_threshold():
    threshold = 2
    norm_data = norm(3, 0.5).rvs(1000)
    gamma_data = gamma(2, 0, 0.5).rvs(1000)
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=threshold)

    with pytest.raises(ValueError, match="There has to be data above and below the threshold."):
        pd.fit(norm_data[norm_data >= threshold])

    with pytest.raises(ValueError, match="There has to be data above and below the threshold."):
        pd.fit(gamma_data[gamma_data < threshold])


def assert_distribution_functions_equal_for_list_args_and_array(
    dist: rv_continuous, data: List[float] = [0.1, 0.2, 2.3, 2.4]
) -> None:
    quantiles = [0.1, 0.2, 0.7, 0.8]
    params = dist.fit(data)
    assert np.allclose(params, dist.fit(np.array(data)))
    frozen_dist = dist(*params)
    assert np.allclose(frozen_dist.pdf(data), frozen_dist.pdf(np.array(data)))
    assert np.allclose(frozen_dist.cdf(data), frozen_dist.cdf(np.array(data)))
    assert np.allclose(frozen_dist.ppf(quantiles), frozen_dist.ppf(np.array(quantiles)))


def assert_distribution_functions_equal_for_list_kwargs_and_array(
    dist: rv_continuous, data: List[float] = [0.1, 0.2, 2.3, 2.4]
) -> None:
    quantiles = [0.1, 0.2, 0.7, 0.8]
    params = dist.fit(data=data)
    assert np.allclose(params, dist.fit(np.array(data)))
    frozen_dist = dist(*params)
    assert np.allclose(frozen_dist.pdf(x=data), frozen_dist.pdf(np.array(data)))
    assert np.allclose(frozen_dist.cdf(x=data), frozen_dist.cdf(np.array(data)))
    assert np.allclose(frozen_dist.ppf(q=quantiles), frozen_dist.ppf(np.array(quantiles)))


def test__truncated_distribution_functions_do_not_raise_with_list_args_and_give_same_results_as_array():
    td = truncated_distribution(dist=norm, lower_bound=-1, upper_bound=3)
    assert_distribution_functions_equal_for_list_args_and_array(dist=td)


def test__truncated_distribution_functions_do_not_raise_with_list_kwargs_and_give_same_results_as_array():
    td = truncated_distribution(dist=norm, lower_bound=-1, upper_bound=3)
    assert_distribution_functions_equal_for_list_kwargs_and_array(dist=td)


def test__piecewise_distribution_functions_do_not_raise_with_list_args_and_give_same_results_as_array():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=2)
    assert_distribution_functions_equal_for_list_args_and_array(dist=pd)


def test__piecewise_distribution_functions_do_not_raise_with_list_kwargs_and_give_same_results_as_array():
    pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=2)
    assert_distribution_functions_equal_for_list_kwargs_and_array(dist=pd)
