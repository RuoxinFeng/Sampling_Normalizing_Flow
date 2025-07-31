#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import math
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats
from scipy.stats import johnsonsu, loglaplace, norm, t
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from tools.models_export.residual_distribution_model import (
    CompositeDistribution,
    NoiseDistributionType,
    SingleDistribution,
)
from tools.regression_models.advanced_residual_distributions import piecewise_distribution, truncated_distribution


def test__single_distribution__given_empty_distribution_string__then_raise():
    with pytest.raises(ValidationError, match="ensure this value has at least 1 characters"):
        SingleDistribution(name="", parameters=[1.0])


def test__single_distribution__given_empty_parameter_list__then_raise():
    with pytest.raises(ValidationError, match="ensure this value has at least 1 items"):
        SingleDistribution(name="t", parameters=[])


def test__single_distribution__given_non_float_parameters__then_raise():
    with pytest.raises(ValidationError, match="value is not a valid float"):
        SingleDistribution(name="t", parameters=["one"])


def test__single_distribution__given_invalid_truncation__then_raise():
    with pytest.raises(ValidationError, match="Lower bound 1.0 needs to be smaller than upper bound 0.5"):
        SingleDistribution(name="t", parameters=[1.0], lower_bound=1.0, upper_bound=0.5)


def test__single_distribution__given_probability_mass_without_truncation__then_raise():
    with pytest.raises(
        ValidationError, match="Distribution needs to be truncated if probability mass is not equal to 1."
    ):
        SingleDistribution(name="t", parameters=[1.0], probability_mass=0.5)


@pytest.mark.parametrize("probability_mass", [0.0, -1.0, 3.0])
def test__single_distribution__given_invalid_probability_mass__then_raise(probability_mass: float):
    with pytest.raises(ValidationError, match="ensure this value is"):
        SingleDistribution(name="t", parameters=[1.0], lower_bound=0.0, probability_mass=probability_mass)


def test__single_distribution__given_parameter_which_is_not_defined__then_raise():
    with pytest.raises(
        ValueError,
        match="extra fields not permitted",
    ):
        SingleDistribution(name="t", parameters=[1.0], my_parameter=1.0)


@pytest.mark.parametrize(
    ["input_dict", "is_truncated"],
    [
        (dict(name="t", parameters=[3.18, 0.03, 0.29]), False),
        (dict(name="t", parameters=[1.0], upper_bound=10.0), True),
        (dict(name="t", parameters=[1.0], lower_bound=10.0), True),
        (dict(name="t", parameters=[1.0], lower_bound=10.0, upper_bound=20.0), True),
        (dict(name="t", parameters=[1.0], lower_bound=10.0, upper_bound=20.0, probability_mass=0.5), True),
    ],
)
def test__single_distribution__from_dict(input_dict: Dict[str, Any], is_truncated: bool):
    unit = SingleDistribution.parse_obj(input_dict)

    assert unit.name == input_dict.get("name")
    assert unit.base_scipy_distribution == stats.t
    assert unit.parameters == input_dict.get("parameters")
    assert math.isclose(unit.lower_bound, input_dict.get("lower_bound", -math.inf))
    assert math.isclose(unit.upper_bound, input_dict.get("upper_bound", math.inf))
    assert math.isclose(unit.probability_mass, input_dict.get("probability_mass", 1.0))
    assert unit.is_truncated == is_truncated


@pytest.mark.parametrize(
    ["input_dict", "expected_quantile_values"],
    [
        (
            dict(name="norm", parameters=[2.0, 0.5]),
            [norm.ppf(0.25, 2.0, 0.5), norm.ppf(0.5, 2.0, 0.5), norm.ppf(0.75, 2.0, 0.5)],
        ),
        (
            dict(name="t", parameters=[4, 2.0, 0.5], upper_bound=4.0),
            [
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.25 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.5 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.75 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
            ],
        ),
        (
            dict(name="t", parameters=[4, 2.0, 0.5], lower_bound=0.0, upper_bound=4.0),
            [
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.25 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.5 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.75 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
            ],
        ),
    ],
)
def test__single_distribution_frozen(input_dict: Dict[str, Any], expected_quantile_values: List[float]):
    unit = SingleDistribution.parse_obj(input_dict)
    frozen_distribution = unit.frozen

    assert type(frozen_distribution) == rv_continuous_frozen
    assert math.isclose(frozen_distribution.ppf(0.25), expected_quantile_values[0])
    assert math.isclose(frozen_distribution.ppf(0.5), expected_quantile_values[1])
    assert math.isclose(frozen_distribution.ppf(0.75), expected_quantile_values[2])


@pytest.mark.parametrize("probability_masses", [[0.1, 0.3], [0.3, 0.9]])
def test__composite_distribution__given_invalid_probability_masses__then_raise(probability_masses: List[float]):
    single_distributions = [
        SingleDistribution(name="t", parameters=[1.0], lower_bound=i, probability_mass=probability_mass)
        for i, probability_mass in enumerate(probability_masses)
    ]
    with pytest.raises(ValidationError, match="Probability masses don't sum up to 1."):
        CompositeDistribution(distributions=single_distributions)


def test__composite_distribution__given_too_many_distributions__then_raise():
    single_distributions = [
        SingleDistribution(name="t", parameters=[1.0], lower_bound=i, probability_mass=1 / 3) for i in range(3)
    ]
    with pytest.raises(ValidationError, match="ensure this value has at most 2 items"):
        CompositeDistribution(distributions=single_distributions)


def test__composite_distribution__given_no_distributions__then_raise():
    with pytest.raises(ValidationError, match="ensure this value has at least 1 items"):
        CompositeDistribution(distributions=[])


def test__composite_distribution__given_incorrect_bounds__then_raise():
    single_distributions = [
        SingleDistribution(name="t", parameters=[1.0], upper_bound=3.0, probability_mass=1 / 2),
        SingleDistribution(name="t", parameters=[1.0], lower_bound=1.0, probability_mass=1 / 2),
    ]
    with pytest.raises(
        ValidationError,
        match="The upper bound of the distributions need to coincide with the lower bound of the subsequent distribution.",
    ):
        CompositeDistribution(distributions=single_distributions)


def test__composite_distribution__given_lower_bound__then_raise():
    single_distributions = [
        SingleDistribution(name="t", parameters=[1.0], lower_bound=1.0, upper_bound=3.0, probability_mass=1 / 2),
        SingleDistribution(name="t", parameters=[1.0], lower_bound=3.0, probability_mass=1 / 2),
    ]
    with pytest.raises(
        ValidationError,
        match="The support of a piecewise distribution needs to be all real numbers.",
    ):
        CompositeDistribution(distributions=single_distributions)


def test__composite_distribution__given_upper_bound__then_raise():
    single_distributions = [
        SingleDistribution(name="t", parameters=[1.0], upper_bound=3.0, probability_mass=1 / 2),
        SingleDistribution(name="t", parameters=[1.0], lower_bound=3.0, upper_bound=5.0, probability_mass=1 / 2),
    ]
    with pytest.raises(
        ValidationError,
        match="The support of a piecewise distribution needs to be all real numbers.",
    ):
        CompositeDistribution(distributions=single_distributions)


@pytest.mark.parametrize(
    ["single_distribution", "expected_type"],
    [
        (SingleDistribution(name="t", parameters=[1.0]), "standard"),
        (SingleDistribution(name="t", parameters=[1.0], lower_bound=1.0), "truncated"),
        (SingleDistribution(name="t", parameters=[1.0], upper_bound=1.0), "truncated"),
        (SingleDistribution(name="t", parameters=[1.0], lower_bound=0.0, upper_bound=1.0), "truncated"),
    ],
)
def test__composite_distribution__thresholds_called_incorrectly_then_raise(
    single_distribution: SingleDistribution, expected_type: str
):
    unit = CompositeDistribution(distributions=[single_distribution])
    with pytest.raises(
        ValueError,
        match=f"{expected_type} distribution does not have thresholds.",
    ):
        _ = unit.thresholds


def test__composite_distribution__given_parameter_which_is_not_defined__then_raise():
    with pytest.raises(
        ValueError,
        match="extra fields not permitted",
    ):
        CompositeDistribution(distributions=[SingleDistribution(name="t", parameters=[1.0])], my_parameter=1.0)


@pytest.mark.parametrize(
    ["single_distributions", "expected_type", "expected_thresholds", "expected_parameters"],
    [
        ([SingleDistribution(name="t", parameters=[1.0])], NoiseDistributionType.STANDARD, None, [1.0]),
        (
            [SingleDistribution(name="t", parameters=[1.0, 2.0], lower_bound=1.0)],
            NoiseDistributionType.TRUNCATED,
            None,
            [1.0, 2.0],
        ),
        (
            [SingleDistribution(name="t", parameters=[1.0, 2.0], upper_bound=1.0)],
            NoiseDistributionType.TRUNCATED,
            None,
            [1.0, 2.0],
        ),
        (
            [SingleDistribution(name="t", parameters=[1.0, 2.0], lower_bound=0.0, upper_bound=1.0)],
            NoiseDistributionType.TRUNCATED,
            None,
            [1.0, 2.0],
        ),
        (
            [
                SingleDistribution(name="t", parameters=[1.0, 2.0], upper_bound=-1.0, probability_mass=0.3),
                SingleDistribution(name="norm", parameters=[3.0, 4.0], lower_bound=-1.0, probability_mass=0.7),
            ],
            NoiseDistributionType.PIECEWISE,
            [-1.0],
            [1.0, 2.0, 3.0, 4.0, 0.3],
        ),
    ],
)
def test__composite_distribution__from_dict(
    single_distributions: List[SingleDistribution],
    expected_type: NoiseDistributionType,
    expected_thresholds: Optional[List[float]],
    expected_parameters: List[float],
):
    ...
    unit: CompositeDistribution = CompositeDistribution.parse_obj({"distributions": single_distributions})

    assert unit.distributions == single_distributions
    assert unit.type == expected_type
    if expected_thresholds is not None:
        assert unit.thresholds == expected_thresholds
    assert unit.parameters == expected_parameters


@pytest.mark.parametrize(
    ["single_distributions", "expected_quantile_values"],
    [
        (
            [SingleDistribution(name="norm", parameters=[2.0, 0.5])],
            [norm.ppf(0.25, 2.0, 0.5), norm.ppf(0.5, 2.0, 0.5), norm.ppf(0.75, 2.0, 0.5)],
        ),
        (
            [SingleDistribution(name="t", parameters=[4, 2.0, 0.5], upper_bound=4.0)],
            [
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.25 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.5 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5) + 0.75 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
            ],
        ),
        (
            [SingleDistribution(name="t", parameters=[4, 2.0, 0.5], lower_bound=0.0, upper_bound=4.0)],
            [
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.25 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.5 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(-0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                t.ppf(
                    t.cdf(0.0, 4, 2.0, 0.5) + 0.75 * (t.cdf(4.0, 4, 2.0, 0.5) - t.cdf(0.0, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
            ],
        ),
        (
            [
                SingleDistribution(name="t", parameters=[4, 2.0, 0.5], upper_bound=-1.0, probability_mass=0.3),
                SingleDistribution(name="norm", parameters=[3.0, 4.0], lower_bound=-1.0, probability_mass=0.7),
            ],
            [
                t.ppf(
                    t.cdf(-math.inf, 4, 2.0, 0.5)
                    + 0.25 / 0.3 * (t.cdf(-1.0, 4, 2.0, 0.5) - t.cdf(-math.inf, 4, 2.0, 0.5)),
                    4,
                    2.0,
                    0.5,
                ),
                norm.ppf(
                    norm.cdf(-1.0, 3.0, 4.0)
                    + (0.5 - 0.3) / (1 - 0.3) * (norm.cdf(math.inf, 3.0, 4.0) - norm.cdf(-1.0, 3.0, 4.0)),
                    3.0,
                    4.0,
                ),
                norm.ppf(
                    norm.cdf(-1.0, 3.0, 4.0)
                    + (0.75 - 0.3) / (1 - 0.3) * (norm.cdf(math.inf, 3.0, 4.0) - norm.cdf(-1.0, 3.0, 4.0)),
                    3.0,
                    4.0,
                ),
            ],
        ),
    ],
)
def test__composite_distribution_frozen(
    single_distributions: List[SingleDistribution], expected_quantile_values: List[float]
):
    unit = CompositeDistribution.parse_obj({"distributions": single_distributions})
    frozen_distribution = unit.frozen

    assert type(frozen_distribution) == rv_continuous_frozen
    assert math.isclose(frozen_distribution.ppf(0.25), expected_quantile_values[0])
    assert math.isclose(frozen_distribution.ppf(0.5), expected_quantile_values[1])
    assert math.isclose(frozen_distribution.ppf(0.75), expected_quantile_values[2])


@pytest.mark.parametrize(
    ["distribution", "expected_name", "expected_parameters"],
    [(norm(1.1, 2.3), "norm", [1.1, 2.3]), (johnsonsu(0.3, 0.5, 0.7, -3.6), "johnsonsu", [0.3, 0.5, 0.7, -3.6])],
)
def test__composite_distribution_from_standard_scipy_distribution(
    distribution: rv_continuous_frozen, expected_name: str, expected_parameters: List[float]
):
    unit = CompositeDistribution.from_scipy_distribution(distribution=distribution)
    assert len(unit.distributions) == 1
    actual_distribution = unit.distributions[0]
    assert actual_distribution.name == expected_name
    assert actual_distribution.parameters == expected_parameters
    assert actual_distribution.lower_bound == -np.inf
    assert actual_distribution.upper_bound == np.inf
    assert actual_distribution.probability_mass == 1


def truncated_distribution_assertions(
    actual_distribution: SingleDistribution,
    expected_distribution: truncated_distribution,
    expected_parameters: List[float],
    expected_probability_mass: float,
) -> None:
    assert actual_distribution.name == expected_distribution.dist.name
    assert actual_distribution.parameters == expected_parameters
    assert actual_distribution.lower_bound == expected_distribution.lower_bound
    assert actual_distribution.upper_bound == expected_distribution.upper_bound
    assert actual_distribution.probability_mass == expected_probability_mass


@pytest.mark.parametrize(
    ["distribution", "expected_parameters"],
    [
        (
            truncated_distribution(dist=norm, lower_bound=10.0, upper_bound=15.2),
            [1.1, 2.3],
        ),
        (truncated_distribution(dist=t, lower_bound=-3.2), [-2.3, 1.5, 6.1]),
        (truncated_distribution(dist=johnsonsu, upper_bound=-3.2), [-2.3, 1.5, 6.1]),
    ],
)
def test__composite_distribution_from_truncated_distribution(
    distribution: truncated_distribution,
    expected_parameters: List[float],
):
    frozen_dist = distribution(*expected_parameters)
    unit = CompositeDistribution.from_scipy_distribution(distribution=frozen_dist)
    assert len(unit.distributions) == 1
    actual_distribution = unit.distributions[0]
    truncated_distribution_assertions(
        actual_distribution=actual_distribution,
        expected_distribution=distribution,
        expected_parameters=expected_parameters,
        expected_probability_mass=1.0,
    )


@pytest.mark.parametrize(
    [
        "distribution",
        "expected_lower_dist",
        "expected_lower_parameters",
        "expected_upper_dist",
        "expected_upper_parameters",
        "expected_ltmass",
    ],
    [
        (
            piecewise_distribution(lower_dist=norm, upper_dist=johnsonsu, threshold=3.0)(
                1.1, 2.3, 0.3, 0.5, 0.7, -3.6, 0.3
            ),
            truncated_distribution(dist=norm, upper_bound=3.0),
            [1.1, 2.3],
            truncated_distribution(dist=johnsonsu, lower_bound=3.0),
            [0.3, 0.5, 0.7, -3.6],
            0.3,
        ),
        (
            piecewise_distribution(lower_dist=loglaplace, upper_dist=t, threshold=-5.0)(
                0.1, -2.0, 1.2, 0.3, 0.5, 0.7, 0.8
            ),
            truncated_distribution(dist=loglaplace, upper_bound=-5.0),
            [0.1, -2.0, 1.2],
            truncated_distribution(dist=t, lower_bound=-5.0),
            [0.3, 0.5, 0.7],
            0.8,
        ),
    ],
)
def test__composite_distribution_from_piecewise_distribution(
    distribution: rv_continuous_frozen,
    expected_lower_dist: truncated_distribution,
    expected_lower_parameters: List[float],
    expected_upper_dist: truncated_distribution,
    expected_upper_parameters: List[float],
    expected_ltmass: float,
):
    unit = CompositeDistribution.from_scipy_distribution(distribution=distribution)
    assert len(unit.distributions) == 2
    actual_lower_dist = unit.distributions[0]
    actual_upper_dist = unit.distributions[1]
    truncated_distribution_assertions(
        actual_distribution=actual_lower_dist,
        expected_distribution=expected_lower_dist,
        expected_parameters=expected_lower_parameters,
        expected_probability_mass=expected_ltmass,
    )
    truncated_distribution_assertions(
        actual_distribution=actual_upper_dist,
        expected_distribution=expected_upper_dist,
        expected_parameters=expected_upper_parameters,
        expected_probability_mass=1 - expected_ltmass,
    )
