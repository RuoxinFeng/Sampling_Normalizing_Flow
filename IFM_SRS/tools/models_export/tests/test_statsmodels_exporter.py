#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import os
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.stats import norm
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tools.conftest import TEST_TIMESTAMPS
from tools.general_utility.helpers import round_output
from tools.models_export.residual_distribution_model import CompositeDistribution, SingleDistribution
from tools.models_export.statsmodels_exporter import SpmModel
from tools.test_helpers.file_creation_helpers import assert_file_content, assert_json_equal


@pytest.fixture
def model_data_dict() -> Dict[str, Any]:
    return dict(
        name="test_model",
        model_class="my_class",
        params={"Intercept": 2.5 + 1e-5 + 1e-11, "a": 0.1 + 1e-5 + 1e-11, "b": 0.2 + 1e-5 + 1e-11},
        dependent_variable="y",
        mse_residuals=0.25 + 1e-5 + 1e-11,
        custom_noise_distribution=CompositeDistribution(
            distributions=[SingleDistribution(name="norm", parameters=[0, 1])]
        ),
        total_failure_rate=0.5 + 1e-5 + 1e-11,
        **TEST_TIMESTAMPS,
    )


def get_fake_model(coef_intercept: float = 1.0, coef_a: float = 1.0, coef_b: float = 1.0) -> RegressionResultsWrapper:
    a = np.array([1, 3, 3.3, 5, 8])
    da = np.array([0.0308021, -0.01696876, 0.07421984, 0.00710233, 0.03168263])
    b = np.array([1, 1, 1, 0, 0])
    data = pd.DataFrame({"a": a, "b": b, "y": coef_intercept + coef_a * a + da + coef_b * b})

    return ols(formula="y ~ a + b", data=data).fit()


def test__from_linear_model_result():
    coef_intercept = -2.5
    coef_a = 0.7
    coef_b = 20

    fake_model = get_fake_model(coef_intercept=coef_intercept, coef_a=coef_a, coef_b=coef_b)

    unit = SpmModel.from_linear_model_result(
        model_result=fake_model,
        name="test_model",
        total_failure_rate=0.5,
        **TEST_TIMESTAMPS,
        custom_noise_distribution=norm(0.1, 2.0),
    )

    assert unit.name == "test_model"
    assert unit.model_class == "<class 'statsmodels.regression.linear_model.OLS'>"
    assert unit.params["Intercept"] == pytest.approx(coef_intercept, rel=1e-2)
    assert unit.params["a"] == pytest.approx(coef_a, rel=1e-2)
    assert unit.params["b"] == pytest.approx(coef_b, rel=1e-2)
    assert unit.dependent_variable == "y"
    assert unit.total_failure_rate == 0.5
    assert unit.timestamp_input_data == TEST_TIMESTAMPS["timestamp_input_data"]
    assert unit.timestamp_export == TEST_TIMESTAMPS["timestamp_export"]
    assert 0 < unit.mse_residuals < 0.01
    assert len(unit.custom_noise_distribution.distributions) == 1
    custom_noise_distribution = unit.custom_noise_distribution.distributions[0]
    assert custom_noise_distribution.name == "norm"
    assert custom_noise_distribution.parameters == [0.1, 2.0]


def assert_noise_distribution_equal(
    expected_custom_distribution: CompositeDistribution, actual_distribution_json: Dict[str, Any]
) -> None:
    assert len(expected_custom_distribution.distributions) == len(actual_distribution_json["distributions"])
    for expected_dist, actual_dist in zip(
        expected_custom_distribution.distributions, actual_distribution_json["distributions"]
    ):
        assert expected_dist.name == actual_dist["name"]
        assert expected_dist.parameters == actual_dist["parameters"]
        assert expected_dist.lower_bound == actual_dist.get("lower_bound", -np.inf)
        assert expected_dist.upper_bound == actual_dist.get("upper_bound", np.inf)
        assert expected_dist.probability_mass == actual_dist.get("probability_mass", 1)


@pytest.mark.parametrize(
    "custom_noise_distribution",
    [
        CompositeDistribution(distributions=[SingleDistribution(name="norm", parameters=[0 + 1e-5, 1 + 1e-5])]),
        CompositeDistribution(
            distributions=[
                SingleDistribution(
                    name="norm", parameters=[0 + 1e-5, 1 + 1e-5], lower_bound=-3 + 1e-5, upper_bound=1.5 + 1e-5
                )
            ]
        ),
        CompositeDistribution(
            distributions=[
                SingleDistribution(
                    name="t", parameters=[1.0 + 1e-5, 2.0 + 1e-5], upper_bound=-1.0 + 1e-5, probability_mass=0.3 - 1e-5
                ),
                SingleDistribution(
                    name="norm",
                    parameters=[3.0 + 1e-5, 4.0 + 1e-5],
                    lower_bound=-1.0 + 1e-5,
                    probability_mass=0.7 + 1e-5,
                ),
            ]
        ),
    ],
)
@pytest.mark.parametrize("rounding_strategy", [{"nr_of_digits_after_comma": 4}, {"nr_of_digits_after_comma": None}, {}])
def test__as_json(
    tmpdir,
    model_data_dict: Dict[str, Any],
    rounding_strategy: Dict[str, Optional[int]],
    custom_noise_distribution: CompositeDistribution,
):
    model_data_dict["custom_noise_distribution"] = custom_noise_distribution
    unit = SpmModel(**model_data_dict)
    file_path = os.path.join(tmpdir, "model.json")
    unit.as_json(file_path=file_path, **rounding_strategy)

    assert_file_content(
        expected_content=model_data_dict,
        file_path=file_path,
        **rounding_strategy,
    )


@pytest.mark.parametrize(
    "custom_noise_distribution",
    [
        CompositeDistribution(distributions=[SingleDistribution(name="norm", parameters=[0 + 1e-5, 1 + 1e-5])]),
        CompositeDistribution(
            distributions=[
                SingleDistribution(
                    name="norm", parameters=[0 + 1e-5, 1 + 1e-5], lower_bound=-3 + 1e-5, upper_bound=1.5 + 1e-5
                )
            ]
        ),
        CompositeDistribution(
            distributions=[
                SingleDistribution(
                    name="t", parameters=[1.0 + 1e-5, 2.0 + 1e-5], upper_bound=-1.0 + 1e-5, probability_mass=0.3 - 1e-5
                ),
                SingleDistribution(
                    name="norm",
                    parameters=[3.0 + 1e-5, 4.0 + 1e-5],
                    lower_bound=-1.0 + 1e-5,
                    probability_mass=0.7 + 1e-5,
                ),
            ]
        ),
    ],
)
@pytest.mark.parametrize("rounding_strategy", [{"nr_of_digits_after_comma": 4}, {"nr_of_digits_after_comma": None}, {}])
def test__as_json_str(
    model_data_dict: Dict[str, Any],
    rounding_strategy: Dict[str, Optional[int]],
    custom_noise_distribution: CompositeDistribution,
):
    model_data_dict["custom_noise_distribution"] = custom_noise_distribution
    unit = SpmModel(**model_data_dict)
    json_data = json.loads(unit.as_json_str(**rounding_strategy))
    model_data_dict["custom_noise_distribution"] = CompositeDistribution(
        **round_output(model_data_dict["custom_noise_distribution"].dict(exclude_unset=True), **rounding_strategy)
    )

    assert_json_equal(actual=json_data, expected=model_data_dict, **rounding_strategy)


def test__given_wrong_statsmodels_version__then_raise():
    with mock.patch("tools.models_export.statsmodels_exporter.get_actual_statsmodels_version") as version_mock:
        version_mock.return_value = "0.13.0"

        with pytest.raises(RuntimeError) as err:
            SpmModel.from_linear_model_result(
                model_result=None,
                name="test_model",
                total_failure_rate=0.5,
                **TEST_TIMESTAMPS,
                custom_noise_distribution=norm(0.0, 1.0),
            )

        assert f"The class SpmModel was developed for statsmodels" in str(err)


@pytest.mark.parametrize("param_to_alter", [{"timestamp_input_data": "01.01.2001"}, {"timestamp_export": "01.01.2001"}])
def test_incorrect_timestamp_format(param_to_alter: Dict[str, str]):
    fake_model = get_fake_model()

    input_dict = {
        "model_result": fake_model,
        "name": "test_model",
        "total_failure_rate": 0.5,
        **TEST_TIMESTAMPS,
        "custom_noise_distribution": norm(0.0, 1.0),
    }
    input_dict.update(param_to_alter)
    with pytest.raises(ValidationError, match="is not in the format"):
        SpmModel.from_linear_model_result(**input_dict)
