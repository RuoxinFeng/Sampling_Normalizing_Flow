#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import unittest
from typing import Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix
from scipy.stats import norm
from tools.general_utility.helpers import round_output
from tools.regression_models.regression_helpers import generate_test_data, predict_statsmodels
from tools.regression_models.regression_models import ConvenientOlsFit
from tools.test_helpers.rounding_helpers import assert_rounded


def test__predicts_as_expected(simple_setup: ConvenientOlsFit, simple_data: Tuple[pd.DataFrame, pd.Series]):
    U = simple_data[0]["U"].head(3)
    y = predict_statsmodels(
        model=simple_setup.result,
        X=simple_setup.X.head(3),
        U=U,
    )
    noise = norm.ppf(U, loc=0.0, scale=np.sqrt(simple_setup.result.mse_resid))
    assert np.array(y - noise) == pytest.approx(
        np.array(simple_data[0]["y_1234567a"].head(3) - simple_data[0]["U"].head(3)), 0.01
    )


def test_generate_test_data(simple_setup: ConvenientOlsFit):
    params = simple_setup.result.params
    input_data = {
        "x1_1234567": [0.0, 5.0],
        "x2_12345678": [1.0, -2.0],
        "x3_1234567a": [2.0, 3.0],
        "x4_12345678b": [3.0, -10.0],
    }
    X = dmatrix(
        "x1_1234567 + x2_12345678 + x3_1234567a + x4_12345678b", data=pd.DataFrame(input_data), return_type="dataframe"
    )
    actual_test_data = generate_test_data(
        model=simple_setup.result,
        input_data=input_data,
        dmatrix=X,
        seed=1,
        inv_trafo=simple_setup.inverse_output_trafo,
        noise_distribution=simple_setup.noise_dist,
        timestamp_input_data=simple_setup.timestamp_input_data,
        timestamp_export=simple_setup.timestamp_export,
    )
    expected_noise = [0.417022004702574, 0.7203244934421581]  # guaranteed because of seed
    expected_test_data = {
        "X_input": input_data,
        "U_noise": expected_noise,
        "Y_predict": [
            round_output(
                params[0]
                + params[1] * 0.0
                + params[2] * 1.0
                + params[3] * 2.0
                + params[4] * 3.0
                + norm.ppf(expected_noise[0], loc=0.0, scale=np.sqrt(simple_setup.result.mse_resid))
            ),
            round_output(
                params[0]
                + params[1] * 5.0
                + params[2] * (-2.0)
                + params[3] * 3.0
                + params[4] * (-10.0)
                + norm.ppf(expected_noise[1], loc=0.0, scale=np.sqrt(simple_setup.result.mse_resid))
            ),
        ],
        "timestamp_input_data": simple_setup.timestamp_input_data,
        "timestamp_export": simple_setup.timestamp_export,
    }
    test_case = unittest.TestCase()
    test_case.assertDictEqual(actual_test_data, expected_test_data)
    assert_rounded(actual_test_data["Y_predict"])


@pytest.mark.parametrize(
    ["input_data", "expected_error"],
    [
        ({"x_1": [1, 2, 3, 4, 5, 6], "x_2": [1, 2, 3, 4, 5]}, "Input data columns do not have the same length!"),
        (
            {"x_1": [1, 2, 3], "x_2": [1, 2, 5], "x_3": [4, 5, 6], "x_4": [1, 2, 3, 4, 5, 6]},
            "Input data columns do not have the same length!",
        ),
        ({"x_1": [1], "x_2": [2], "x_3": [3]}, "Define at least two test vectors!"),
        ({"x_1": [0]}, "Define at least two test vectors!"),
    ],
)
def test__generate_test_data_raises_for_invalid_input_data(input_data: Dict[str, List[float]], expected_error: str):
    with pytest.raises(ValueError, match=expected_error):
        generate_test_data(model=Mock(), input_data=input_data, dmatrix=Mock(), seed=1)
