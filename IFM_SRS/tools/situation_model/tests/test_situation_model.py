#
# Copyright (C) 2022-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import Any, Dict

import pydantic
import pytest
from tools.conftest import TEST_TIMESTAMPS
from tools.situation_model.situation_model import (
    ConditionalSituationModel,
    CorrelationMatrix,
    MarginalDistributionTestCase,
    SituationModel,
    SituationModelParameter,
)


def get_valid_situation_model_entries() -> Dict[str, Any]:
    dummy_test_cases = MarginalDistributionTestCase(u_noise=[0.5], y_predict=[2.0])
    return {
        "correlation_matrix": CorrelationMatrix(
            parameter_order=["Initial Host Velocity", "Duration"],
            matrix_rows=[
                [1.0, -0.0728023],
                [-0.0728023, 1.0],
            ],
        ),
        "filtering_function": "initial_host_velocity > 0",
        "parameters": [
            SituationModelParameter(
                name="initial_host_velocity",
                codebeamer_reference="123",
                marginal_distribution="Kernel density",
                marginal_distribution_parameters=[1, 2, 3, 5, 6],
                marginal_distribution_test_cases=dummy_test_cases,
                transformation_function="x",
            ),
            SituationModelParameter(
                name="duration",
                codebeamer_reference="123",
                marginal_distribution="genextreme",
                marginal_distribution_parameters=[1, 2, 3],
                marginal_distribution_test_cases=dummy_test_cases,
                transformation_function="x",
            ),
        ],
        "scenario_occurrence_rate_per_hour": 1.0,
        **TEST_TIMESTAMPS,
    }


def get_valid_conditional_situation_model_entries() -> Dict[str, Any]:
    return {
        "conditional_distribution": "vehicle_type",
        "conditional_distribution_states": ["Passenger", "Car", "Truck"],
        "model_proportions": [0.09705042816365367, 1 - 0.9149434022639095],
        **TEST_TIMESTAMPS,
        "models": [
            SituationModel(
                correlation_matrix=CorrelationMatrix(
                    parameter_order=[
                        "Initial Host Velocity",
                        "(virtual) TTC",
                        "Duration",
                        "relative Velocity (Cut-Out)",
                    ],
                    matrix_rows=[
                        [1.0, -0.0728023, -0.29714534, 0.62930377],
                        [-0.0728023, 1.0, -0.05373979, -0.23335824],
                        [-0.29714534, -0.05373979, 1.0, -0.23100442],
                        [0.62930377, -0.23335824, -0.23100442, 1.0],
                    ],
                ),
                filtering_function="initial_host_velocity > 0 & initial_host_velocity < 60/3.6, duration < 30 and duration > 0 , vRel >= initial_host_velocity,TTC > 0",
                parameters=[
                    SituationModelParameter(
                        name="initial_host_velocity",
                        codebeamer_reference="123",
                        marginal_distribution="Kernel density",
                        marginal_distribution_parameters=[1, 2, 3, 5, 6],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="virtual_ttc",
                        codebeamer_reference="123",
                        marginal_distribution="Kernel density",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="duration",
                        codebeamer_reference="123",
                        marginal_distribution="genextreme",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="relative_velocity",
                        codebeamer_reference="123",
                        marginal_distribution="t",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                ],
                scenario_occurrence_rate_per_hour=0.5473852452735175,
            ),
            SituationModel(
                correlation_matrix=CorrelationMatrix(
                    parameter_order=[
                        "Initial Host Velocity",
                        "(virtual) TTC",
                        "Duration",
                        "relative Velocity (Cut-Out)",
                    ],
                    matrix_rows=[
                        [1.0, -0.0728023, -0.29714534, 0.62930377],
                        [-0.0728023, 1.0, -0.05373979, -0.23335824],
                        [-0.29714534, -0.05373979, 1.0, -0.23100442],
                        [0.62930377, -0.23335824, -0.23100442, 1.0],
                    ],
                ),
                filtering_function="initial_host_velocity > 0 & initial_host_velocity < 60/3.6, duration < 30 and duration > 0 , vRel >= initial_host_velocity,TTC > 0",
                parameters=[
                    SituationModelParameter(
                        name="initial_host_velocity",
                        codebeamer_reference="123",
                        marginal_distribution="Kernel density",
                        marginal_distribution_parameters=[1, 2, 3, 4, 5, 6],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="virtual_ttc",
                        codebeamer_reference="123",
                        marginal_distribution="Kernel density",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="duration",
                        codebeamer_reference="123",
                        marginal_distribution="genextreme",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                    SituationModelParameter(
                        name="relative_velocity",
                        codebeamer_reference="123",
                        marginal_distribution="t",
                        marginal_distribution_parameters=[1, 2, 3],
                        marginal_distribution_test_cases=MarginalDistributionTestCase(
                            u_noise=[0.9537625822517408, 0.9921264707372405, 0.4796029245278556],
                            y_predict=[15.906343761738382, 16.732472226444035, 8.597356259522341],
                        ),
                        transformation_function="x",
                    ),
                ],
                scenario_occurrence_rate_per_hour=0.5473852452735175,
            ),
        ],
    }


def test_valid_conditional_situation_model():
    ConditionalSituationModel(**get_valid_conditional_situation_model_entries())


@pytest.mark.parametrize("timestamp_param", ["timestamp_input_data", "timestamp_export"])
def test__raise_when_timestamps_wrongly_formatted_in_conditional_situation_model(timestamp_param: str):
    entries = get_valid_conditional_situation_model_entries()
    entries[timestamp_param] = "01.01.2001"
    with pytest.raises(
        pydantic.error_wrappers.ValidationError, match=re.escape("01.01.2001 is not in the format %d/%m/%Y %H:%M:%S")
    ):
        ConditionalSituationModel(**entries)


def test_valid_situation_model():
    SituationModel(**get_valid_situation_model_entries())


def test_valid_correlation_matrix():
    unit = CorrelationMatrix(
        parameter_order=["A", "B", "C", "D"], matrix_rows=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    )

    assert unit.size == 4


def test_valid_correlation_matrix_with_timestamps():
    unit = CorrelationMatrix(
        parameter_order=["A", "B", "C", "D"],
        matrix_rows=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        **TEST_TIMESTAMPS,
    )

    assert unit.size == 4


@pytest.mark.parametrize("timestamp_param", ["timestamp_input_data", "timestamp_export"])
def test_correlation_matrix_raises_for_incorrect_timestamp_format(timestamp_param: str):
    input_dict = {
        "parameter_order": ["A", "B", "C", "D"],
        "matrix_rows": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        timestamp_param: "01.01.2001",
    }
    with pytest.raises(
        pydantic.error_wrappers.ValidationError, match=re.escape("01.01.2001 is not in the format %d/%m/%Y %H:%M:%S")
    ):
        CorrelationMatrix(**input_dict)


def test_invalid_correlation_matrix_bad_parameter_order():
    with pytest.raises(ValueError):
        CorrelationMatrix(
            parameter_order=["A", "B", "C"], matrix_rows=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        )


def test_invalid_correlation_matrix_bad_matrix_rows():
    with pytest.raises(ValueError):
        CorrelationMatrix(
            parameter_order=["A", "B", "C", "D"],
            matrix_rows=[
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3],
            ],
        )


def test_invalid_correlation_matrix_not_square():
    with pytest.raises(ValueError):
        CorrelationMatrix(
            parameter_order=["A", "B", "C", "D"],
            matrix_rows=[
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
        )


def test_invalid_marginal_distribution_test_case_bad_lengths():
    with pytest.raises(ValueError):
        MarginalDistributionTestCase(
            u_noise=[1, 2, 3],
            y_predict=[1, 2, 3, 4],
        )


def test_invalid_situation_model_number_of_parameters_does_not_match_correlation_matrix_size():
    dummy_test_cases = MarginalDistributionTestCase(u_noise=[0.5], y_predict=[2.0])
    with pytest.raises(ValueError) as exc:
        SituationModel(
            correlation_matrix=CorrelationMatrix(
                parameter_order=["Initial Host Velocity", "Duration", "3rd parameter"],
                matrix_rows=[
                    [1.0, -0.0728023, -0.29714534],
                    [-0.0728023, 1.0, -0.05373979],
                    [-0.29714534, -0.05373979, 1.0],
                ],
            ),
            filtering_function="initial_host_velocity > 0",
            parameters=[
                SituationModelParameter(
                    name="initial_host_velocity",
                    codebeamer_reference="123",
                    marginal_distribution="Kernel density",
                    marginal_distribution_parameters=[1, 2, 3, 5, 6],
                    marginal_distribution_test_cases=dummy_test_cases,
                    transformation_function="x",
                ),
                SituationModelParameter(
                    name="duration",
                    codebeamer_reference="123",
                    marginal_distribution="genextreme",
                    marginal_distribution_parameters=[1, 2, 3],
                    marginal_distribution_test_cases=dummy_test_cases,
                    transformation_function="x",
                ),
            ],
            scenario_occurrence_rate_per_hour=1.0,
        )

    assert "Number of parameters (2) does not match correlation matrix size (3)" == exc.value.errors()[0]["msg"]


@pytest.mark.parametrize("timestamp_param", ["timestamp_input_data", "timestamp_export"])
def test_raise_when_timestamps_wrongly_formatted_in_situation_model(timestamp_param: str):
    entries = get_valid_situation_model_entries()
    entries[timestamp_param] = "01.01.2001"
    with pytest.raises(
        pydantic.error_wrappers.ValidationError, match=re.escape("01.01.2001 is not in the format %d/%m/%Y %H:%M:%S")
    ):
        SituationModel(**entries)


def test_invalid_situation_model_two_parameters_must_not_have_equal_marginal_distribution_parameters():
    dummy_test_cases = MarginalDistributionTestCase(u_noise=[0.5], y_predict=[2.0])
    marginal_distribution = "genextreme"
    marginal_distribution_parameters = [1, 2, 3]
    with pytest.raises(ValueError) as exc:
        SituationModel(
            correlation_matrix=CorrelationMatrix(
                parameter_order=["Initial Host Velocity", "Duration"],
                matrix_rows=[[1.0, -0.0728023], [1 - 0.0728023, 1.0]],
            ),
            filtering_function="initial_host_velocity > 0",
            parameters=[
                SituationModelParameter(
                    name="initial_host_velocity",
                    codebeamer_reference="123",
                    marginal_distribution=marginal_distribution,
                    marginal_distribution_parameters=marginal_distribution_parameters,
                    marginal_distribution_test_cases=dummy_test_cases,
                    transformation_function="x",
                ),
                SituationModelParameter(
                    name="duration",
                    codebeamer_reference="123",
                    marginal_distribution=marginal_distribution,
                    marginal_distribution_parameters=marginal_distribution_parameters,
                    marginal_distribution_test_cases=dummy_test_cases,
                    transformation_function="x",
                ),
            ],
            scenario_occurrence_rate_per_hour=1.0,
        )

    assert (
        "Parameters initial_host_velocity and duration must not have equal marginal distribution parameters"
        == exc.value.errors()[0]["msg"]
    )
