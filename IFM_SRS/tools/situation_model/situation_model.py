#
# Copyright (C) 2022-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from itertools import combinations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, conlist, fields, root_validator, validator
from tools.general_utility.helpers import TIMESTAMP_FORMAT, validate_timestamp_format


class ModelTimestamps(BaseModel):
    timestamp_input_data: Optional[str] = None
    timestamp_export: Optional[str] = None

    @validator("timestamp_input_data", "timestamp_export")
    def timestamps_in_correct_format(cls, timestamp: Optional[str], field: fields.ModelField) -> Optional[str]:
        if not validate_timestamp_format(timestamp):
            raise ValueError(f"{field.name}: {timestamp} is not in the format {TIMESTAMP_FORMAT}")
        return timestamp


class CorrelationMatrix(ModelTimestamps):
    parameter_order: conlist(str, min_items=2, unique_items=True)
    matrix_rows: List[List[float]]

    @validator("matrix_rows")
    def matrix_must_be_non_empty(cls, v: List[List[float]]) -> List[List[float]]:
        matrix_rows_lengths = set(map(len, v))
        if not (len(matrix_rows_lengths) == 1 and len(v) == len(v[0])):
            raise ValueError("invalid correlation matrix")
        return v

    @root_validator
    def parameter_order_length_must_fit_matrix(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not "parameter_order" in values or not "matrix_rows" in values:
            raise ValueError("subvalidators failed")

        if len(values["parameter_order"]) != len(values["matrix_rows"][0]):
            raise ValueError("invalid parameter order")

        return values

    @property
    def size(self) -> int:
        return len(self.parameter_order)


class MarginalDistributionTestCase(BaseModel):
    u_noise: conlist(float, min_items=1)
    y_predict: conlist(float, min_items=1)

    @root_validator
    def parameter_order_length_must_fit_matrix(cls, values: Dict[str, List[float]]) -> Dict[str, List[float]]:
        if len(values["u_noise"]) != len(values["y_predict"]):
            raise ValueError("invalid marginal distribution test case")
        return values


class SituationModelParameter(BaseModel):
    name: str
    transformation_function: str
    marginal_distribution: str
    marginal_distribution_parameters: conlist(float, min_items=1)
    marginal_distribution_test_cases: MarginalDistributionTestCase
    codebeamer_reference: str


class SituationModel(ModelTimestamps):
    parameters: List[SituationModelParameter]
    correlation_matrix: CorrelationMatrix
    scenario_occurrence_rate_per_hour: float
    filtering_function: str

    @validator("parameters")
    def two_parameters_must_not_have_equal_marginal_distribution_parameters(
        cls, v: List[SituationModelParameter]
    ) -> List[SituationModelParameter]:
        for lhs, rhs in combinations(v, 2):
            if lhs.marginal_distribution == rhs.marginal_distribution:
                if lhs.marginal_distribution_parameters == rhs.marginal_distribution_parameters:
                    raise ValueError(
                        f"Parameters {lhs.name} and {rhs.name} must not have equal marginal distribution parameters"
                    )

        return v

    @root_validator
    def number_of_parameters_must_fit_correlation_matrix_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not "parameters" in values or not "correlation_matrix" in values:
            raise ValueError("subvalidators failed")

        number_of_parameters = len(values["parameters"])
        correlation_matrix_size = values["correlation_matrix"].size
        if number_of_parameters != correlation_matrix_size:
            raise ValueError(
                f"Number of parameters ({number_of_parameters}) does not match correlation matrix size ({correlation_matrix_size})"
            )

        return values


class ConditionalSituationModel(ModelTimestamps):
    conditional_distribution: str
    models: List[SituationModel]
    model_proportions: conlist(float, min_items=2)
    conditional_distribution_states: conlist(str, min_items=2)


if __name__ == "__main__":
    print(SituationModel.schema_json(indent=4))
