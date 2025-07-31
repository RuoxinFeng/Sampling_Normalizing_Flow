#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

# From IFAT

from typing import Dict, List, Union

from pydantic import BaseModel, Extra, Field, root_validator, validator


class Info(BaseModel):
    date: str
    srs: str

    class Config:
        extra = Extra.allow


class IfvEntry(BaseModel):
    cb_name: str
    legacy_name: str
    unit: Union[None, str]
    tolerance: Union[None, int, float]

    class Config:
        extra = Extra.allow

    @validator("*")
    def check_no_whitespaces(cls, entry):
        if isinstance(entry, str):
            if entry != entry.strip():
                raise ValueError(f"Unnecessary whitespaces in entry {entry}")
        return entry


class SpvEntry(BaseModel):
    spv_cb_id: str
    spv_name: str
    kpi_table: str

    class Config:
        extra = Extra.allow

    @root_validator
    def check_additional_keys(cls, values):
        cb_id = values["spv_cb_id"]
        for key in values.keys():
            if not (key in cls.__fields__ or key.startswith(cb_id)):
                raise ValueError(f"Unexpected field in SPV entry: {key}")
        return values


class Marker(BaseModel):
    A: int = Field(..., ge=0)
    B: int = Field(..., ge=0)
    C: int = Field(..., ge=0)
    D: int = Field(..., ge=0)

    class Config:
        extra = Extra.forbid


class TestMatrixRow(BaseModel):
    marker: Marker
    ind_factor: dict
    is_worst_case: int = Field(..., ge=0, le=1)

    class Config:
        extra = Extra.allow


class DesignSpecificationModel(BaseModel):
    info: Info
    ifv: Dict[str, IfvEntry]
    spv: List[SpvEntry]
    test_matrix: List[TestMatrixRow]

    class Config:
        extra = Extra.forbid

    @validator("test_matrix")
    def compare_test_matrix_keys(cls, test_matrix, values):
        """
        Test if all rows of the test_matrix share the same keys (all columns have a value for each row).
        """
        if "info" in values.keys():
            srs = values["info"].srs
            keys = cls._generate_keys_set(cls, test_matrix[0])
            cls._compare_keys(cls, test_matrix, keys, srs)

        return test_matrix

    def _generate_keys_set(cls, row: Union[dict, TestMatrixRow]) -> set:
        """
        Outputs set of all keys in the lowest level of a nested dict.
        """
        output = set()
        e = row if isinstance(row, dict) else row.dict()
        for key, value in e.items():
            if isinstance(value, dict):
                output = output.union(cls._generate_keys_set(cls, value))
            else:
                output.add(key)
        return output

    def _compare_keys(cls, test_matrix, expected_keys, srs):
        for row in test_matrix:
            actual_keys = cls._generate_keys_set(cls, row)
            if actual_keys != expected_keys:
                raise ValueError(
                    f"""Error for {srs} decoder:
                Not all rows of test_matrix have the same columns defined.
                The first row has the keys {expected_keys} and the scenario
                {row.scenario_id['scenario_id']} has the keys {actual_keys}"""
                )

    @validator("test_matrix")
    def test_matrix_ind_factors(cls, test_matrix, values):
        """
        Test if all rows in test_matrix have exactly the ind_factors defined that are
        defined in the ifv.
        """
        if all(key in values.keys() for key in ["ifv", "info"]):
            ifv = values["ifv"]
            srs = values["info"].srs
            ind_factors = cls._get_ind_factors(ifv)
            for row in test_matrix:
                if set(row.ind_factor.keys()) != ind_factors:
                    raise ValueError(
                        f"""Error for {srs} decoder: Unexpected set of ind_factors for scenario {row.scenario_id['scenario_id']}.
                        Expected {ind_factors}, got {set(row.ind_factor.keys())}"""
                    )
        return test_matrix

    def _get_ind_factors(ifv: dict) -> set:
        """
        Get all allowed (and mandatory) ind_factors in test_matrix.
        This list includes all ind_factors defined in the ifv.
        Therefore, each entry of test matrix will have to exactly the
        ind_factors defined in the ifv.
        """

        return set(ifv.keys())

    @validator("test_matrix")
    def test_matrix_ind_factor_values(cls, test_matrix, values):
        """
        Test if all values set for ind_factors in the test_matrix are previously
        defined in the ifv dict.
        """
        if all(key in values.keys() for key in ["ifv", "info"]):
            ifv = values["ifv"]
            srs = values["info"].srs
            for row in test_matrix:
                for key, value in row.ind_factor.items():
                    if value not in ifv[key].dict().values():
                        raise ValueError(
                            f"""Error for {srs} decoder: Value {value} for ind_factor {key} in
                            scenario {row.scenario_id['scenario_id']} is not defined in ifv!"""
                        )
        return test_matrix
