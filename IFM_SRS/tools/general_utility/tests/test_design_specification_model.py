#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

# From IFAT

import json
from os.path import dirname, join

import pydantic
import pytest
from tools.general_utility.design_specification_model import DesignSpecificationModel

DESIGN_SPECIFICATION_PATH = join(
    dirname(__file__), "test_data", "design_specification", "valid_design_specification.json"
)


def test__valid_input():
    with open(DESIGN_SPECIFICATION_PATH, "r") as input_stream:
        _ = DesignSpecificationModel(**json.loads(input_stream.read()))


class TestInvalidInput:
    def test__missing_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        del design_specification_dict["info"]
        self.template("field required", **design_specification_dict)

    def test__additional_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        design_specification_dict["additional_field"] = ""
        self.template("extra fields not permitted", **design_specification_dict)

    def test__ifv_missing_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        del design_specification_dict["ifv"]["8517320"]["cb_name"]
        self.template("field required", **design_specification_dict)

    def test__spv_missing_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        del design_specification_dict["spv"][0]["kpi_table"]
        self.template("field required", **design_specification_dict)

    def test__spv_additional_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        design_specification_dict["spv"][0]["additional_field"] = ""
        self.template("Unexpected field in SPV entry: additional_field", **design_specification_dict)

    def test__marker_additional_field(self):
        design_specification_dict = self.__get_valid_design_specification()
        design_specification_dict["test_matrix"][0]["marker"]["additional_marker"] = ""
        self.template("extra fields not permitted", **design_specification_dict)

    def test__test_matrix_row_missing_key(self):
        design_specification_dict = self.__get_valid_design_specification()
        del design_specification_dict["test_matrix"][0]["number_of_measurements"]
        self.template("Not all rows of test_matrix have the same columns defined.", **design_specification_dict)

    def test__test_matrix_unexpected_ind_factor(self):
        design_specification_dict = self.__get_valid_design_specification()
        del design_specification_dict["ifv"]["8517334"]
        self.template("Unexpected set of ind_factors for scenario", **design_specification_dict)

    def test__test_matrix_unexpected_ind_factor_value(self):
        design_specification_dict = self.__get_valid_design_specification()
        design_specification_dict["test_matrix"][0]["ind_factor"]["8474898"] = "dog"
        self.template("Value dog for ind_factor 8474898", **design_specification_dict)

    def test__ifv_whitespace(self):
        design_specification_dict = self.__get_valid_design_specification()
        design_specification_dict["ifv"]["8517320"]["cb_name"] = " dog"
        self.template("Unnecessary whitespaces in entry  dog", **design_specification_dict)

    def template(self, err_string, **kwargs):
        with pytest.raises(pydantic.error_wrappers.ValidationError, match=err_string) as err:
            DesignSpecificationModel(**kwargs)

    def __get_valid_design_specification(self):
        with open(DESIGN_SPECIFICATION_PATH, "r") as input_stream:
            return json.loads(input_stream.read())
