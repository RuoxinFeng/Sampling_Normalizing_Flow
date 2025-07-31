#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import pandas as pd
import pytest
from tools.general_utility.extract_columns import get_columns_with_IFVs, get_columns_with_SPVs


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "some_unrelated_column": [1, 2, 3],
            "an_IFV_column_1234567": [1, 2, 3],
            "another_unrelated_12345678A_column": [1, 2, 3],
            "not_this_1234567one": [1, 2, 3],
            "this_IFV_12345678": [1, 2, 3],
            "an_SPV_1234567A": [1, 2, 3],
            "another_SPV_12345678F": [1, 2, 3],
            "lowercase_SPV_2345678b": [1, 2, 3],
            "w_o_underscore2345678": [1, 2, 3],
        }
    )


def test_get_columns_with_IFVs(dummy_df: pd.DataFrame):
    actual_columns = get_columns_with_IFVs(dummy_df)
    expected_columns = ["an_IFV_column_1234567", "this_IFV_12345678", "w_o_underscore2345678"]

    assert actual_columns == expected_columns


def test_get_columns_with_SPVs(dummy_df: pd.DataFrame):
    actual_columns = get_columns_with_SPVs(dummy_df)
    expected_columns = ["an_SPV_1234567A", "another_SPV_12345678F", "lowercase_SPV_2345678b"]

    assert actual_columns == expected_columns
