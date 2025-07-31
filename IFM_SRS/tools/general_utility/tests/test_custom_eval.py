#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import re

import numpy as np
import pandas as pd
import pytest
from tools.general_utility.custom_eval import evaluate_formula


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame({"column_1": [1, 2.3, 3.2, 1, 0], "column_2": [1, 1, 2, 2, 3]})


def test__evaluate_formula__evaluates_correctly(dummy_df: pd.DataFrame):
    pd.testing.assert_series_equal(
        np.power(abs(-np.exp(dummy_df["column_1"])), 3) + 5,
        evaluate_formula("np.power(abs(-np.exp(df['column_1'])), 3) + 5", subscriptable_dict={"df": dummy_df}),
    )


def test__evaluate_formula__evaluates_method_correctly(dummy_df: pd.DataFrame):
    pd.testing.assert_series_equal(
        dummy_df["column_1"].add(dummy_df["column_2"]),
        evaluate_formula("df['column_1'].add(df['column_2'])", subscriptable_dict={"df": dummy_df}),
    )


def test_errors_get_passed_correctly():
    with pytest.raises(ZeroDivisionError):
        evaluate_formula("1/0")


def test__evaluate_formula__raises_when_trying_to_evaluate_unknown_method_of_a_module():
    with pytest.raises(
        AttributeError,
        match=re.escape("module 'numpy' has no attribute 'abcd'"),
    ):
        evaluate_formula("np.abcd(1)")


def test__evaluate_formula__raises_when_trying_to_evaluate_unknown_method_of_an_object(dummy_df: pd.DataFrame):
    with pytest.raises(
        AttributeError,
        match=re.escape("'DataFrame' object has no attribute 'absolute_value_calculation'"),
    ):
        evaluate_formula("df.absolute_value_calculation()", {"df": dummy_df})


@pytest.mark.parametrize("formula, unknown_function", [("os.getcwd()", "getcwd()"), ("exp(np.abs(-1))", "exp()")])
def test__evaluate_formula__raises_when_trying_to_evaluate_unknown_function(formula: str, unknown_function: str):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unknown function '{unknown_function}' occurred in a formula evaluation"),
    ):
        evaluate_formula(formula)


def test__evaluate_formula__raises_for_unsupported_operation():
    with pytest.raises(
        ValueError,
        match=re.escape("Error in a formula evaluation. Operation of type <class '_ast.Pow'> is not supported."),
    ):
        evaluate_formula("1**2")


def test__evaluate_formula__raises_for_multiple_comparisons():
    with pytest.raises(
        ValueError,
        match=re.escape("Multiple Comparison not supported in a formula evaluation, got comparators"),
    ):
        evaluate_formula(formula="1<2&2<1")


def test__evaluate_formula__raises_for_subscription_not_performed_on_the_dataframe(dummy_df: pd.DataFrame):
    with pytest.raises(
        ValueError,
        match=re.escape("Unknown subscript 'not_the_df' encountered in a formula evaluation."),
    ):
        evaluate_formula("not_the_df['a']", subscriptable_dict={"df": dummy_df})


def test__evaluate_formula__raises_for_node_of_unknown_type():
    with pytest.raises(
        ValueError,
        match=re.escape("Unsupported element of type <class '_ast.Dict'> found in a formula."),
    ):
        evaluate_formula("{'a': 1}")
