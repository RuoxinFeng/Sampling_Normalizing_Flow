#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
from os.path import join
from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel
from tools.general_utility.helpers import round_output
from tools.test_helpers.rounding_helpers import assert_rounded


def assert_file_content(
    expected_content: Dict[str, Any],
    tmpdir=None,
    filename: Optional[str] = None,
    file_path: Optional[str] = None,
    nr_of_digits_after_comma: Optional[int] = None,
) -> None:
    file_path = file_path or join(tmpdir, filename)
    with open(file_path, "r") as file:
        actual_content = json.load(file)
    assert_json_equal(actual_content, expected_content, nr_of_digits_after_comma)


def assert_json_equal(actual: Any, expected: Any, nr_of_digits_after_comma: Optional[int] = None) -> None:
    expected = _convert_pydantic_to_dict(expected)
    if nr_of_digits_after_comma is not None:
        expected = round_output(expected, nr_of_digits_after_comma)
    error_message = f"Got {actual}, expected {expected}"
    if isinstance(actual, list):
        assert len(actual) == len(expected), error_message
        for a, e in zip(actual, expected):
            assert_json_equal(actual=a, expected=e, nr_of_digits_after_comma=nr_of_digits_after_comma)
    elif isinstance(actual, dict):
        assert len(actual) >= len(
            expected
        ), error_message  # Actual could have more entries as we use exclude_unset in _convert_pydantic_to_dict
        for key, value in actual.items():
            assert_json_equal(
                actual=value, expected=expected.get(key), nr_of_digits_after_comma=nr_of_digits_after_comma
            )
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected), error_message
        if nr_of_digits_after_comma is not None:
            assert_rounded(actual, 10 ** (-nr_of_digits_after_comma))
    else:
        assert actual == expected, error_message


def _convert_pydantic_to_dict(expected: Any) -> Any:
    if isinstance(expected, BaseModel):
        return expected.dict(exclude_unset=True)
    return expected
