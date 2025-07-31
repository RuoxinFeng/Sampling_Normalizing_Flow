#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import Any, Dict, List

import pydantic
import pytest
from tools.sessions_to_manually_change.sessions_to_manually_change_schema import SessionsToChange


@pytest.fixture
def valid_config() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "8363935A": {"standstill_error_not_evaluated_CB_ID": {"287313dd-e770-479c-a405-018a21e65a76": 0.4594}},
        "2468153": {"some_error": {"session_id": "new_string_1", "session_id_2": "new_string_2"}},
        "39847938A": {"error_1": {"session_id": 1.9}, "error_2": {"session_id_2": 3423}},
    }


def test__valid_config(valid_config: Dict[str, Dict[str, Any]]):
    unit = SessionsToChange(**valid_config)
    assert unit.mappings_by_cb_id == {
        "8363935A": {"287313dd-e770-479c-a405-018a21e65a76": 0.4594},
        "2468153": {"session_id": "new_string_1", "session_id_2": "new_string_2"},
        "39847938A": {"session_id": 1.9, "session_id_2": 3423},
    }


@pytest.mark.parametrize("cb_id", ["123456", "123456A", "2344567AB", "2345678A1", "12345678A ", " 1234456"])
def test__raises_for_incorrect_cb_id(valid_config: Dict[str, Dict[str, Dict[str, Any]]], cb_id: str):
    valid_config[cb_id] = {"error": {"session_id": 1}}
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=f'Invalid codebeamer id "{cb_id}" provided!'):
        SessionsToChange(**valid_config)


@pytest.mark.parametrize(
    ["invalid_config", "incorrect_value"],
    [
        ({"123456789": {"error": {"session_id": {"too_deep": 1}}}}, {"too_deep": 1}),
        (
            {"123456789": {"error": {"session_id": ["this", "should", "not", "be", "a", "list"]}}},
            ["this", "should", "not", "be", "a", "list"],
        ),
    ],
)
def test__raises_for_too_deep_hierarchy(invalid_config: Dict[str, Dict[str, Dict[str, Any]]], incorrect_value: Any):
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match=re.escape(
            f"json file of sessions to manually change can have a maximum hierarchy of 3 nested dictionaries and no lists! Got {incorrect_value} but expected a float, int or string."
        ),
    ):
        SessionsToChange(**invalid_config)


@pytest.mark.parametrize(
    "values_to_use",
    [
        [1, "abc"],
        [True, 1.1],
        [True, 1.1, 1],
        ["abc", False],
        [None, 1.1],
        [None, False],
        [None, 1.1],
        [True, None, 1.1, 1, "abc"],
    ],
)
def test__raises_for_inconsistent_datatypes_different_reasons(values_to_use: List[Any]):
    invalid_config = {"1234567": {f"error_{i}": {f"session_{i}": value} for i, value in enumerate(values_to_use)}}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="Variable 1234567 got an inconsistent set of datatypes for replacements:",
    ):
        SessionsToChange(**invalid_config)


@pytest.mark.parametrize(
    "values_to_use",
    [
        [1, "abc"],
        [True, 1.1],
        [True, 1.1, 1],
        ["abc", False],
        [None, 1.1],
        [None, False],
        [None, 1.1],
        [True, None, 1.1, 1, "abc"],
    ],
)
def test__raises_for_inconsistent_datatypes_same_reason(values_to_use: List[Any]):
    invalid_config = {"1234567": {"error": {f"session_{i}": value for i, value in enumerate(values_to_use)}}}
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="Variable 1234567 got an inconsistent set of datatypes for replacements:",
    ):
        SessionsToChange(**invalid_config)


def test__raises_if_sessions_are_not_unique():
    invalid_config = {
        "1234567": {"some_error": {"session_id_2": "correct_value"}},
        "39847938A": {"error_1": {"session_id": 1.9}, "error_2": {"session_id": 3423}},
    }
    with pytest.raises(
        pydantic.error_wrappers.ValidationError,
        match="Some sessions have been defined multiple times for cb id 39847938A",
    ):
        SessionsToChange(**invalid_config)
