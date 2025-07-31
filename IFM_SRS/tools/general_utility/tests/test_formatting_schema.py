#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Dict

import pytest
from tools.general_utility.srs_dependent_formatting_schema import FormattingModel


@pytest.fixture
def valid_formatting() -> Dict[str, Dict[str, str]]:
    return {"SRS-XX": {"a": "b"}, "SRS-XY": {"c": "d"}}


def test__valid_formatting(valid_formatting: Dict[str, Dict[str, str]]):
    FormattingModel(**valid_formatting)


def test__raises_when_keys_dont_have_correct_starting_string(valid_formatting: Dict[str, Dict[str, str]]):
    valid_formatting["no valid srs str"] = valid_formatting.pop("SRS-XX")
    with pytest.raises(ValueError, match="Key 'no valid srs str' must start with 'SRS-'"):
        FormattingModel(**valid_formatting)


def test__raises_when_cols_renaming_field_is_not_a_dict(valid_formatting: Dict[str, Dict[str, str]]):
    SRS = "SRS-XX"
    valid_formatting[SRS] = "not a dict"
    with pytest.raises(
        ValueError,
        match=f"Renaming should be defined as a dictionary, but '{valid_formatting[SRS]}' was given for {SRS}",
    ):
        FormattingModel(**valid_formatting)


def test__raises_when_initial_col_name_is_not_string(valid_formatting: Dict[str, Dict[str, str]]):
    SRS = "SRS-XY"
    valid_formatting[SRS]["c"] = 23
    with pytest.raises(ValueError, match=f"Column names should be strings, but {valid_formatting[SRS]} was given"):
        FormattingModel(**valid_formatting)


def test__raises_when_final_col_name_is_not_string(valid_formatting: Dict[str, Dict[str, str]]):
    SRS = "SRS-XY"
    valid_formatting[SRS][0.1] = "d"
    with pytest.raises(ValueError, match=f"Column names should be strings, but {valid_formatting[SRS]} was given"):
        FormattingModel(**valid_formatting)
