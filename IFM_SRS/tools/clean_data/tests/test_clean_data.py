#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import Dict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tools.clean_data.clean_data import (
    _extract_cb_id,
    _get_cb_link,
    _get_df_to_display,
    _make_clickable,
    extract_and_rename_columns,
)


@pytest.mark.parametrize(
    "codebeamer_ids, input_df, expected_df",
    [
        (
            {
                "1234567": "New Column-Name One",
                "2345678": "New Column/Name$Two",
                "3456789": "New Column_Name Three",
                "345678512154": "New Column_Name Four",
                "987654321A": "New Column.Name Five",
                "XXXXW": "New column name without numbers in CB id",
            },
            pd.DataFrame(
                {
                    "1234567_col1": [1, 2, 3],
                    "2345678_col2": [4.0, 5.0, 6.0],
                    "3456789_col3": ["A", "B", "C"],
                    "345678512154_col_4": ["this", "is", "a column"],
                    "col_5_987654321A": ["S", "P", "V"],
                    "6666666_col_not_interesting": [6, 6, 6],
                    "session_id": ["A", "B", "C"],
                    "column_no_numbers_XXXXW": ["a", "b", "c"],
                }
            ),
            pd.DataFrame(
                {
                    "session_id": ["A", "B", "C"],
                    "new_column_name_one_1234567": [1, 2, 3],
                    "new_column_name_two_2345678": [4.0, 5.0, 6.0],
                    "new_column_name_three_3456789": ["A", "B", "C"],
                    "new_column_name_four_345678512154": ["this", "is", "a column"],
                    "new_column_name_five_987654321A": ["S", "P", "V"],
                    "new_column_name_without_numbers_in_cb_id_XXXXW": ["a", "b", "c"],
                }
            ),
        ),
        (
            {
                "1234567.1": "New Column Name Component One",
                "1234567.2": "New Column Name Component Two",
                "3456789": "New Column Name Three",
            },
            pd.DataFrame(
                {
                    "1234567.1_col1": [1, 2, 3],
                    "1234567.2_col2": [4.0, 5.0, 6.0],
                    "3456789_col3": ["A", "B", "C"],
                }
            ),
            pd.DataFrame(
                {
                    "new_column_name_component_one_1234567.1": [1, 2, 3],
                    "new_column_name_component_two_1234567.2": [4.0, 5.0, 6.0],
                    "new_column_name_three_3456789": ["A", "B", "C"],
                }
            ),
        ),
    ],
)
def test_extract_and_rename_columns(codebeamer_ids: Dict[str, str], input_df: pd.DataFrame, expected_df: pd.DataFrame):
    actual_df = extract_and_rename_columns(input_df, codebeamer_ids)
    assert_frame_equal(actual_df, expected_df)


def test_extract_and_rename_columns_raises_for_same_cb_id():
    df = pd.DataFrame({"12345678_col_1": ["a", "b", "c"], "12345678_col_2": [1, 2, 3]})
    ids_and_names = {"12345678": "New_name"}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Multiple columns may not have the same CB ID! Please review the following columns: ['12345678_col_1', '12345678_col_2'"
        ),
    ):
        extract_and_rename_columns(df, ids_and_names)


def test_extract_and_rename_columns_raises_for_incorrect_key():
    df = pd.DataFrame({"12345678A12_col_1": ["a", "b", "c"]})
    ids_and_names = {"12345678A12": "New_name"}
    with pytest.raises(
        ValueError, match=re.escape("Id 12345678A12 has multiple sequences of digits: ['12345678', '12']")
    ):
        extract_and_rename_columns(df, ids_and_names)


@pytest.mark.parametrize(
    "key,expected_cb_id",
    [
        ("abcde_1234567", "1234567"),
        ("abcde1234567", "1234567"),
        ("ABCDE1234567", "1234567"),
        ("abcde_1234567.1", "1234567.1"),
        ("1234567", "1234567"),
        ("abcdefg_12345678A", "12345678"),
        ("abcdefg12345678A", "12345678"),
        ("ABCDEFG12345678A", "12345678"),
        ("abcdefg_12345678.1A", "12345678.1"),
        ("12345678A", "12345678"),
        ("XXXXW", "XXXXW"),
    ],
)
def test_extract_cb_id(key: str, expected_cb_id: str):
    assert expected_cb_id == _extract_cb_id(key)


def test_get_cb_link():
    assert _get_cb_link("1234567") == "https://codebeamer.bmwgroup.net/cb/item/1234567"


def test_get_df_to_display():
    matching_col_names = ["1234567_col1", "2345678_col2", "3456789_col3"]
    new_col_names = ["new_column_name_one_1234567", "new_column_name_two_2345678", "new_column_name_three_3456789"]
    cb_links = [
        "https://codebeamer.bmwgroup.net/cb/item/1234567",
        "https://codebeamer.bmwgroup.net/cb/item/2345678",
        "https://codebeamer.bmwgroup.net/cb/item/3456789",
    ]
    expected_df = pd.DataFrame(
        {
            "Old": matching_col_names,
            "New": new_col_names,
            "Codebeamer": [_make_clickable(link) for link in cb_links],
        }
    )
    pd.testing.assert_frame_equal(
        expected_df,
        _get_df_to_display(matching_col_names=matching_col_names, new_col_names=new_col_names, cb_links=cb_links),
    )


def test_make_clickable():
    assert '<a href="some-url.com">some-url.com</a>' == _make_clickable("some-url.com")
