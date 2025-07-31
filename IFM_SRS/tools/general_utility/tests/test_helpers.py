#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import re
import warnings
from datetime import datetime, timedelta
from os.path import dirname, join
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from tools.general_utility.helpers import (
    NR_OF_DIGITS_AFTER_COMMA,
    OUTPUT_TOLERANCE,
    TIMESTAMP_FORMAT,
    SRSFormattingCategories,
    _translate_categorical_interval,
    format_data,
    get_now_string,
    print_and_align,
    provide_formula_with_all_entries,
    read_and_verify_csv_data,
    round_output,
    translate_categorical_intervals,
    validate_timestamp_format,
)
from tools.general_utility.tests.test_data.srs_dependent_formatting.srs_dependent_transformations import (
    apply_data_transformation,
)
from tools.remove_invalid_sessions.remove_invalid_sessions import SRS, InvalidSessionsSpec

MOCK_CWD = join(dirname(__file__), "test_data")


@pytest.fixture
def input_provide_formula_with_all_entries() -> Dict[str, Any]:
    list_of_SPV = ["not_correct", "not_this_one", "correct_SPV", "danger_do_not_use"]
    index_SPV = 2
    list_of_IFV = ["IFV_1", "IFV_2", "IFV_3", "last_IFV"]

    return {"list_of_SPV": list_of_SPV, "index_SPV": index_SPV, "list_of_IFV": list_of_IFV}


def test_provide_formula_with_all_entries(input_provide_formula_with_all_entries: Dict[str, Any]):
    expected_formula = "'correct_SPV ~ '\n'IFV_1 + ' \n'IFV_2 + ' \n'IFV_3 + ' \n'last_IFV'"
    actual_formula = provide_formula_with_all_entries(**input_provide_formula_with_all_entries)

    assert expected_formula == actual_formula


@pytest.mark.parametrize("invalid_index", [0.5, -1, 4])
def test_provide_formula_with_all_entries_raises_for_invalid_index(
    invalid_index: float, input_provide_formula_with_all_entries: Dict[str, Any]
):
    input_provide_formula_with_all_entries["index_SPV"] = invalid_index

    with pytest.raises(
        ValueError,
        match=re.escape(f"Incorrect index {invalid_index} given. Please provide an integer between 0 and 3"),
    ):
        provide_formula_with_all_entries(**input_provide_formula_with_all_entries)


def create_initial_dataframes(colnames: List[str], colnames_exp: List[str]):
    data = np.ones((1, len(colnames))).tolist()
    df = pd.DataFrame(columns=colnames, data=data)
    df_exp = pd.DataFrame(columns=colnames_exp, data=data)
    return df, df_exp


@patch("tools.general_utility.helpers.os.getcwd", return_value=MOCK_CWD)
def test_formatting_works_as_expected_without_cleaning(mock_getcwd):
    input_df = pd.DataFrame(
        {
            "column_1": [1, 2, 3],
            "column_2": [2, -3, 4],
            "some_other_column": [0, 0.1, 0.003],
            "categorical_column": ["car", "motorcycle", "space_shuttle"],
            "column_stays_the_same": ["a", "b", "c"],
            "dont clean this column(-)": [0, 0, 0],
        }
    )
    expected_df = pd.DataFrame(
        {
            "column_2": [1, 2, 3],
            "column_3": [2, -3, 4],
            "some_other_column_new": [0, 10, 0.3],
            "categorical_column": ["car or motorcycle", "car or motorcycle", "space_shuttle"],
            "column_stays_the_same": ["a", "b", "c"],
            "dont clean this column(-)": [0, 0, 0],
            "combined_column": [2, 6, 12],
        }
    )
    actual_df = format_data(df_=input_df, srs=SRSFormattingCategories.SRS04_1)
    pd.testing.assert_frame_equal(expected_df, actual_df)


def formatting_test_asserts(
    srs: SRSFormattingCategories,
    colnames: List[str],
    colnames_exp: List[str],
    capsys: pytest.CaptureFixture,
    value_translations: Dict[str, Callable[[float], float]] = dict(),
    clean_column_names: bool = False,
    expected_missing_columns: Optional[List[str]] = None,
) -> None:
    df, df_exp = create_initial_dataframes(colnames=colnames, colnames_exp=colnames_exp)
    for column, value_translation in value_translations.items():
        df_exp[column] = df_exp[column].apply(value_translation)
    pd.testing.assert_frame_equal(format_data(df_=df, srs=srs, clean_column_names=clean_column_names), df_exp)
    captured = capsys.readouterr()
    if expected_missing_columns is None:
        assert "The following columns were not found in the dataframe:" not in captured.out
    else:
        expected_missing_columns_formatted = [f"'{column}'" for column in expected_missing_columns]
        assert (
            f"The following columns were not found in the dataframe: [{', '.join(expected_missing_columns_formatted)}]"
            in captured.out
        )


@patch("tools.general_utility.helpers.os.getcwd", return_value=MOCK_CWD)
def test_cleaning_of_column_names_works_as_expected(mock_getcwd, capsys: pytest.CaptureFixture):
    colnames = ["Some Random-colname()", "branch", "initial_host_velocity_8533202", "ego_velocity_start"]
    colnames_exp = ["some_randomcolname", "branch", "initial_host_velocity_categorial", "initial_host_velocity_8533202"]
    formatting_test_asserts(
        srs=SRSFormattingCategories.SRS_template,
        colnames=colnames,
        colnames_exp=colnames_exp,
        capsys=capsys,
        clean_column_names=True,
    )


@patch("tools.general_utility.helpers.os.getcwd", return_value=MOCK_CWD)
def test_correct_print_for_missing_columns(mock_getcwd, capsys: pytest.CaptureFixture):
    colnames = ["Some Random-colname()", "branch"]
    colnames_exp = ["some_randomcolname", "branch"]
    formatting_test_asserts(
        srs=SRSFormattingCategories.SRS_template,
        colnames=colnames,
        colnames_exp=colnames_exp,
        capsys=capsys,
        clean_column_names=True,
        expected_missing_columns=["initial_host_velocity_8533202", "ego_velocity_start"],
    )


def test_raises_when_srs_is_not_a_valid_srs_enum():
    with pytest.raises(ValueError, match="SRS SRS-XX has to be a member of SRSFormattingCategories."):
        format_data(df_=pd.DataFrame(), srs="SRS-XX")


@patch("tools.general_utility.helpers.os.getcwd", return_value=MOCK_CWD)
def test_prints_correct_formatting_info(mock_getcwd, capsys: pytest.CaptureFixture):
    with open(join(MOCK_CWD, "srs_dependent_formatting", "srs_dependent_formatting.json")) as f:
        formatting_info = json.load(f)
    for srs in [
        SRSFormattingCategories.SRS_template,
        SRSFormattingCategories.SRS_template_ifm,
        SRSFormattingCategories.SRS04_1,
    ]:
        format_data(df_=MagicMock(), srs=srs)
        captured = capsys.readouterr()
        for col in formatting_info[srs.value]:
            assert str(col) in captured.out


def test_now_string_has_correct_format_and_is_up_to_date():
    actual_string = get_now_string()
    expected_time = datetime.now()

    actual_time = datetime.strptime(actual_string, "%d/%m/%Y %H:%M:%S")

    assert abs(expected_time - actual_time) < timedelta(seconds=5)


def test_read_and_verify_csv_data_minimal_example(capsys: pytest.CaptureFixture):
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example.csv")
    expected_df = pd.DataFrame({"Column_1": ["a", "d"], "Column_2": ["b", "e"], "another_column": ["c", "f"]})
    expected_timestamp = "03/07/2023 09:39:55"
    expected_header_prints = [
        "Header lines of the read file:",
        "This is a header line.",
        "Some more information",
        "Export timestamp: 03/07/2023 09:39:55 UTC+0",
        "Metadata: a",
    ]

    actual_input_data = read_and_verify_csv_data(path=input_path)

    captured = capsys.readouterr()
    pd.testing.assert_frame_equal(actual_input_data.df, expected_df)
    assert actual_input_data.timestamp_input_data == expected_timestamp
    assert all([line in captured.out for line in expected_header_prints])


def test_read_and_verify_csv_data_minimal_example_no_header():
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example_without_header.csv")
    expected_df = pd.DataFrame({"Column_1": ["a", "d"], "Column_2": ["b", "e"], "another_column": ["c", "f"]})

    actual_input_data = read_and_verify_csv_data(path=input_path)

    pd.testing.assert_frame_equal(actual_input_data.df, expected_df)
    assert actual_input_data.timestamp_input_data == None


def test_read_and_verify_csv_data_minimal_example_custom_header_lines(capsys: pytest.CaptureFixture):
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example_without_blank_line.csv")
    expected_df = pd.DataFrame({"Column_1": ["a", "d"], "Column_2": ["b", "e"], "another_column": ["c", "f"]})
    expected_timestamp = "03/07/2023 09:39:55"
    expected_header_prints = [
        "Header lines of the read file:",
        "This is a header line.",
        "Some more information",
        "Export timestamp: 03/07/2023 09:39:55 UTC+0",
        "Metadata: a",
    ]

    actual_input_data = read_and_verify_csv_data(path=input_path, header_lines=4)

    captured = capsys.readouterr()
    pd.testing.assert_frame_equal(actual_input_data.df, expected_df)
    assert actual_input_data.timestamp_input_data == expected_timestamp
    assert all([line in captured.out for line in expected_header_prints])


@pytest.mark.parametrize(
    "test_file",
    [
        ("input_with_timestamp.csv"),
        ("input_with_timestamp_v2.csv"),
        ("arbitrary_timestamp_placement.csv"),
        ("input_with_commas.csv"),
    ],
)
def test_read_and_verify_csv_data_gets_correct_timestamp(test_file: str):
    input_path = join(MOCK_CWD, "dummy_input_data", test_file)
    actual_timestamp = read_and_verify_csv_data(path=input_path).timestamp_input_data

    assert actual_timestamp == "03/07/2023 09:39:55"


def test_read_and_verify_csv_data_without_timestamp(capsys: pytest.CaptureFixture):
    input_path = join(MOCK_CWD, "dummy_input_data", "input_without_timestamp.csv")
    actual_timestamp = read_and_verify_csv_data(path=input_path).timestamp_input_data

    captured = capsys.readouterr()

    assert "Warning: No timestamp found!" in captured.out
    assert actual_timestamp is None


def test_read_and_verify_csv_data_warns_for_few_occurrences_of_categories():
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example_with_categorical_ifvs.csv")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        read_and_verify_csv_data(path=input_path, min_occurrences_ifv_coverage_check=2)
        assert "The categories {'d'} of the IFV Column_1 1234567, have a small number of occurrences({2})" in str(
            w[0].message
        )
        assert "The categories {'e'} of the IFV Column_2 23456789, have a small number of occurrences({1})" in str(
            w[1].message
        )
        assert len(w) == 2


@patch("tools.general_utility.helpers.os.getcwd")
def test_read_and_verify_csv_data_warns_for_categories_in_df_but_not_in_design_spec_and_vice_versa(mock_cwd: MagicMock):
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example_with_categorical_ifvs.csv")
    mock_cwd.return_value = MOCK_CWD
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        read_and_verify_csv_data(
            path=input_path,
            min_occurrences_ifv_coverage_check=2,
            design_specification_filename="design_spec_minimal_example.json",
        )
        assert (
            "The categories {'e'} of the IFV Column_1 1234567 are specified in the design specification file design_spec_minimal_example.json, but have not been found in the dataframe."
            in str(w[2].message)
        )
        assert (
            "The categories {'e'} of the IFV Column_2 23456789 are not specified in the design specification file design_spec_minimal_example.json but have been found in the dataframe."
            in str(w[3].message)
        )
        assert len(w) == 4


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd")
def test_read_and_verify_csv_data_removes_invalid_data_correctly(mock_cwd: MagicMock):
    input_path = join(MOCK_CWD, "dummy_input_data", "minimal_example_with_categorical_ifvs.csv")
    mock_cwd.return_value = MOCK_CWD
    actual = read_and_verify_csv_data(
        path=input_path,
        invalid_sessions_spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=["adcam", "lidar", "frr"]),
    ).df
    expected = pd.DataFrame(
        {"session_id": ["s_id"], "Column_1 1234567": ["a"], "Column_2 23456789": ["b"], "another_column": ["c"]}
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_validate_timestamp_format_correct_format():
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    assert validate_timestamp_format(timestamp)


def get_timestamp(format: str = TIMESTAMP_FORMAT) -> str:
    return datetime.now().strftime(format)


def test_validate_timestamp_format_correct_format():
    timestamp = get_timestamp()
    assert validate_timestamp_format(timestamp)


def test_validate_timestamp_format_correct_custom_format():
    format = "%d.%m.%Y %H-%aM-%S"
    timestamp = get_timestamp(format)
    assert validate_timestamp_format(timestamp=timestamp, expected_format=format)


def test_validate_timestamp_format_incorrect_format():
    format = "%d.%m.%Y %H-%M-%S"
    timestamp = get_timestamp(format)
    assert not validate_timestamp_format(timestamp=timestamp)


def test_validate_timestamp_format_incorrect_custom_format():
    format = "%d.%m.%Y %H-%M-%S"
    timestamp = get_timestamp()

    assert not validate_timestamp_format(timestamp=timestamp, expected_format=format)


def test_translate_categorical_interval():
    assert "from 0.5 to 1.0" == _translate_categorical_interval("(0.5, 1.0]")
    assert "from 0.5 to 1.0" == _translate_categorical_interval("(0.5,1.0]")


def test_translate_categorical_intervals():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "cat1": ["(0.0, 0.5]", "(0.0, 0.5]", "(0.0, 0.5]"],
            "cat2": ["dry", "spray", "wet"],
            "cat3": ["(0.1, 0.7]", "(0.1, 0.7]", "(0.7, 1.0]"],
            "cat4": ["(0.1, 0.7]", "(0.1, 0.7]", "(0.7, 1.0]"],
        }
    )
    expected_df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "cat1": ["from 0.0 to 0.5", "from 0.0 to 0.5", "from 0.0 to 0.5"],
            "cat2": ["dry", "spray", "wet"],
            "cat3": ["from 0.1 to 0.7", "from 0.1 to 0.7", "from 0.7 to 1.0"],
            "cat4": ["(0.1, 0.7]", "(0.1, 0.7]", "(0.7, 1.0]"],
        }
    )
    df_before_execution = df.copy(deep=True)
    assert expected_df.equals(translate_categorical_intervals(df, ["cat1", "cat3"]))
    assert df.equals(df_before_execution)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (1.1e-12, 1e-12),
        (1.6e-12, 2e-12),
        (1, 1),
        ([1.1e-12, 1.6e-12, 0.1], [1e-12, 2e-12, 0.1]),
        ({"a": 1.1e-12, "b": 1.6e-12, "c": 0.1}, {"a": 1e-12, "b": 2e-12, "c": 0.1}),
        ((1.1e-12, 1.6e-12, 0.1), (1e-12, 2e-12, 0.1)),
        ({1.1e-12, 1.6e-12, 0.1}, {1e-12, 2e-12, 0.1}),
        (None, None),
        (
            [{1.1e-12}, (1.6e-12, 1), {"d": None}],
            [{1e-12}, (2e-12, 1), {"d": None}],
        ),
        (
            {"a": {"abc"}, "b": (1.6e-12, 1), "c": {"d": None}},
            {"a": {"abc"}, "b": (2e-12, 1), "c": {"d": None}},
        ),
        (
            ({True}, (1.6e-12, 1), {"d": None}),
            ({True}, (2e-12, 1), {"d": None}),
        ),
        (
            {1.1e-12, (1.6e-12, 1), None},
            {1e-12, (2e-12, 1), None},
        ),
    ],
)
def test_round_output_default_nr_of_digits(input_value: Any, expected_output: Any):
    assert round_output(input_value) == expected_output


def test_round_output_default_nr_of_digits_np_array():
    np.testing.assert_array_equal(round_output(np.array([1.1e-12, 1.6e-12, 0.1])), np.array([1e-12, 2e-12, 0.1]))


@pytest.mark.parametrize(
    "value_before_rounding, expected_output",
    [
        (1.1e-5, 1e-5),
        (1.6e-5, 2e-5),
        (1, 1),
        ([1.1e-5, 1.6e-5, 0.1], [1e-5, 2e-5, 0.1]),
        ({"a": 1.1e-5, "b": 1.6e-5, "c": 0.1}, {"a": 1e-5, "b": 2e-5, "c": 0.1}),
        ((1.1e-5, 1.6e-5, 0.1), (1e-5, 2e-5, 0.1)),
        ({1.1e-5, 1.6e-5, 0.1}, {1e-5, 2e-5, 0.1}),
        (None, None),
        (
            [{1.1e-5}, (1.6e-5, 1), {"d": None}],
            [{1e-5}, (2e-5, 1), {"d": None}],
        ),
        (
            {"a": {"abc"}, "b": (1.6e-5, 1), "c": {"d": None}},
            {"a": {"abc"}, "b": (2e-5, 1), "c": {"d": None}},
        ),
        (
            ({True}, (1.6e-5, 1), {"d": None}),
            ({True}, (2e-5, 1), {"d": None}),
        ),
        (
            {1.1e-5, (1.6e-5, 1), None},
            {1e-5, (2e-5, 1), None},
        ),
    ],
)
def test_round_output_custom_nr_of_digits(value_before_rounding: Any, expected_output: Any):
    assert round_output(value_before_rounding=value_before_rounding, nr_of_digits_after_comma=5) == expected_output


def test_round_output_custom_nr_of_digits_np_array():
    np.testing.assert_array_equal(
        round_output(value_before_rounding=np.array([1.1e-5, 1.6e-5, 0.1]), nr_of_digits_after_comma=5),
        np.array([1e-5, 2e-5, 0.1]),
    )


def test_default_nr_of_digits_and_output_tolerance_consistent():
    nr_to_be_rounded_to_tolerance = OUTPUT_TOLERANCE + OUTPUT_TOLERANCE / 3
    assert round_output(nr_to_be_rounded_to_tolerance) == OUTPUT_TOLERANCE
    assert 10 ** (-NR_OF_DIGITS_AFTER_COMMA) == OUTPUT_TOLERANCE


def test__print_and_align(capsys: pytest.CaptureFixture):
    print_and_align({"a": "long_value", "long_key": "b"})

    captured = capsys.readouterr().out
    assert "a       : long_value" in captured
    assert "long_key: b         "
