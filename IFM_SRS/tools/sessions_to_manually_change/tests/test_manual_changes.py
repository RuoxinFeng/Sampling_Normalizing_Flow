#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


from os.path import dirname
from unittest.mock import patch

import pandas as pd
import pytest
from tools.remove_invalid_sessions.remove_invalid_sessions import SRS
from tools.sessions_to_manually_change.manual_changes import manually_change_values

cwd = dirname(__file__)


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A_8363935A": [1, 2, 3, 4],
            "B_8363935B": [1, 2, 3, 4],
            "C_12345678": ["value", "value", "value", "value"],
            "session_id": ["1", "2", "3", "4"],
        }
    )


@patch("tools.sessions_to_manually_change.manual_changes.os.getcwd", return_value=cwd)
def test__manually_change_values(mock_getcwd, dummy_df: pd.DataFrame, capsys: pytest.CaptureFixture):
    expected_df = pd.DataFrame(
        {
            "A_8363935A": [0.4594, 2, 3, 4],
            "B_8363935B": [3, 0.74, 0.22, 4],
            "C_12345678": ["value", "value", "new_value", "value"],
            "session_id": ["1", "2", "3", "4"],
        }
    )
    actual_df = manually_change_values(df=dummy_df, srs=SRS.SRS04)

    pd.testing.assert_frame_equal(expected_df, actual_df)

    captured = capsys.readouterr()
    assert "Successfully replaced the values for the following columns and session ids:" in captured.out
    assert "'B_8363935B': {'1': 3, '2': 0.74, '3': 0.22}" in captured.out
    assert "'C_12345678': {'3': 'new_value'}" in captured.out
    assert "'A_8363935A': {'1': 0.4594}" in captured.out
    assert "The following session ids were not found in the Dataframe:" not in captured.out


@patch("tools.sessions_to_manually_change.manual_changes.os.getcwd", return_value=cwd)
def test__manually_change_values_with_unknown_cb_id_and_session_id(
    mock_getcwd, dummy_df: pd.DataFrame, capsys: pytest.CaptureFixture
):
    dummy_df = dummy_df.drop("A_8363935A", axis=1)
    dummy_df = dummy_df.drop([2, 3], axis=0)
    expected_df = pd.DataFrame(
        {
            "B_8363935B": [3, 0.74],
            "C_12345678": ["value", "value"],
            "session_id": ["1", "2"],
        }
    )
    # SRS06 json contains session 3 twice, so we can check if it only gets printed once
    actual_df = manually_change_values(df=dummy_df, srs=SRS.SRS06)

    pd.testing.assert_frame_equal(expected_df, actual_df)

    captured = capsys.readouterr()
    assert "No matching column found for codebeamer id 8363935A!" in captured.out
    assert "Successfully replaced the values for the following columns and session ids:" in captured.out
    assert "'B_8363935B': {'1': 3, '2': 0.74}" in captured.out
    assert "The following session ids were not found in the Dataframe: ['3']" in captured.out


@patch("tools.sessions_to_manually_change.manual_changes.os.getcwd", return_value=cwd)
def test__manually_change_values_raises_for_multiple_columns_with_cb_id(mock_getcwd, dummy_df: pd.DataFrame):
    dummy_df["D_8363935A"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="Found multiple columns for CB id 8363935A!"):
        actual_df = manually_change_values(df=dummy_df, srs=SRS.SRS04)
