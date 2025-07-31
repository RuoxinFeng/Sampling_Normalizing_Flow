#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from os.path import dirname
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from tools.remove_invalid_sessions.remove_invalid_sessions import (
    SRS,
    InvalidSessionsSpec,
    ReasonsToRemove,
    remove_invalid_sessions,
)

cwd = dirname(__file__)


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd", return_value=cwd)
def test_remove_invalid_sessions_all_reasons(mock_getcwd: Mock, capsys: pytest.CaptureFixture):
    df = pd.DataFrame(
        {
            "session_id": [
                "93016ba4-3580-4479-a354-01879468971d",
                "73df3533-c85e-443e-a95f-01880b21b826",
                "0e39c16e-187a-47df-ae20-01879464707d",
                "c3697164-8e18-4831-ae81-0185f36d0265",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d0",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
                "1f9c0674-34ad-4727-aa5b-018534cf5c6f",
            ],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    removed_df, actual_total_failure_info = remove_invalid_sessions(
        df=df, spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=["frr", "lidar"])
    )
    prints = capsys.readouterr().out

    expected_prints = "No duplicate sessions detected in the dataframe.\nNo duplicate session ids detected in total failure sessions file.\nRemoved 5/17 degraded_sessions successfully from df.\nRemoved 1/18 invalid_sessions successfully from df.\nRemoved 0/4 total_failure_sessions successfully from df.\nRemoved a total of 6/39 sessions successfully from df.\nDataframe now has 3 entries, had 9.\nFollowing sessions are removed:\ndegraded_sessions : {'Bad radome auto': ['b6896bd6-6186-4b37-a9fa-0187c1b286d0'], 'kpi_tool_issues': ['73df3533-c85e-443e-a95f-01880b21b826'], 'outside_SRS06_scope': ['93016ba4-3580-4479-a354-01879468971d', '0e39c16e-187a-47df-ae20-01879464707d'], 'rt_range_issues_automatic': ['c3697164-8e18-4831-ae81-0185f36d0265']}\ninvalid_sessions : {'plp_issues': [], 'list_of_suspected': ['1f9c0674-34ad-4727-aa5b-018534cf5c6f'], 'rt_range_issues_automatic': []}\ntotal_failure_sessions : {'kpi_tool_issues': [], 'another_issue': []}\nObserved total failure rate: 0 per 3 sessions.\nPosterior total failure probability with uniform prior: 1 per 5 sessions (20.000%).\n"
    expected_df = pd.DataFrame(
        {
            "session_id": [
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
            ],
            "other_column": [6, 7, 8],
        }
    )

    pd.testing.assert_frame_equal(removed_df, expected_df)
    assert expected_prints == prints
    assert actual_total_failure_info.posterior_total_failure_probability == pytest.approx(1 / 5)
    assert actual_total_failure_info.observed_total_failures == 0


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd", return_value=cwd)
def test_remove_invalid_sessions_one_reason(mock_getcwd: Mock, capsys: pytest.CaptureFixture):
    df = pd.DataFrame(
        {
            "session_id": [
                "93016ba4-3580-4479-a354-01879468971d",
                "73df3533-c85e-443e-a95f-01880b21b826",
                "0e39c16e-187a-47df-ae20-01879464707d",
                "c3697164-8e18-4831-ae81-0185f36d0265",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d0",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
                "1f9c0674-34ad-4727-aa5b-018534cf5c6f",
            ],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    removed_df, actual_total_failure_info = remove_invalid_sessions(
        df=df,
        spec=InvalidSessionsSpec(
            srs=SRS.SRS04, sensor_names=["frr", "lidar"], reasons_to_remove=[ReasonsToRemove.DegradedSessions]
        ),
    )

    expected_df = pd.DataFrame(
        {
            "session_id": [
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
                "1f9c0674-34ad-4727-aa5b-018534cf5c6f",
            ],
            "other_column": [6, 7, 8, 9],
        }
    )
    prints = capsys.readouterr().out

    expected_prints = "No duplicate sessions detected in the dataframe.\nNo duplicate session ids detected in total failure sessions file.\nRemoved 5/17 degraded_sessions successfully from df.\nRemoved a total of 5/17 sessions successfully from df.\nDataframe now has 4 entries, had 9.\nFollowing sessions are removed:\ndegraded_sessions : {'Bad radome auto': ['b6896bd6-6186-4b37-a9fa-0187c1b286d0'], 'kpi_tool_issues': ['73df3533-c85e-443e-a95f-01880b21b826'], 'outside_SRS06_scope': ['93016ba4-3580-4479-a354-01879468971d', '0e39c16e-187a-47df-ae20-01879464707d'], 'rt_range_issues_automatic': ['c3697164-8e18-4831-ae81-0185f36d0265']}\nObserved total failure rate: 0 per 4 sessions.\nPosterior total failure probability with uniform prior: 1 per 6 sessions (16.667%).\n"

    pd.testing.assert_frame_equal(removed_df, expected_df)
    assert expected_prints == prints
    assert actual_total_failure_info.posterior_total_failure_probability == pytest.approx(1 / 6)
    assert actual_total_failure_info.observed_total_failures == 0


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd", return_value=cwd)
def test_total_failure_info(mock_getcwd: Mock):
    df = pd.DataFrame(
        {
            "session_id": [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    removed_df, actual_total_failure_info = remove_invalid_sessions(
        df=df,
        spec=InvalidSessionsSpec(
            srs=SRS.SRS04, sensor_names=["frr", "lidar"], reasons_to_remove=[ReasonsToRemove.TotalFailureSessions]
        ),
    )

    expected_df = pd.DataFrame(
        {
            "session_id": [
                "e",
                "f",
                "g",
                "h",
                "i",
            ],
            "other_column": [5, 6, 7, 8, 9],
        }
    )

    pd.testing.assert_frame_equal(removed_df, expected_df)
    assert actual_total_failure_info.posterior_total_failure_probability == pytest.approx(5 / 11)
    assert actual_total_failure_info.observed_total_failures == 4


def test_raises_when_filter_function_called_without_srs_enum():
    with pytest.raises(ValueError, match="SRS SRS-04 has to be a member of the SRS Enum."):
        remove_invalid_sessions(
            df=pd.DataFrame({"session_id": []}),
            spec=InvalidSessionsSpec(
                srs="SRS-04", sensor_names=["frr", "lidar"], reasons_to_remove=[ReasonsToRemove.DegradedSessions]
            ),
        )


def test_raises_when_filter_function_called_without_reasons_to_remove_enum():
    with pytest.raises(
        ValueError,
        match=re.escape("Reasons to remove ['degraded_sessions'] have to be a member of the ReasonsToRemove Enum."),
    ):
        remove_invalid_sessions(
            df=pd.DataFrame({"session_id": []}),
            spec=InvalidSessionsSpec(
                srs=SRS.SRS04,
                sensor_names=["frr", "lidar"],
                reasons_to_remove=["degraded_sessions", ReasonsToRemove.InvalidSessions],
            ),
        )


def test_raises_when_filter_function_called_with_invalid_sensor_names_parameter():
    with pytest.raises(ValueError, match="sensor_names parameter not_a_list is not a valid list"):
        remove_invalid_sessions(
            df=pd.DataFrame({"session_id": []}),
            spec=InvalidSessionsSpec(
                srs=SRS.SRS04, sensor_names="not_a_list", reasons_to_remove=[ReasonsToRemove.InvalidSessions]
            ),
        )


def test_raises_when_duplicates_are_in_dataframe():
    df = pd.DataFrame(
        {
            "session_id": ["a", "b", "a", "d"],
            "other_column": [1, 2, 3, 2],
        }
    )
    with pytest.raises(
        ValueError, match=re.escape("Duplicates of the following session IDs exist in the dataframe: ['a']")
    ):
        remove_invalid_sessions(
            df=df,
            spec=InvalidSessionsSpec(
                srs=SRS.SRS04, sensor_names=["frr", "lidar"], reasons_to_remove=[ReasonsToRemove.TotalFailureSessions]
            ),
        )


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd", return_value=cwd)
@pytest.mark.parametrize("sensor_names", [["global"], ["global", "frr"], ["frr", "global"]])
def test_raises_when_called_with_global(mock_getcwd: Mock, sensor_names: List[str]):
    df = pd.DataFrame(
        {
            "session_id": ["a", "b", "c", "d"],
            "other_column": [1, 2, 3, 2],
        }
    )
    with pytest.raises(
        ValueError,
        match="'global' invalid sessions get removed by default. Please only provide additional sensors whose invalid sessions should be removed.",
    ):
        remove_invalid_sessions(
            df=df,
            spec=InvalidSessionsSpec(
                srs=SRS.SRS04, sensor_names=sensor_names, reasons_to_remove=[ReasonsToRemove.TotalFailureSessions]
            ),
        )


@pytest.mark.parametrize(
    ["file_input", "detected_sessions"],
    [
        (
            {
                "sensor1": {"category1": {"value": ["id_1", "id_2", "id_3"]}},
                "sensor2": {
                    "category2": {"value": ["id_1", "id_5", "id_6"]},
                    "category3": {"__comment": "some comment", "value": ["id_7", "id_8", "id_9"]},
                },
            },
            "id_1",
        ),
        (
            {
                "sensor1": {"category1": {"value": ["id_1", "id_2", "id_3"]}},
                "sensor2": {
                    "category2": {"value": ["id_1", "id_1", "id_4"]},
                    "category3": {"__comment": "some comment", "value": ["id_2", "id_4", "id_9"]},
                },
            },
            "id_1<br>id_2",
        ),
    ],
)
@patch("tools.remove_invalid_sessions.remove_invalid_sessions._open_and_validate_file")
@patch("tools.remove_invalid_sessions.remove_invalid_sessions.Markdown")
def test_duplicate_sessions_in_total_failure_sessions_detected_across_multiple_sensors(
    md_mock: MagicMock, file_mock: MagicMock, file_input: Dict[str, Dict[str, Dict[str, Any]]], detected_sessions: str
):
    file_mock.return_value = file_input
    remove_invalid_sessions(
        df=pd.DataFrame({"session_id": []}),
        spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=[], reasons_to_remove=[]),
    )
    md_mock.assert_called_once_with(
        f"**<span style='background-color: red'>Correlated Total Failures detected. Please urgently contact Stanislav Braun, Moritz Werling or Felix Modes for deeper investigation. Affected sessions:<br>{detected_sessions}</span>**"
    )


@patch("tools.remove_invalid_sessions.remove_invalid_sessions.os.getcwd", return_value=cwd)
def test_remove_invalid_sesssions_with_prefix(mock_getcwd: Mock, capsys: pytest.CaptureFixture):
    df = pd.DataFrame(
        {
            "session_id": [
                "93016ba4-3580-4479-a354-01879468971d",
                "73df3533-c85e-443e-a95f-01880b21b826",
                "0e39c16e-187a-47df-ae20-01879464707d",
                "c3697164-8e18-4831-ae81-0185f36d0265",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d0",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
                "1f9c0674-34ad-4727-aa5b-018534cf5c6f",
            ],
            "other_column": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    removed_df, actual_total_failure_info = remove_invalid_sessions(
        df=df,
        spec=InvalidSessionsSpec(
            srs=SRS.SRS04,
            sensor_names=["frr", "lidar"],
            reasons_to_remove=[ReasonsToRemove.DegradedSessions],
            file_prefix="some_prefix_",
        ),
    )

    expected_df = pd.DataFrame(
        {
            "session_id": [
                "0e39c16e-187a-47df-ae20-01879464707d",
                "c3697164-8e18-4831-ae81-0185f36d0265",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d1",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d2",
                "b6896bd6-6186-4b37-a9fa-0187c1b286d3",
                "1f9c0674-34ad-4727-aa5b-018534cf5c6f",
            ],
            "other_column": [3, 4, 6, 7, 8, 9],
        }
    )
    prints = capsys.readouterr().out

    expected_prints = "No duplicate sessions detected in the dataframe.\nNo duplicate session ids detected in total failure sessions file.\nRemoved 3/10 degraded_sessions successfully from df.\nRemoved a total of 3/10 sessions successfully from df.\nDataframe now has 6 entries, had 9.\nFollowing sessions are removed:\ndegraded_sessions : {'Bad radome auto': ['b6896bd6-6186-4b37-a9fa-0187c1b286d0'], 'kpi_tool_issues': ['73df3533-c85e-443e-a95f-01880b21b826'], 'outside_SRS06_scope': ['93016ba4-3580-4479-a354-01879468971d'], 'rt_range_issues_automatic': []}\nObserved total failure rate: 0 per 6 sessions.\nPosterior total failure probability with uniform prior: 1 per 8 sessions (12.500%).\n"

    pd.testing.assert_frame_equal(removed_df, expected_df)
    assert expected_prints == prints
    assert actual_total_failure_info.posterior_total_failure_probability == pytest.approx(1 / 8)
    assert actual_total_failure_info.observed_total_failures == 0
