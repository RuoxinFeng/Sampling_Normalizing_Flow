#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import re
from math import sqrt
from os.path import dirname, join
from typing import List, Tuple
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest
from tools.comparison_tools.dataset_comparison import DatasetComparison
from tools.remove_invalid_sessions.remove_invalid_sessions import SRS, InvalidSessionsSpec, TotalFailureInfo

BASE_PATH = join(dirname(__file__), "test_data", "dataset_comparison")
OLD_DATA_PATH = join(BASE_PATH, "dummy_old_data.csv")
NEW_DATA_PATH = join(BASE_PATH, "dummy_new_data.csv")
DATASET_COMPARISON_PATH = "tools.comparison_tools.dataset_comparison"


@pytest.fixture()
def dummy_new_data() -> pd.DataFrame:
    return pd.read_csv(NEW_DATA_PATH, skiprows=7)


@pytest.fixture()
def dummy_old_data() -> pd.DataFrame:
    return pd.read_csv(OLD_DATA_PATH, skiprows=7)


@pytest.fixture
def simple_dataset_comparison(dummy_new_data: pd.DataFrame) -> DatasetComparison:
    def dummy_remove_invalid_sessions(
        df: pd.DataFrame, spec: InvalidSessionsSpec
    ) -> Tuple[pd.DataFrame, TotalFailureInfo]:
        if df.equals(dummy_new_data):
            return df, TotalFailureInfo(posterior_total_failure_probability=0.5, observed_total_failures=10)
        return df, TotalFailureInfo(posterior_total_failure_probability=0.6, observed_total_failures=5)

    with patch("tools.comparison_tools.dataset_comparison.remove_invalid_sessions", dummy_remove_invalid_sessions):
        return DatasetComparison(
            path_new_data=NEW_DATA_PATH,
            path_old_data=OLD_DATA_PATH,
            invalid_sessions_spec=Mock(),
            name_new_data="A new name",
            name_old_data="An old name",
        )


@patch(f"{DATASET_COMPARISON_PATH}.remove_invalid_sessions", return_value=(None, None))
@patch(f"{DATASET_COMPARISON_PATH}.read_and_verify_csv_data")
def test__data_read_correctly(
    mock_read_and_verify_csv_data,
    mock_remove_invalid_sessions,
    dummy_new_data: pd.DataFrame,
    dummy_old_data: pd.DataFrame,
):
    mock_read_and_verify_new_data_return = Mock()
    mock_read_and_verify_new_data_return.df = dummy_new_data
    mock_read_and_verify_old_data_return = Mock()
    mock_read_and_verify_old_data_return.df = dummy_old_data
    mock_read_and_verify_csv_data.side_effect = [
        mock_read_and_verify_new_data_return,
        mock_read_and_verify_old_data_return,
    ]
    DatasetComparison(
        path_new_data=NEW_DATA_PATH,
        path_old_data=OLD_DATA_PATH,
        invalid_sessions_spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=["adcam"]),
        header_lines_old_data=7,
    )

    assert mock_read_and_verify_csv_data.call_count == 2
    assert mock_read_and_verify_csv_data.call_args_list[0] == call(path=NEW_DATA_PATH, header_lines=None)
    assert mock_read_and_verify_csv_data.call_args_list[1] == call(path=OLD_DATA_PATH, header_lines=7)
    assert mock_remove_invalid_sessions.call_count == 2
    assert mock_remove_invalid_sessions.call_args_list[0] == call(
        df=dummy_new_data, spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=["adcam"])
    )
    assert mock_remove_invalid_sessions.call_args_list[1] == call(
        df=dummy_old_data, spec=InvalidSessionsSpec(srs=SRS.SRS04, sensor_names=["adcam"])
    )


def test__properties(simple_dataset_comparison: DatasetComparison):
    assert simple_dataset_comparison.columns_new_data == ["session_id", "column_1", "new_column"]
    assert simple_dataset_comparison.columns_old_data == ["session_id", "column_1", "old_column"]


@patch.object(DatasetComparison, "summarize")
@patch.object(DatasetComparison, "compare_box_plots")
@patch.object(DatasetComparison, "compare_divergence_metrics")
@patch.object(DatasetComparison, "compare_outliers")
@patch.object(DatasetComparison, "sessions_with_changes")
def test__full_analysis(
    mock_sessions_with_changes,
    mock_compare_outliers,
    mock_compare_divergence_metrics,
    mock_compare_box_plots,
    mock_summarize,
    simple_dataset_comparison: DatasetComparison,
):
    column_name = "some_column_name"
    simple_dataset_comparison.full_analysis(
        column_name=column_name,
        plot_width=1000,
        plot_height=100,
        nbins_for_histograms=10,
        threshold_for_changes=0.5,
        exclude_missing_sessions=False,
    )
    mock_summarize.assert_called_once_with(column_name)
    mock_compare_box_plots.assert_called_once_with(column_name, width=1000, height=100)
    mock_compare_divergence_metrics.assert_called_once_with(column_name, nbins=10)
    mock_compare_outliers.assert_called_once_with(column_name)
    mock_sessions_with_changes.assert_called_once_with(column_name, threshold=0.5, exclude_missing_sessions=False)


def test__full_analysis__smoke(simple_dataset_comparison: DatasetComparison, mock_plots: None):
    old_data_before_run = simple_dataset_comparison._old_data.copy()
    new_data_before_run = simple_dataset_comparison._new_data.copy()
    simple_dataset_comparison.full_analysis("column_1")
    pd.testing.assert_frame_equal(old_data_before_run, simple_dataset_comparison._old_data)
    pd.testing.assert_frame_equal(new_data_before_run, simple_dataset_comparison._new_data)


@patch(f"{DATASET_COMPARISON_PATH}.display_dataframe")
def test__summarize(mock_display_dataframe, simple_dataset_comparison: DatasetComparison):
    simple_dataset_comparison.summarize("column_1")
    mock_display_dataframe.assert_called_once()
    pd.testing.assert_frame_equal(
        mock_display_dataframe.call_args[0][0],
        pd.DataFrame(
            {
                "A new name": {
                    "Total number of sessions": 4,
                    "Total failure sessions": 10,
                    "Posterior total failure probability": 0.5,
                    "Mean": -23.5,
                    "Std": sqrt(((1 + 23.5) ** 2 + (2 + 23.5) ** 2 + (3 + 23.5) ** 2 + (-100 + 23.5) ** 2) / 3),
                    "Min": -100,
                    "Max": 3,
                    "Values outside std": 1,
                    "Outliers (outside whisker)": 1,
                },
                "An old name": {
                    "Total number of sessions": 5,
                    "Total failure sessions": 5,
                    "Posterior total failure probability": 0.6,
                    "Mean": 31.6,
                    "Std": sqrt(
                        ((5 - 31.6) ** 2 + (6 - 31.6) ** 2 + (7 - 31.6) ** 2 + (-60 - 31.6) ** 2 + (200 - 31.6) ** 2)
                        / 4
                    ),
                    "Min": -60,
                    "Max": 200,
                    "Values outside std": 1,
                    "Outliers (outside whisker)": 2,
                },
            }
        ),
    )


@pytest.mark.parametrize(
    "invalid_column, missing_dataset", [("old_column", "A new name"), ("new_column", "An old name")]
)
def test__summarize__raises_for_invalid_column(
    invalid_column: str, missing_dataset: str, simple_dataset_comparison: DatasetComparison
):
    with pytest.raises(ValueError, match=f"Column {invalid_column} does not exist in {missing_dataset}."):
        simple_dataset_comparison.summarize(invalid_column)


def test__compare_box_plots__smoke(simple_dataset_comparison: DatasetComparison, mock_plots: None):
    simple_dataset_comparison.compare_box_plots("column_1")


@pytest.mark.parametrize(
    "invalid_column, missing_dataset", [("old_column", "A new name"), ("new_column", "An old name")]
)
def test__compare_box_plots__raises_for_invalid_column(
    invalid_column: str, missing_dataset: str, simple_dataset_comparison: DatasetComparison
):
    with pytest.raises(ValueError, match=f"Column {invalid_column} does not exist in {missing_dataset}."):
        simple_dataset_comparison.compare_box_plots(invalid_column)


def test__compare_divergence_metrics__smoke(mock_plots: None, simple_dataset_comparison: DatasetComparison):
    simple_dataset_comparison.compare_divergence_metrics("column_1")


@patch(f"{DATASET_COMPARISON_PATH}.empirical_js_divergence", return_value=1)
@patch(f"{DATASET_COMPARISON_PATH}.ks_test")
@patch(f"{DATASET_COMPARISON_PATH}.plot_ks_test_statistic")
def test__compare_divergence_metrics__correct_print(
    mock_plot_ks_test_statistic,
    mock_ks_test,
    mock_empirical_js_divergence,
    simple_dataset_comparison: DatasetComparison,
    capsys: pytest.CaptureFixture,
):
    mock_ks_test_result = Mock()
    mock_ks_test_result.statistic = 2
    mock_ks_test.return_value = mock_ks_test_result
    simple_dataset_comparison.compare_divergence_metrics("column_1")
    captured = capsys.readouterr().out
    assert "Divergence metrics for column_1:" in captured
    assert "Jensen-Shannon Divergence: 1" in captured
    assert "Kullback-Leibler Divergence: 2" in captured


@patch(f"{DATASET_COMPARISON_PATH}.plot_ks_test_statistic")
@patch(f"{DATASET_COMPARISON_PATH}.px.histogram")
@pytest.mark.parametrize(
    "plot_arg, expected_plot_ks_test_statistic_called, expected_histogram_called",
    [("both", True, True), ("cdf", True, False), ("hist", False, True)],
)
def test__compare_divergence_metrics__correct_plots(
    mock_px,
    mock_plot_ks_test_statistic,
    simple_dataset_comparison: DatasetComparison,
    plot_arg: str,
    expected_plot_ks_test_statistic_called: bool,
    expected_histogram_called: bool,
):
    simple_dataset_comparison.compare_divergence_metrics("column_1", plot=plot_arg)
    if expected_plot_ks_test_statistic_called:
        mock_plot_ks_test_statistic.assert_called_once()
    if expected_histogram_called:
        mock_px.assert_called_once()


def test__compare_divergence_metrics__raises_for_unknown_plot_arg(
    simple_dataset_comparison: DatasetComparison,
):
    with pytest.raises(
        ValueError, match=re.escape("Unknown value for 'plot': unknown. Allowed values: ['cdf', 'hist', 'both]")
    ):
        simple_dataset_comparison.compare_divergence_metrics("column_1", plot="unknown")


@pytest.mark.parametrize(
    "invalid_column, missing_dataset", [("old_column", "A new name"), ("new_column", "An old name")]
)
def test__compare_divergence_metrics__raises_for_invalid_column(
    invalid_column: str, missing_dataset: str, simple_dataset_comparison: DatasetComparison
):
    with pytest.raises(ValueError, match=f"Column {invalid_column} does not exist in {missing_dataset}."):
        simple_dataset_comparison.compare_divergence_metrics(invalid_column)


def test__compare_outliers(simple_dataset_comparison: DatasetComparison, capsys: pytest.CaptureFixture):
    simple_dataset_comparison.compare_outliers("column_1")
    captured = capsys.readouterr().out
    assert "Outliers in both datasets   : {'session_4'}\n" in captured
    assert "Outliers only in A new name :  \n" in captured
    assert "Outliers only in An old name: {'session_3'}\n" in captured


@pytest.mark.parametrize(
    "invalid_column, missing_dataset", [("old_column", "A new name"), ("new_column", "An old name")]
)
def test__compare_outliers__raises_for_invalid_column(
    invalid_column: str, missing_dataset: str, simple_dataset_comparison: DatasetComparison
):
    with pytest.raises(ValueError, match=f"Column {invalid_column} does not exist in {missing_dataset}."):
        simple_dataset_comparison.compare_outliers(invalid_column)


@pytest.mark.parametrize(
    "threshold, exclude_missing_sessions, expected_printed_lines",
    [
        (
            0.0,
            False,
            [
                "session_1: Changed from 6 to 1",
                "session_2: Changed from 7 to 2",
                "session_3: Changed from -60 to 3",
                "session_4: Changed from 200 to -100",
                "session_5: Changed from 5 to nan",
            ],
        ),
        (100.0, True, ["session_4: Changed from 200 to -100"]),
        (1e9, True, []),
    ],
)
def test__sessions_with_changes(
    threshold: float,
    exclude_missing_sessions: bool,
    expected_printed_lines: List[str],
    simple_dataset_comparison: DatasetComparison,
    capsys: pytest.CaptureFixture,
):
    simple_dataset_comparison.sessions_with_changes(
        "column_1", threshold=threshold, exclude_missing_sessions=exclude_missing_sessions
    )

    captured = capsys.readouterr().out

    if not expected_printed_lines:
        assert f"There were no sessions whose values have changed by more than {threshold}." in captured
    for l in expected_printed_lines:
        assert l in captured


@pytest.mark.parametrize(
    "invalid_column, missing_dataset", [("old_column", "A new name"), ("new_column", "An old name")]
)
def test__sessions_with_changes__raises_for_invalid_column(
    invalid_column: str, missing_dataset: str, simple_dataset_comparison: DatasetComparison
):
    with pytest.raises(ValueError, match=f"Column {invalid_column} does not exist in {missing_dataset}."):
        simple_dataset_comparison.sessions_with_changes(invalid_column)
