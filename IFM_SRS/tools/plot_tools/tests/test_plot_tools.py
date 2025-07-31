#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from tools.comparison_tools.divergence_metrics import ks_test
from tools.plot_tools.plot_tools import ifv_histograms, plot_hist_and_fit, plot_ks_test_statistic
from tools.regression_models.regression_models import ConvenientOlsFit


def test__plot_residuals_dist_hist__smoke(simple_setup: ConvenientOlsFit, mock_plots: None):
    simple_setup.plot_residuals_dist_hist()


@pytest.mark.parametrize("cumulative", [True, False])
def test__toggling_between_pdf_and_cdf_works_as_expected(cumulative: bool, simple_setup: ConvenientOlsFit):
    dist_mock = Mock()
    plot_hist_and_fit(
        df=simple_setup.df, col_name="resid", dist=dist_mock, bins=10, x_lim=[-1, 1], cumulative=cumulative
    )
    if cumulative:
        dist_mock.cdf.assert_called()
        dist_mock.pdf.assert_not_called()
    else:
        dist_mock.cdf.assert_not_called()
        dist_mock.pdf.assert_called()


@patch("plotly.express.histogram")
def test__correct_data_used_for_plotting_residuals(hist_mock: MagicMock, simple_setup: ConvenientOlsFit):
    plot_hist_and_fit(df=simple_setup.df, col_name="resid", dist=norm, bins=10, x_lim=[-1, 1], cumulative=True)
    assert "resid" in hist_mock.call_args.args[0].columns
    assert hist_mock.call_args.kwargs["x"] == "resid"


@patch("plotly.express.histogram")
@patch("plotly.express.line")
def test__correct_distribution_function_printed(
    line_mock: MagicMock, hist_mock: MagicMock, simple_setup: ConvenientOlsFit
):
    dist_mock = Mock()
    mock_fun = lambda x: [0] * len(x)
    x = np.linspace(*(x_lim := [-1, 1]))
    df_plot = pd.DataFrame({"x": x, "y": mock_fun(x)})
    dist_mock.cdf = mock_fun
    plot_hist_and_fit(df=simple_setup.df, col_name="resid", dist=dist_mock, bins=10, x_lim=x_lim, cumulative=True)
    np.testing.assert_array_equal(line_mock.call_args.args[0], df_plot)


def test__plot_ks_test_statistic__smoke(mock_plots: None):
    data_1 = pd.Series(range(1, 101))
    data_2 = pd.Series(range(50, 200))
    plot_ks_test_statistic(
        data=(data_1, data_2),
        names=("name_1", "name_2"),
        ks_result=ks_test(data_1, data_2),
        title="title",
        xlabel="x_label",
        ylabel="y_label",
    )


@pytest.mark.parametrize("columns_to_plot", [None, ["some_IFV_12345678", "another_12345688"]])
def test__ifv_histograms__smoke(columns_to_plot: Optional[List[str]], mock_plots: None):
    df = pd.DataFrame({"some_IFV_12345678": ["a", "a", "b", "b"], "another_12345688": [1, 2, 3, 4]})
    ifv_histograms(df, columns_to_plot=columns_to_plot)


@pytest.mark.parametrize(
    "columns_to_plot, expected_plotted_columns",
    [(None, ["some_IFV_12345678", "another_12345688"]), (["another_12345688"], ["another_12345688"])],
)
@patch("plotly.express.histogram")
def test__ifv_histograms__correct_columns(
    hist_mock: MagicMock,
    columns_to_plot: Optional[List[str]],
    expected_plotted_columns: List[str],
    mock_plots: None,
):
    df = pd.DataFrame({"some_IFV_12345678": ["a", "a", "b", "b"], "another_12345688": [1, 2, 3, 4]})
    ifv_histograms(df, columns_to_plot=columns_to_plot)
    assert hist_mock.call_count == len(expected_plotted_columns)
    for column in expected_plotted_columns:
        assert call(df, x=column) in hist_mock.call_args_list
