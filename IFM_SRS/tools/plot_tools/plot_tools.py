#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from tools.general_utility.extract_columns import get_columns_with_IFVs

data_portal_prefix_url = "https://data-portal.apps.devops.advantagedp.org/#/session-preview?id="


def scatter_matrix(df_: pd.DataFrame) -> go.Figure:
    """Creates scatter plots for all cross-combinations of columns of a DataFrame.

    Parameters
    ----------
    df_ : pd.DataFrame
        The DataFrame whose values should be plotted.

    Returns
    -------
    plotly.graph_objs.Figure
        The figure containing all scatter plots.
    """
    df = df_.copy()
    labels = {col: col.replace("_", " ") for col in df.columns}  # remove underscore

    dimensions = df.columns.delete(0)  # Remove session_id from plot
    fig = px.scatter_matrix(df, labels=labels, dimensions=dimensions, hover_data=["session_id"])

    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    return fig


def parallel_coordinates(df_: pd.DataFrame) -> go.Figure:
    """Creates a parallel coordinate plot for all columns of a DataFrame.

    Parameters
    ----------
    df_ : pd.DataFrame
        The DataFrame whose values should be plotted.

    Returns
    -------
    plotly.graph_objs.Figure
        The figure containing the parallel coordinate plot.
    """
    df = df_.copy()
    labels = {col: col.replace("_", " ") for col in df.columns}  # remove underscore

    dimensions = df.columns.delete(0)  # Remove session_id from plot
    fig = px.parallel_coordinates(df, labels=labels, dimensions=dimensions)
    return fig


def plot_hist_and_fit(
    df: pd.DataFrame,
    col_name: str,
    dist: st.rv_continuous,
    x_lim: Tuple[float, float],
    bins: int = 25,
    cumulative: bool = False,
) -> go.Figure:
    """Fits a distribution on a dataset and plots a histogram of the data together with the pdf or cdf of the
    fitted distribution.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be analyzed.
    col_name : str
        The name of the column of `df` that contains the data to be analyzed.
    dist : st.rv_continuous
        The distribution family to be fitted on the data.
    x_lim : Tuple[float,float]
        The range of the x-axis to be plotted.
    bins : int
        The number of bins to be used for the histogram.
    cumulative : bool, optional
        Whether the cumulative distribution function or the probability density function should be plotted.

    Returns
    -------
    plotly.graph_objs.Figure
        The figure containing the qq plot.
    """
    xVal = np.linspace(*x_lim)
    if cumulative:
        fun = dist.cdf
        marginal = None
        yaxis_title = "Cumulative Density"
    else:
        fun = dist.pdf
        marginal = "box"
        yaxis_title = "Probability Density"

    df_plot = pd.DataFrame(dict(x=xVal, y=fun(xVal)))
    fig = px.histogram(
        df,
        x=col_name,
        marginal=marginal,
        histnorm="probability density",
        nbins=bins,
        range_x=x_lim,
        cumulative=cumulative,
    )
    fig.update_layout(yaxis_title=yaxis_title)
    line = px.line(df_plot, x="x", y="y")
    line["data"][0]["line"]["color"] = "black"
    fig.add_trace(line.data[0])
    return fig


def qqplot(data: pd.Series, dist: st.rv_continuous, title: Optional[str] = None) -> go.Figure:
    """Creates a qq plot of a data series vs a distribution.
    This plot plot has the theoretical quantiles of the distribution on its x-axis
    and the empirical quantiles of the data on its y axis.

    Parameters
    ----------
    data : pd.Series
        The data to be plotted.
    dist : scipy.stats.rv_continuous
        The distribution to be plotted against the data.
    title : str or None, optional
        A title for the plot.

    Returns
    -------
    plotly.graph_objs.Figure
        The figure containing the qq plot.
    """
    qqplot_data = sm.qqplot(data, dist=dist, line="45").gca().lines
    plt.close()
    fig = go.Figure()
    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[0].get_xdata(),
            "y": qqplot_data[0].get_ydata(),
            "mode": "markers",
            "marker": {"color": "blue"},
        }
    )

    fig.add_trace(
        {
            "type": "scatter",
            "x": qqplot_data[1].get_xdata(),
            "y": qqplot_data[1].get_ydata(),
            "mode": "lines",
            "line": {"color": "red"},
        }
    )

    if title:
        title = title
    else:
        title = "Quantile-Quantile Plot"

    fig["layout"].update(
        {
            "title": title,
            "xaxis": {"title": "Theoretical Quantities", "zeroline": False},
            "yaxis": {"title": "Sample Quantities"},
            "showlegend": False,
            "width": 800,
            "height": 700,
        }
    )
    return fig


def plot_ks_test_statistic(
    data: Tuple[pd.Series, pd.Series],
    names: Tuple[str, str],
    ks_result: st._stats_py.KstestResult,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Probability",
) -> None:
    """Creates a plot the cdfs of two data series together with their Kolmogorov-Smirnov test statistic.
    The Kolmogorov-Smirnov test statistic is the biggest vertical distance between the two cdfs.

    Parameters
    ----------
    data : Tuple[pd.Series, pd.Series]
        The data series to be used for the plot.
    names : Tuple[str, str]
        The names of the data series which they should be referred as in the plot.
    ks_result : scipy.stats._stats_py.KstestResult
        The result of the Kolmogorov-Smirnov test whose statistic should be plotted.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The title of the x axis.
    ylabel : str, optional
        The title of the y axis.
    """
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6))
    for data_series, label in zip(data, names):
        ecdf = ECDF(data_series)
        x = np.sort(data_series)
        ax.plot(x, ecdf(x), label=label)
    ecdf_val = ecdf(ks_result.statistic_location)
    ax.plot(
        [ks_result.statistic_location] * 2,
        [ecdf_val, ecdf_val + ks_result.statistic_sign * ks_result.statistic],
        label="KS test statistic",
    )
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def ifv_histograms(df: pd.DataFrame, *, columns_to_plot: Optional[List[str]] = None) -> None:
    """Creates histograms for all IFV columns of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose values should be plotted.
    columns_to_plot : List[str] or None, optional
        An list of column names to be plotted. If this list is None, all IFVs will be plotted.
    """
    ifv_columns = columns_to_plot or get_columns_with_IFVs(df)
    for column in ifv_columns:
        fig = px.histogram(df, x=column)
        fig.show()
