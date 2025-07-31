#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


from typing import List

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error


def empirical_js_divergence(data_1: pd.Series, data_2: pd.Series, *, nbins: int) -> float:
    """Calculates the Jensen-Shannon divergence of two pd.Series.

    Parameters
    ----------
    data_1 : pd.Series
        The first data series.
    data_2 : pd.Series
        The second data series.
    nbins : int
        The number of bins to be used to calculate the Jensen-Shannon divergence.

    Returns
    -------
    float
        The calculated Jensen-Shannon divergence.

    See also
    --------
    scipy.spatial.distance.jensenshannon : The function used to calculate the Jensen-Shannon divergence.

    Example
    -------
    >>> import pandas as pd
    >>> empirical_js_divergence(pd.Series(range(1,101)),pd.Series(range(50,151)),nbins=10)
    0.5607102079941114
    """
    all_data = pd.concat([data_1, data_2])
    return jensenshannon(
        _estimate_discrete_density(data_1, range=[all_data.min(), all_data.max()], nbins=nbins),
        _estimate_discrete_density(data_2, range=[all_data.min(), all_data.max()], nbins=nbins),
    )


def ks_test(data_1: pd.Series, data_2: pd.Series) -> st._stats_py.KstestResult:
    """Calculates the Kolmogorov-Smirnov test of two pd.Series.

    Parameters
    ----------
    data_1 : pd.Series
        The first data series.
    data_2 : pd.Series
        The second data series.

    Returns
    -------
    scipy.stats._stats_py.KstestResult
        The result of the Kolmogorov-Smirnov test.

    See also
    --------
    scipy.stats.ks_2samp : The function used to calculate the Kolmogorov-Smirnov test.

    Example
    -------
    >>> import pandas as pd
    >>> ks_test(pd.Series(range(1,101)),pd.Series(range(50,150)))
    KstestResult(statistic=0.49, pvalue=2.948425133635738e-11, statistic_location=53, statistic_sign=1)
    """
    return st.ks_2samp(data_1, data_2)


def mse(pred_1: pd.Series, pred_2: pd.Series) -> float:
    """Calculates the Mean Squared Error between two arrays of predictions.

    Parameters
    ----------
    pred_1 : pd.Series
        The first predictions (or the ground truth).
    pred_2 : pd.Series
        The second predictions (or the ground truth).

    Returns
    -------
    float
        The calculated Mean Squared Error.

    See also
    --------
    sklearn.metrics.mean_squared_error : The function used to calculate the Mean Squared Error.

    Example
    -------
    >>> import pandas as pd
    >>> ks_test_statistic(pd.Series(range(1,101)),pd.Series(range(50,150)))
    0.49
    """
    return mean_squared_error(pred_1, pred_2)


def emd(pred_1: pd.Series, pred_2: pd.Series) -> float:
    """Calculates the Earth Mover's Distance (or Wasserstein distance) between two arrays of predictions.

    Parameters
    ----------
    pred_1 : pd.Series
        The first predictions (or the ground truth).
    pred_2 : pd.Series
        The second predictions (or the ground truth).

    Returns
    -------
    float
        The calculated Earth Mover's Distance.

    See also
    --------
    scipy.stats.wasserstein_distance : The function used to calculate the Earth Mover's Distance.

    Example
    -------
    >>> import pandas as pd
    >>> emd(pd.Series(range(1,101)),pd.Series(range(50,150)))
    49.00000000000001
    """
    return st.wasserstein_distance(pred_1, pred_2)


def _estimate_discrete_density(data: pd.Series, *, range: List[float], nbins: int) -> np.ndarray:
    return np.histogram(data, bins=nbins, range=range, density=True)[0]
