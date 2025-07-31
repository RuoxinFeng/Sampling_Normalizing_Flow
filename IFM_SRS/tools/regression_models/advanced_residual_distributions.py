#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import fmin
from scipy.stats import rv_continuous


def _list_to_array(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if type(arg) == list:
                args[i] = np.asarray(arg)
        for key, value in kwargs.items():
            if type(value) == list:
                kwargs[key] = np.asarray(value)
        result = func(*args, **kwargs)
        return result

    return wrapper


class truncated_distribution(rv_continuous):
    """This class provides a truncated distribution family for a given base distribution family regarding the passed
    lower and upper bounds.

    Parameters
    ----------
    dist : rv_continuous
        The distribution that should be truncated (has to be a non-frozen continuous distribution).
    lower_bound : float, optional
        The lower bound for the truncation, default is minus infinity.
    upper_bound : float, optional
        The upper bound for the truncation, default is plus infinity.

    Attributes
    ----------
    dist : rv_continuous
        The provided base distribution family.

    Notes
    -----
    Only use this class if there is an actual explanation why a distribution has to be truncated.
    For further information about the calculations of the pdf, cdf and ppf see https://cc-github.bmwgroup.net/moritzwerling/reliability-engineering/tree/main/Model_Fits/tools/regression_models#truncated-distribution.

    Examples
    --------
    To create a truncated normal distribution with lower bound -1 and upper bound 2, we do the following:

    >>> from scipy.stats import norm
    >>> from tools.regression_models.advanced_residual_distributions import truncated_distribution
    >>> td = truncated_distribution(dist=norm, lower_bound=-1, upper_bound=2)

    To freeze this truncated distribution, we call it with its parameters again:

    >>> norm_params = (0, 1)
    >>> frozen_td = td(*norm_params)"""

    def __init__(
        self,
        dist: rv_continuous,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        *args,
        **kwds,
    ) -> None:
        if not isinstance(dist, rv_continuous):
            raise TypeError("'dist' cannot be frozen.")
        self.dist = dist
        self.dist.xtol = 1e-10
        if lower_bound >= upper_bound:
            raise ValueError(f"The lower bound {lower_bound} has to be lower than the upper bound {upper_bound}.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        kwds["shapes"] = self.dist.shapes
        kwds["name"] = f"truncated_{self.dist.name}"
        super().__init__(*args, **kwds)

    def _updated_ctor_param(self) -> Dict[str, Any]:
        dct = super()._updated_ctor_param()
        dct["dist"] = self.dist
        dct["lower_bound"] = self._lower_bound
        dct["upper_bound"] = self._upper_bound
        return dct

    def _construct_doc(self, docdict, shapes_vals=None) -> None:
        pass

    @_list_to_array
    def pdf(self, x, *args, **kwds):
        return (
            self.dist.pdf(x, *args, **kwds)
            * np.where((self._lower_bound <= x) & (x < self._upper_bound), 1, 0)
            / (self.dist.cdf(self._upper_bound, *args, **kwds) - self.dist.cdf(self._lower_bound, *args, **kwds))
        )

    @_list_to_array
    def cdf(self, x, *args, **kwds):
        res = (self.dist.cdf(x, *args, **kwds) - self.dist.cdf(self._lower_bound, *args, **kwds)) / (
            self.dist.cdf(self._upper_bound, *args, **kwds) - self.dist.cdf(self._lower_bound, *args, **kwds)
        )
        res_trunc_0 = np.where(res < 0, 0, res)
        return np.where(res_trunc_0 > 1, 1, res_trunc_0)

    @_list_to_array
    def ppf(self, q, *args, **kwds):
        return self.dist.ppf(
            q * (self.dist.cdf(self._upper_bound, *args, **kwds) - self.dist.cdf(self._lower_bound, *args, **kwds))
            + self.dist.cdf(self._lower_bound, *args, **kwds),
            *args,
            **kwds,
        )

    def _fitstart(self, data, *args, **kwds):
        return self.dist._fitstart(data, *args, **kwds)

    @_list_to_array
    def fit(self, data) -> Tuple[float]:
        """Returns fitted parameters based on the passed data by minimizing negative log-likelihood.

        Parameters
        ----------
        data : array_like
            Data to use in estimating the distribution parameters.

        Returns
        -------
        Tuple[float]
            A tuple containing the fitted parameters.
        """

        def neg_log_likelihood(params, x):
            return -np.sum(np.log(self.pdf(x, *params)))

        if min(data) < self._lower_bound or max(data) >= self._upper_bound:
            raise ValueError(
                f"The provided data is not in the specified bounds [{self._lower_bound}, {self._upper_bound})."
            )
        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        x0 = self.dist.fit(data)
        return fmin(neg_log_likelihood, x0, args=(np.ravel(data),), disp=0)

    @property
    def lower_bound(self) -> float:
        """float : The lower bound of the truncated distribution."""
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        """float : The upper bound of the truncated distribution."""
        return self._upper_bound


class piecewise_distribution(rv_continuous):
    """With this class one can concatenate two distributions at a given threshold.

    The shape parameters of the piecewise distribution are the shape parameters of the lower and upper distribution, the respective loc and scale parameters and the probability mass of the lower tail, e.g. for the distribution    `piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=2)` we get the following shape parameters:
    `al, locl, scalel, locu, scaleu, ltmass`

    Parameters
    ----------
    lower_dist : rv_continuous
        The distribution below the threshold (has to be a non-frozen continuous distribution).
    upper_dist : rv_continuous
        The distribution above the threshold (has to be a non-frozen continuous distribution).
    threshold : float
        The threshold at which to concatenate the two distributions.

    Attributes
    ----------
    lower_dist : rv_continuous
            The distribution below the threshold (has to be a non-frozen continuous distribution).
    upper_dist : rv_continuous
        The distribution above the threshold (has to be a non-frozen continuous distribution).
    threshold : float
        The threshold at which to concatenate the two distributions.
    numargs : int
        The number of shape parameters.

    Notes
    -----
    For further information about the calculations of the pdf, cdf and ppf see https://cc-github.bmwgroup.net/moritzwerling/reliability-engineering/tree/main/Model_Fits/tools/regression_models#piecewise-distribution.

    Examples
    --------
    To create a piecewise distribution with 2 as a threshold, lower distribution gamma and upper distribution norm, we can do the following:

    >>> from scipy.stats import gamma, norm
    >>> from tools.regression_models.advanced_residual_distributions import piecewise_distribution
    >>> pd = piecewise_distribution(lower_dist=gamma, upper_dist=norm, threshold=2)

    To freeze this piecewise distribution, we call it with its parameters again:

    >>> params = (2, 0, 0.5, 3, 0.5, 0.48)
    >>> frozen_pd = pd(*params)
    """

    def __init__(self, lower_dist: rv_continuous, upper_dist: rv_continuous, threshold: float, *args, **kwds) -> None:
        if not isinstance(lower_dist, rv_continuous):
            raise TypeError("'lower_dist' cannot be frozen.")
        if not isinstance(upper_dist, rv_continuous):
            raise TypeError("'upper_dist' cannot be frozen.")
        self.threshold = threshold
        self.lower_dist = truncated_distribution(lower_dist, upper_bound=self.threshold)
        self.upper_dist = truncated_distribution(upper_dist, lower_bound=self.threshold)
        kwds["shapes"] = self._get_shapes()
        kwds["name"] = f"piecewise_{lower_dist.name}_{upper_dist.name}"
        self.numargs = len(kwds["shapes"].split(","))
        super().__init__(*args, **kwds)

    def _get_shapes(self) -> str:
        lower_shapes = ",".join([c + "l" for c in self.lower_dist.shapes.split(",")]) if self.lower_dist.shapes else ""
        upper_shapes = ",".join([c + "u" for c in self.upper_dist.shapes.split(",")]) if self.upper_dist.shapes else ""
        shapes = f"{lower_shapes}, " if lower_shapes else ""
        shapes += "locl, scalel, "
        shapes += f"{upper_shapes}, " if upper_shapes else ""
        shapes += "locu, scaleu, "
        shapes += "ltmass"
        return shapes

    def _updated_ctor_param(self) -> Dict[str, Any]:
        dct = super()._updated_ctor_param()
        dct["lower_dist"] = self.lower_dist.dist
        dct["upper_dist"] = self.upper_dist.dist
        dct["threshold"] = self.threshold
        return dct

    def _construct_doc(self, docdict, shapes_vals=None) -> None:
        pass

    def _get_dist_args(self, args: Tuple[float]) -> Tuple[Tuple[float], Tuple[float], float]:
        lower_args_num = self.lower_dist.numargs + 2
        lower_args = args[:lower_args_num]
        upper_args_num = self.upper_dist.numargs + 2
        upper_args = args[lower_args_num : lower_args_num + upper_args_num]
        ltmass = args[-1]
        if ltmass < 0 or ltmass > 1:
            raise ValueError("ltmass must be in [0, 1].")
        return lower_args, upper_args, ltmass

    @_list_to_array
    def pdf(self, x, *args, **kwds):
        args, _, _ = self._parse_args(*args, **kwds)
        lower_args, upper_args, ltmass = self._get_dist_args(args)
        return ltmass * self.lower_dist.pdf(x, *lower_args) + (1 - ltmass) * self.upper_dist.pdf(x, *upper_args)

    @_list_to_array
    def cdf(self, x, *args, **kwds):
        args, _, _ = self._parse_args(*args, **kwds)
        lower_args, upper_args, ltmass = self._get_dist_args(args)
        return ltmass * self.lower_dist.cdf(x, *lower_args) + (1 - ltmass) * self.upper_dist.cdf(x, *upper_args)

    @_list_to_array
    def ppf(self, q, *args, **kwds):
        args, _, _ = self._parse_args(*args, **kwds)
        lower_args, upper_args, ltmass = self._get_dist_args(args)
        return np.where(
            (q <= ltmass),
            self.lower_dist.ppf(q / ltmass, *lower_args),
            self.upper_dist.ppf((q - ltmass) / (1 - ltmass), *upper_args),
        )

    @_list_to_array
    def fit(self, data) -> Tuple[float]:
        """Returns the fitting results of the lower and upper distribution and the probability mass of the lower tail.

        Parameters
        ----------
        data : array_like
            Data to use in estimating the distribution parameters.

        Raises
        ------
        ValueError
            If there is no data above or below the threshold.

        Returns
        -------
        Tuple[float]
            A tuple containing the fitting results of the lower and upper distribution and the probability mass of the lower tail.
        """
        lower_data = data[data < self.threshold]
        upper_data = data[self.threshold <= data]

        if len(lower_data) == 0 or len(upper_data) == 0:
            raise ValueError("There has to be data above and below the threshold.")

        ltmass = len(lower_data) / len(data)

        lower_args = self.lower_dist.fit(lower_data)
        upper_args = self.upper_dist.fit(upper_data)

        return (*lower_args, *upper_args, ltmass)
