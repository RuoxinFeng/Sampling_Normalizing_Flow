#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Literal, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
import statsmodels.api as sm
from IPython.display import display
from statsmodels.distributions.copula.api import GaussianCopula
from tools.general_utility.custom_eval import evaluate_formula
from tools.general_utility.helpers import display_markdown
from tools.models_export.situation_model_export import (
    get_situation_model_parameter_from_distribution,
    save_model_as_json_file,
)
from tools.plot_tools.plot_tools import qqplot
from tools.situation_model.situation_model import CorrelationMatrix, SituationModel


class ConvenientMarginalDistributionFit(ABC):
    """Abstract class to fit a marginal distribution.

    Parameters
    ----------
    ds : pd.Series
        The training data used for the fit.
    name : str or None, optional
        The name of the marginal distribution
    codebeamer_reference : str or None, optional
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None, optional
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).

    Attributes
    ----------
    ds : pd.Series
        The training data used for the fit.
    name : str or None
        The name of the marginal distribution
    codebeamer_reference : str or None
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).
    """

    def __init__(
        self,
        ds: pd.Series,
        name: Optional[str] = None,
        codebeamer_reference: Optional[str] = None,
        transformation_function: Optional[str] = None,
    ) -> None:
        self.ds = ds.reset_index(drop=True)
        self.name = name if name else ds.name
        self.codebeamer_reference = codebeamer_reference
        self.transformation_function = transformation_function

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Calculate the cumulative distribution function of the fitted distribution.

        Parameters
        ----------
        x : float
            The value where to evaluate the cdf.

        Returns
        -------
        float
            The probability that the variable takes a value less than or equal to x.
        """
        pass

    @abstractmethod
    def icdf(self, x: float) -> float:
        """Calculate the inverse of the cumulative distribution function (percent point function) of the fitted distribution.

        Parameters
        ----------
        x : float
            The probability value where to evaluate the icdf.

        Returns
        -------
        float
            A threshold value, so that the probability of the variable being less or equal than this threshold is x.
        """
        pass

    def u_space(self) -> pd.Series:
        """Outputs the training data, transformed into uniform space (values between 0 and 1).

        Returns
        -------
        pd.Series
            The transformed training data (in the range [0,1]).
        """
        return self.ds.apply(self.cdf).rename("u_" + self.name)

    def z_space(self) -> pd.Series:
        """Outputs the training data, transformed into z space (normally distributed).

        Returns
        -------
        pd.Series
            The transformed training data (with mean 0 and std 1).
        """
        return self.u_space().apply(st.norm.ppf).rename("z_" + self.name)

    def x_u_z_space(self) -> pd.DataFrame:
        """Outputs the training data, transformed into u and z space (normally distributed).

        Returns
        -------
        pd.DataFrame
            A Dataframe containing the untransformed and transformed training data (in u and z space) in its columns.
        """
        return pd.DataFrame(
            {
                self.name: self.ds,
                self.u_space().name: self.u_space(),
                self.z_space().name: self.z_space(),
            }
        )

    def compare_fit_and_data(
        self, x_lim: List[float] = [-1, 1], bins: int = 25, type: Literal["pdf", "cdf", "qq"] = "pdf"
    ) -> None:
        """Generates a plot of the empirical distribution of the data vs the fitted model.

        Parameters
        ----------
        x_lim : List[float], optional
            The range of the x axis to be plotted.
        bins : int, optional
            The nr of bins for discretization.
        type : Literal["pdf", "cdf", "qq"]
            The type of plot to be generated (probability density function, cumulative distribution function or QQ-plot)

        Raises
        ------
        ValueError
            If an unknown type is passed.
        """
        xVal = np.linspace(*x_lim)
        if type != "qq":
            _, ax = plt.subplots(nrows=1, ncols=1)
            if type == "pdf":
                if hasattr(self.dist, "pdf") == True:  # TODO: define abstract interface in base class
                    fun = self.dist.pdf
                    y_label = "PDF"
                else:  # KDF
                    fun = self.dist.density
                    y_label = "Density"
            elif type == "cdf":
                fun = self.dist.cdf
                bins = 100
                y_label = "CDF"
            else:
                raise ValueError(f"Unknown type {type}. Possible types: 'pdf', 'cdf' or 'qq'")
            ax.set_xlabel(self.name)
            ax.set_ylabel(y_label)
            if hasattr(self.dist, "pdf") == True:  # See above
                ax.plot(xVal, fun(xVal), "k-", lw=2, label="Fit")
            else:
                ax.plot(self.dist.support, fun, "k-", lw=2, label="Fit")
            ax.hist(
                self.ds,
                density=True,
                cumulative=(type == "cdf"),
                bins=bins,
                label="Data",
            )
            ax.legend()
            plt.show()
        else:
            fig = qqplot(data=self.ds, dist=self.dist, title="QQ-Plot")
            fig.show()


class ConvenientMarginalDistributionFitSciPy(ConvenientMarginalDistributionFit):
    """Fit a marginal distribution from scipy.stats.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        The distribution family to be fitted.
    ds : pd.Series
        The training data used for the fit.
    name : str or None, optional
        The name of the marginal distribution
    codebeamer_reference : str or None, optional
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None, optional
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).

    Attributes
    ----------
    dist : scipy.stats.rv_continuous
        The distribution family to be fitted.
    ds : pd.Series
        The training data used for the fit.
    name : str or None
        The name of the marginal distribution
    codebeamer_reference : str or None
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).
    pars: Tuple[float]
        The fitted parameters.
    """

    def __init__(
        self,
        dist: st.rv_continuous,
        ds: pd.Series,
        name: Optional[str] = None,
        codebeamer_reference: Optional[str] = None,
        transformation_function: Optional[str] = None,
    ) -> None:
        super().__init__(
            ds=ds, codebeamer_reference=codebeamer_reference, transformation_function=transformation_function, name=name
        )
        self.pars = dist.fit(ds.to_numpy())
        self.dist = dist(*self.pars)

    def cdf(self, x: float) -> float:
        return self.dist.cdf(x)

    def icdf(self, u: float) -> float:
        return self.dist.ppf(u)


class ConvenientMarginalDistributionFitSciPyFrozen(ConvenientMarginalDistributionFit):
    """Similar to ConvenientMarginalDistributionFitSciPy, but already takes a fitted distribution.

    Parameters
    ----------
    dist : scipy.stats.distributions.rv_frozen
        The pre-fitted distribution.
    ds : pd.Series
        The training data used for the fit.
    name : str or None, optional
        The name of the marginal distribution
    codebeamer_reference : str or None, optional
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None, optional
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).

    Attributes
    ----------
    dist : scipy.stats.distributions.rv_frozen
        The pre-fitted distribution.
    ds : pd.Series
        The training data used for the fit.
    name : str or None
        The name of the marginal distribution
    codebeamer_reference : str or None
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).
    pars : Tuple[float]
        The fitted parameters.
    """

    def __init__(
        self,
        dist: st.distributions.rv_frozen,
        ds: pd.Series,
        name: Optional[str] = None,
        codebeamer_reference: Optional[str] = None,
        transformation_function: Optional[str] = None,
    ) -> None:
        super().__init__(
            ds=ds, codebeamer_reference=codebeamer_reference, transformation_function=transformation_function, name=name
        )
        self.pars = dist.args  # No fit, use given distribution
        self.dist = dist

    def cdf(self, x: float) -> float:
        return self.dist.cdf(x)

    def icdf(self, u: float) -> float:
        return self.dist.ppf(u)


class ConvenientMarginalDistributionFitKDEUnivariate(ConvenientMarginalDistributionFit):
    """Fit a kernel density as a marginal distribution.

    Parameters
    ----------
    ds : pd.Series
        The training data used for the fit.
    name : str or None
        The name of the marginal distribution
    codebeamer_reference : str or None, optional
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None, optional
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).

    Attributes
    ----------
    ds : pd.Series
        The training data used for the fit.
    name : str or None
        The name of the marginal distribution
    codebeamer_reference : str or None
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None
        A function to transform the data sampled from this model
        (not used in this class but will be passed during export to a SituationModel).
    kdensity : statsmodels.nonparametric.KDEUnivariate
        The kernel density estimator.
    dist : statsmodels.nonparametric.KDEUnivariate
        The fitted kernel density.
    """

    def __init__(
        self,
        ds: pd.Series,
        name: Optional[str] = None,
        codebeamer_reference: Optional[str] = None,
        transformation_function: Optional[str] = None,
    ) -> None:
        super().__init__(
            ds=ds, codebeamer_reference=codebeamer_reference, transformation_function=transformation_function, name=name
        )
        self.kdensity = sm.nonparametric.KDEUnivariate(ds.to_numpy())
        self.dist = self.kdensity.fit()

    def cdf(self, x: float) -> float:
        return np.interp(x, self.dist.support, self.dist.cdf)

    def icdf(self, u: float) -> float:
        return np.interp(u, np.linspace(0, 1, num=self.dist.icdf.size), self.dist.icdf)


@dataclass
class SamplingInfo:
    df: pd.DataFrame
    warning_messages: Set[str]


class ConvenientNataf:
    """Fit the correlation structure of two or more pre-fitted distributions using a Nataf transformation.

    Parameters
    ----------
    name : str
        The name of the multivariate distribution.
    dists : List[ConvenientMarginalDistributionFit]
        The pre-fitted marginal distributions.
    timestamp_input_data : str or None, optional
        The timestamp when the training data was generated.
    timestamp_export : str or None, optional
        The timestamp when the training data was read.
    filtering_function : str or None, optional
        A function to restrict the variables to a specific range; will be applied during sampling
        from the multivariate distribution.

    Raises
    ------
    ValueError
        If the provided distributions were not fitted on training data of the same length.

    Attributes
    ----------
    filtering_function : str or None
        A function to restrict the variables to a specific range; will be applied during sampling
        from the multivariate distribution.
    timestamps : Dict[str, str]
        A dictionary containing the timestamps when the training data was generated and read.
    """

    def __init__(
        self,
        name: str,
        dists: List[ConvenientMarginalDistributionFit],
        timestamp_input_data: Optional[str] = None,
        timestamp_export: Optional[str] = None,
        filtering_function: Optional[str] = None,
    ) -> None:
        self._dists = dists
        self._name = name
        self.filtering_function = filtering_function
        self._sample_size = dists[0].ds.size
        if not all(dist.ds.size == self._sample_size for dist in dists):
            for dist in dists:
                print(dist.ds.size)
            raise ValueError("Not all fits have same length!")

        self.timestamps = (
            {"timestamp_input_data": timestamp_input_data, "timestamp_export": timestamp_export}
            if timestamp_input_data or timestamp_export
            else {}
        )

    @property
    def name(self) -> str:
        """str : The name of the multivariate distribution."""
        return self._name

    @property
    def dists(self) -> List[ConvenientMarginalDistributionFit]:
        """List[ConvenientMarginalDistributionFit] : The pre-fitted marginal distributions."""
        return self._dists

    @property
    def correlation_matrix(self) -> CorrelationMatrix:
        """CorrelationMatrix : The fitted correlation matrix."""
        return CorrelationMatrix(
            **self.timestamps,
            parameter_order=[d.name for d in self._dists],
            matrix_rows=self.corrcoef().tolist(),
        )

    @property
    def correlation_matrix_without_timestamps(self) -> CorrelationMatrix:
        """CorrelationMatrix : The fitted correlation matrix, not including the timestamps of the training data."""
        return CorrelationMatrix(
            parameter_order=[d.name for d in self._dists],
            matrix_rows=self.corrcoef().tolist(),
        )

    @property
    def _sampling_filters(self) -> Dict[str, str]:
        if not self.filtering_function:
            return {}
        common_filtering_function = self.filtering_function

        # Find all ifvs that need to be wrapped in df['']
        all_ifvs = list(set(re.findall(r"[^\d\W]+\d*", common_filtering_function)))
        for non_ifv_word in ["abs", "np", "log"]:
            while non_ifv_word in all_ifvs:
                all_ifvs.remove(non_ifv_word)

        # Wrap all ifvs in df['']
        for ifv in all_ifvs:
            common_filtering_function = common_filtering_function.replace(ifv, f"df['{ifv}']")

        expressions = common_filtering_function.split(",")
        expressions_before_replacement = self.filtering_function.split(",")
        return {
            new_expression.strip(): old_expression.strip()
            for new_expression, old_expression in zip(expressions, expressions_before_replacement)
            if "df" in new_expression
        }

    def _calculate_correlations_diff(self, max_sampling_iterations: int) -> pd.DataFrame:
        number_of_samples = len(self._dists[0].ds)
        col_names = [d.name for d in self.dists]
        corr_model = self.sample(number_of_samples, seed=1234, max_iterations=max_sampling_iterations).corr()
        corr_data = pd.DataFrame({col: dist.ds for dist, col in zip(self._dists, col_names)}).corr()
        return abs(corr_model - corr_data)

    def check_correlations(self, max_sampling_iterations: int = 100) -> None:
        """Displays the absolute differences between the correlation matrix of the training data and samples from the fitted model.

        Parameters
        ----------
        max_sampling_iterations: int, optional
            The number of iterations to be performed trying to find the expected number of samples. A warning will be raised if the expected number of samples has not been found. Created samples will still be used for calculating the sample correlation matrix in that case.
        """
        corr = self._calculate_correlations_diff(max_sampling_iterations=max_sampling_iterations)
        display_markdown(f"Absolute difference of data correlation matrix and model correlation matrix", font_size=15)
        display(corr.style.background_gradient(cmap="coolwarm"))

    def corrcoef(self) -> np.ndarray:
        """Fits the correlation matrix of the training data in z space.

        Returns
        -------
        np.ndarray
            The calculated correlation matrix.

        See also
        --------
        ConvenientMarginalDistributionFit.z_space : Transforming the training data to z space.
        """
        m = [d.z_space().to_numpy() for d in self._dists]
        return np.corrcoef(m)

    def _sample_gaussian_copula(self, n: int, seed: int) -> np.ndarray:
        correlation_matrix = self.corrcoef()
        dim = len(correlation_matrix)
        copula = GaussianCopula(corr=correlation_matrix, k_dim=dim)
        return copula.rvs(n, random_state=seed).transpose()

    def _append_batch_sample_from_gaussian_copula(self, n: int, df: pd.DataFrame) -> SamplingInfo:
        warning_messages = set()
        U = self._sample_gaussian_copula(n=n, seed=np.random.randint(np.iinfo(np.int32).max))
        new_samples_df = pd.DataFrame({d.name: pd.Series(d.icdf(U[i])) for i, d in enumerate(self._dists)})
        for single_filter, old_expression in self._sampling_filters.items():
            try:
                new_samples_df = new_samples_df[
                    evaluate_formula(single_filter, subscriptable_dict={"df": new_samples_df})
                ]
            except:
                warning_messages.add(f"Failed to apply filtering {old_expression.strip()}.")
        if new_samples_df.shape[0] < n * 0.7:
            warning_messages.add(
                "In at least one iteration of sampling, more than 30% of the samples have been rejected by your filter function. This results in very inefficient sampling."
            )
        df = pd.concat([df, new_samples_df], ignore_index=True)
        return SamplingInfo(df, warning_messages)

    def sample(self, n: int, seed: int, max_iterations: int = 100) -> pd.DataFrame:
        """Generates samples from the fitted distribution.

        Parameters
        ----------
        n : int
            The number of samples to be generated.
        seed : int
            A seed that is passed to statsmodels to generate random variates.
        max_iterations: int, optional
            The number of iterations to be performed trying to find the expected number of samples. A warning will be raised if the expected number of samples has not been found. Created samples will still be outputted in that case.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing samples for each variable in its columns.
        """
        np.random.seed(seed)
        df = pd.DataFrame(columns=[d.name for d in self._dists])
        number_of_iterations = 0
        warning_messages = set()
        while len(df.index) < n:
            number_of_iterations += 1
            if number_of_iterations > max_iterations:
                warnings.warn(f"Could not find valid number of samples in {max_iterations} iterations!")
                break
            df_len = len(df.index)
            sample_info = self._append_batch_sample_from_gaussian_copula(n - df_len, df)
            df = sample_info.df
            warning_messages = warning_messages.union(sample_info.warning_messages)
        for w in warning_messages:
            warnings.warn(w)
        return df

    def plot_model_vs_original(
        self,
        dimensions: Optional[List[str]] = None,
        seed: int = 1234,
        n: Optional[int] = None,
        max_sampling_iterations: int = 100,
    ) -> go.Figure:
        """Generates scatterplots of the training data and data sampled from the fitted distribution together
        with their marginal distributions.

        Parameters
        ----------
        dimensions : List[str] or None, optional
            The variables to be plotted, defaults to all variables.
        seed : int, optional
            A seed that is passed to statsmodels to generate random variates.
        n : int or None, optional.
            The number of samples to be plotted, defaults to the number of samples in the training data.
        max_sampling_iterations: int, optional
            The number of iterations to be performed trying to find the expected number of samples. A warning will be raised if the expected number of samples has not been found. Created samples will still be plotted in that case.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated plots.

        Raises
        ------
        ValueError
            If n is higher than the number of samples in the training_data.
        """

        n_points_to_plot = n or self._sample_size
        if n_points_to_plot > self._sample_size:
            raise ValueError(
                f"Can only plot a maximum of {self._sample_size} points. Requested number of points: {n_points_to_plot}"
            )

        df_model = self.sample(n_points_to_plot, seed=seed, max_iterations=max_sampling_iterations)
        df_model["source"] = "model"

        df_data = pd.DataFrame({d.name: d.ds.to_numpy() for d in self._dists})
        df_data["source"] = "observations"
        df_data = df_data.loc[: n_points_to_plot - 1, :]

        df = pd.concat([df_model, df_data], ignore_index=True)

        dims = dimensions or [d.name for d in self._dists]

        title = "observations vs model of " + self._name
        if len(dims) == 2:
            fig = px.scatter(
                df,
                color="source",
                x=dims[0],
                y=dims[1],
                title=title,
                marginal_x="histogram",
                marginal_y="histogram",
            )
        else:
            fig = px.scatter_matrix(df, color="source", dimensions=dims, title=title)
            marker = dict(size=3, opacity=0.2)
            fig.update_traces(marker=marker, diagonal_visible=False)

        fig.show()
        return fig

    def plot_scatter_matrix(self) -> None:
        """Generates scatterplots of the training data."""
        df = pd.DataFrame({d.name: d.ds for d in self._dists})
        self._plot_scatter_matrix(df=df, title="P-Space of " + self._name)

    def plot_scatter_matrix_z(self) -> None:
        """Generates scatterplots of the training data in z space.

        See also
        --------
        ConvenientMarginalDistributionFit.z_space : Transforming the training data to z space.
        """
        df = pd.DataFrame({d.z_space().name: d.z_space() for d in self._dists})
        self._plot_scatter_matrix(df=df, title="Z-Space " + self._name)

    def plot_scatter_matrix_u(self) -> None:
        """Generates scatterplots of the training data in u space.

        See also
        --------
        ConvenientMarginalDistributionFit.u_space : Transforming the training data to u space.
        """
        df = pd.DataFrame({d.u_space().name: d.u_space() for d in self._dists})
        self._plot_scatter_matrix(df=df, title="U-Space " + self._name)

    @staticmethod
    def _plot_scatter_matrix(df: pd.DataFrame, title: str) -> None:
        fig = px.scatter_matrix(df, title=title)
        fig.update_traces(diagonal_visible=False)
        fig.show()

    def _check_timestamps_present(self) -> None:
        if not all(self.timestamps.values()) or not self.timestamps:
            warnings.warn(
                f"Either input data timestamp or export timestamp was not provided while exporting the model",
                UserWarning,
            )

    def to_situation_model(
        self,
        seed: int,
        n_test_cases: int,
        scenario_occurrence_rate_per_hour: float,
        codebeamer_reference: Optional[str] = None,
        transformation_function: Optional[str] = None,
    ) -> SituationModel:
        """Converts the ConvenientNataf to a SituationModel object.

        Parameters
        ----------
        seed : int
            The seed used to generate noise for the test cases.
        n_test_cases : int
            The number of test cases to be generated.
        scenario_occurrence_rate_per_hour : float
            How often such a scenario occurs in the real world per hour.
        codebeamer_reference : str or None, optional
            A reference to a codebeamer item of the fitted model.
        transformation_function : str or None, optional
            A function to transform the data sampled from this model.

        Returns
        -------
        SituationModel
            The generated situation model.
        """
        parameters = []
        for d in self._dists:
            parameters.append(
                get_situation_model_parameter_from_distribution(
                    distribution=d.dist,
                    name=d.name,
                    codebeamer_reference=codebeamer_reference if codebeamer_reference else d.codebeamer_reference,
                    transformation_function=transformation_function
                    if transformation_function
                    else d.transformation_function,
                    n_test_cases=n_test_cases,
                    seed=seed,
                )
            )

        return SituationModel(
            correlation_matrix=self.correlation_matrix_without_timestamps,
            filtering_function=self.filtering_function or "None",
            parameters=parameters,
            scenario_occurrence_rate_per_hour=scenario_occurrence_rate_per_hour,
            **self.timestamps,
        )

    def save_correlation_matrix(self, folder: str) -> None:
        """Saves the correlation matrix to a json file.

        Parameters
        ----------
        folder : str
            The folder where the file should be saved.
        """
        self._check_timestamps_present()
        save_model_as_json_file(
            model=self.correlation_matrix, file_path=join(folder, self._name + "_correlation_matrix.json")
        )

    def save_model(
        self,
        folder: str,
        codebeamer_reference: str,
        scenario_occurrence_rate_per_hour: float,
        seed: int = 1234,
        n_test_cases: int = 3,
    ) -> None:
        """Saves the model to a json file.

        Parameters
        ----------
        folder : str
            The folder where the file should be saved.
        codebeamer_reference : str
            A reference to a codebeamer item of the fitted model.
        scenario_occurrence_rate_per_hour : float
            How often such a scenario occurs in the real world per hour.
        seed : int, optional
            The seed used to generate noise for the test cases.
        n_test_cases : int, optional
            The number of test cases to be generated.
        """
        self._check_timestamps_present()
        transformation_function = "x"  # Not used yet
        situation_model = self.to_situation_model(
            transformation_function=transformation_function,
            codebeamer_reference=codebeamer_reference,
            seed=seed,
            n_test_cases=n_test_cases,
            scenario_occurrence_rate_per_hour=scenario_occurrence_rate_per_hour,
        )

        save_model_as_json_file(model=situation_model, file_path=join(folder, self._name + ".json"))


# From Nadine Berner, DE-301
def find_best_fit(
    data: pd.Series, bins: int = 200, exclude_dists: List[str] = [], ax: Optional[plt.axes] = None
) -> pd.DataFrame:
    """Fits all distributions from scipy.stats._continuous_distns on the data and calculates the sum of squared errors (SSE)
    as a goodness-of-fit criterion. This function can be used to find a model that best explains 1-dimensional data.
    The following distributions are excluded from the evaluation by default and can not be included manually:
    levy_stable, studentized_range, kstwo, norminvgauss, genhyperbolic

    Parameters
    ----------
    data : pd.Series
        The data to be fitted.
    bins : int, optional
        Number of bins to use for the calculation of SSE.
    exclude_dists : List[str], optional
        Additional distributions from scipy.stats to be excluded from the evaluation.
    ax : plt.axes or None, optional to plot the fitted pdfs.
        A matplotlib.pyplot.axes object

    Returns
    -------
    pd.DataFrame
        A Dataframe containing all fitted distributions together with their SSE.
    """
    # binning of data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    best_distributions = []

    # estimate distribution parameters from data
    for ii, distribution in enumerate(
        [
            d
            for d in st._continuous_distns._distn_names
            if not d in ["levy_stable", "studentized_range", "kstwo", "norminvgauss", "genhyperbolic"] + exclude_dists
        ]
    ):
        print("{:>3} / {:<3}: {}".format(ii + 1, len(st._continuous_distns._distn_names), distribution))

        distribution = getattr(st, distribution)

        # try to fit the distribution
        try:
            # ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # calculate fitted pdf and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))

        except Exception as e:
            print(f"Do I really want to know 😶 -> {distribution.__class__.__name__}\n\t{e}")

        best_distributions = sorted(best_distributions, key=lambda x: x[2])
    return pd.DataFrame(
        [[i[0].__class__.__name__.replace("_gen", ""), i[-1]] for i in best_distributions],
        columns=["distribution", "SSE"],
    )


# TODOs:
# compare_correlations() # p space vs u space correlation
