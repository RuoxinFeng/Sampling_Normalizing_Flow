#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import copy
import json
import os
import re
import types
import warnings
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as st
import statsmodels.api as sm
from IPython.display import display
from matplotlib import pyplot as plt
from patsy import ModelDesc, dmatrices, dmatrix
from patsy.desc import Term
from plotly.subplots import make_subplots
from scipy.linalg import null_space
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tools.general_utility.helpers import NR_OF_DIGITS_AFTER_COMMA, display_dataframe, display_markdown, get_now_string
from tools.marginal_distributions.marginal_distributions import (
    ConvenientMarginalDistributionFitSciPyFrozen,
    find_best_fit,
)
from tools.models_export.statsmodels_exporter import SpmModel
from tools.plot_tools.plot_tools import plot_hist_and_fit, qqplot
from tools.regression_models.regression_helpers import generate_test_data, predict_statsmodels
from tools.remove_invalid_sessions.remove_invalid_sessions import TotalFailureInfo

OUTPUT_FOLDER = "./models/"
CB_ID_REGEX = r"[0-9a-zA-Z_]+_[0-9]{7,}[a-zA-Z]?"
LINES_SEPARATOR = "\n-----------------------------------------------------------------\n"


class ConvenientRegressionFit(ABC):
    """An abstract class to fit regressions.

    Parameters
    ----------
    model_name : str
        Name of the model.
    scenario_name : str
        Name of the scenario.
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Regression formula.
    timestamp_input_data : str or None
        Timestamp of the input data.
    timestamp_export : str or None
        Timestamp of the model export.
    total_failure_info : TotalFailureInfo or None
        Total failure info received from remove_invalid_sessions.
    inverse_output_trafo : Callable or None, optional
        Inverse output transformation of the SPV.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Model formula.
    inverse_output_trafo : Callable or None
        Inverse output transformation of the SPV.
    response_name : str
        Name of the SPV.
    model_name : str
        Name of the model.
    residual_write_file : str
        Path to the csv of residuals.
    model_write_file : str
        Path to the model.json file.
    tests_write_file : str
        Path to the tests.json file.
    timestamp_input_data : str or None
        Timestamp of the input data.
    timestamp_export : str or None
        Timestamp of the model export.
    posterior_total_failure_probability : float
        Posterior total failure probability.
    observed_total_failures : int
        Observed total failures.
    """

    def __init__(
        self,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        formula: str,
        timestamp_input_data: Optional[str],
        timestamp_export: Optional[str],
        total_failure_info: Optional[TotalFailureInfo],
        inverse_output_trafo: Optional[Callable] = None,
    ) -> None:
        if df.isnull().values.any():
            raise ValueError("There are nan values in the data frame!")
        self.df = df.copy()

        self._check_cb_ids_in_formula(formula)

        self.formula = formula  # This is only used to bring the test vector data into the right form
        self.inverse_output_trafo = inverse_output_trafo
        self.response_name = self._get_response_name()
        self._fit()

        self.model_name = model_name
        self.residual_write_file = model_name + "_extreme_resids.csv"
        self.model_write_file = OUTPUT_FOLDER + scenario_name + "/" + model_name + "_model.json"
        self.tests_write_file = OUTPUT_FOLDER + scenario_name + "/" + model_name + "_tests.json"

        self.timestamp_input_data = timestamp_input_data
        self.timestamp_export = timestamp_export
        self.posterior_total_failure_probability = self.observed_total_failures = None
        if total_failure_info is not None:
            self.posterior_total_failure_probability = total_failure_info.posterior_total_failure_probability
            self.observed_total_failures = total_failure_info.observed_total_failures

    def _check_cb_ids_in_formula(self, formula: str) -> None:
        model = ModelDesc.from_formula(formula)
        for spvs in model.lhs_termlist:
            for spv in spvs.factors:
                spv_name = spv.code
                if not re.findall(CB_ID_REGEX, spv_name):
                    raise ValueError(f"Codebeamer ID missing in SPV {spv_name}.")

        for ifvs in model.rhs_termlist:
            for ifv in ifvs.factors:
                ifv_name = ifv.code
                if not re.findall(CB_ID_REGEX, ifv_name):
                    raise ValueError(f"Codebeamer ID missing in IFV {ifv_name}.")

    @property
    @abstractmethod
    def noise_dist_name(self) -> str:
        """str : Name of the noise distribution."""
        pass

    @abstractmethod
    def _fit_noise_dist(self) -> None:
        pass

    @abstractmethod
    def _return_fit(self, formula: str, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, RegressionResultsWrapper]:
        pass

    @property
    def predictors(self) -> List[str]:
        """List[str] : List of the predictors of the model."""
        return sorted(
            list(
                set(
                    ConvenientRegressionFit.__remove_category_from_string(s)
                    for s in self.result.pvalues.drop("Intercept").index
                )
            )
        )

    @property
    def leverage(self) -> np.array:
        """np.array : The leverage of the design matrix."""
        X = np.array(self.X)
        H = (X @ np.linalg.inv(X.T @ X)) @ X.T
        return np.array([np.diag(H)]).T

    @property
    def loocv(self) -> float:
        """float : The leave one out cross validation error."""
        y = np.array(self.y)
        y_predict = np.array([self.df["y_predict"]]).T
        h = self.leverage
        return np.mean(np.power((y - y_predict) / (1 - h), 2))

    @property
    def _endog_part_of_formula(self) -> str:
        return self.formula.split("~")[0].replace(" ", "").replace("\n", "")

    def _get_response_name(self) -> str:
        endog_part = self._endog_part_of_formula
        match = re.match(r"([a-zA-Z_]+\.)?(.+)\((.+)\)", endog_part)
        if match and self.inverse_output_trafo:
            module_name = match.group(1)[:-1] if match.group(1) is not None else match.group(1)  # remove the .
            function_name = match.group(2)
            arg = match.group(3)
            if module_name != "np":
                raise ValueError("Only numpy functions are supported")
            elif not hasattr(np, function_name):
                raise ValueError(f"No {function_name} found in numpy")
            else:
                function = getattr(np, function_name)
                self._check_inverse_output_trafo(function)
                return arg
        elif self.inverse_output_trafo:
            raise ValueError(
                "Invalid input format for working with an inverse output trafo. Valid format np.log(var_name)"
            )
        return endog_part

    def _check_inverse_output_trafo(self, output_trafo: Callable) -> None:
        if self.inverse_output_trafo is not None:
            test_data = np.linspace(-100, 100, 100_000)
            with warnings.catch_warnings():  # As we don't know the support of output_trafo, this might raise some warnings which we can suppress
                warnings.simplefilter("ignore")
                trafo_data = output_trafo(test_data)
                # remove nan values, since the output_trafo is not defined for those values
                remaining_trafo_data = trafo_data[~np.isnan(trafo_data)]
                remaining_test_data = test_data[~np.isnan(trafo_data)]
                inverse_data = self.inverse_output_trafo(remaining_trafo_data)
            if not np.all(np.isclose(inverse_data, remaining_test_data)):
                raise ValueError(
                    f"The given function {self.inverse_output_trafo} is not the inverse of the transformation function {output_trafo}."
                )

    def print_summary(self) -> None:
        """Prints a summary of the model."""
        display(self.result.summary())

    def plot_residuals(self) -> None:
        """Plots the residuals in a scatter plot."""
        fig = px.scatter(self.df, x="y_predict", y="resid", title=self.response_name)
        fig.show()

    def plot_residuals_hist(self, nbins: int = 25) -> None:
        """Plots the residuals in a histogram plot.

        Parameters
        ----------
        nbins : int, optional
            Number of bins.
        """
        fig = px.histogram(
            self.df,
            x="resid",
            title=self.response_name,
            nbins=nbins,
            histnorm="probability density",
            marginal="box",
        )
        fig.show()

    def plot_residuals_dist_hist(
        self, x_limits: Tuple[float, float] = (-5, 5), nbins: int = 25, bcumulative: bool = False
    ) -> None:
        """Plots a histogram of the residuals together with the pdf or cdf of the fitted residual distribution.

        Parameters
        ----------
        x_limits : List[float], optional
            Range of the x-axis to be plotted.
        nbins : int, optional
            Number of bins.
        bcumulative : bool, optional
            Whether the cumulative distribution function or the probability density function should be plotted.
        """
        fig = plot_hist_and_fit(
            df=self.df, col_name="resid", dist=self.noise_dist, x_lim=x_limits, bins=nbins, cumulative=bcumulative
        )
        fig.show()

    def resid_qqplot(self) -> None:
        """Plots the residuals in a qq-plot."""
        fig = qqplot(
            self.df["resid"],
            dist=self.noise_dist,
            title="QQ-Plot of residuals for " + self.noise_dist_name,
        )
        fig.show()

    def _set_figure_axes(self, fig: plt.figure, fontsize: float) -> None:
        all_axes = fig.get_axes()
        for ax in all_axes:
            ax.set_xlabel("\n : \n".join(ax.xaxis.get_label().get_text().split(":")), fontsize=fontsize)
            ax.set_ylabel("\n : \n".join(ax.yaxis.get_label().get_text().split(":")), fontsize=fontsize)

    def plot_partregress_grid(
        self, figsize: Tuple[int, int] = (10, 12), figpad: float = 1.0, fontsize: float = 20.0
    ) -> None:
        """Generates all partial regression plots of the model (SPM vs one regressor).

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Size of the figure.
        figpad : float, optional
            Padding between the figure edge and the edges of subplots, as a fraction of the font size.
        fontsize : float, optional
            Size of the font.
        """
        fig = plt.figure(figsize=figsize)
        sm.graphics.plot_partregress_grid(self.result, fig=fig)
        self._set_figure_axes(fig=fig, fontsize=fontsize)
        fig.tight_layout(pad=figpad)
        plt.show()

    def plot_ccpr_grid(self, figsize: Tuple[int, int] = (10, 12), figpad: float = 1.0, fontsize: float = 20.0) -> None:
        """Generates a grid of component and component-plus-residual (CCPR) plots.

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Size of the figure.
        figpad : float, optional
            Padding between the figure edge and the edges of subplots, as a fraction of the font size.
        fontsize : float, optional
            Size of the font.
        """
        fig = plt.figure(figsize=figsize)
        sm.graphics.plot_ccpr_grid(self.result, fig=fig)
        self._set_figure_axes(fig=fig, fontsize=fontsize)
        fig.tight_layout(pad=figpad)
        plt.show()

    def plot_model_vs_original(self, seed: int = 123, n: Optional[int] = None) -> None:
        """Visual plausibility check of the model, where the model output is random samples from the model for given regressor values in the origin data and the original output is samples of the independent variable in the original data.

        Parameters
        ----------
        seed : int, optional
            Seed for the model prediction.
        n : int or None, optional
            Number of samples to plot.
        """
        df_plot = self._get_model_prediction(seed)

        n_points_to_plot = n or df_plot.shape[0]
        if n_points_to_plot > df_plot.shape[0]:
            raise ValueError(
                f"Can only plot a maximum of {df_plot.shape[0]} points. Requested number of points: {n_points_to_plot}"
            )

        number_of_df_columns = df_plot.shape[1]
        fig = make_subplots(rows=number_of_df_columns, cols=1)
        regressors = df_plot.drop(
            ["session_id", "resid", "model_output", "y_predict", self.response_name],
            axis=1,
        ).columns
        for r in regressors:
            fig = px.scatter(
                df_plot.loc[: n_points_to_plot - 1, :], x=r, y=["model_output", self.response_name], opacity=0.8
            )
            fig.update_layout(
                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                )
            )
            fig.show()

    @property
    def _standardized_data(self) -> pd.DataFrame:
        data = self.df.copy()
        # Identify numerical columns in the DataFrame
        # Notice that we may not scale the response variable!
        cols_to_be_standardized = data.drop(self.response_name, axis=1).select_dtypes(include=[int, float]).columns
        # Scale only the numerical columns
        if len(cols_to_be_standardized) > 0:
            scaler = StandardScaler()
            data[cols_to_be_standardized] = scaler.fit_transform(data[cols_to_be_standardized])

        return data

    def _get_fitted_standardized_data_coefficients_and_confidence_intervals(self) -> Tuple[pd.Series, pd.DataFrame]:
        _, _, result = self._return_fit(formula=self.formula, data=self._standardized_data)
        coefficients = result.params.drop("Intercept")
        conf_int = result.conf_int(alpha=0.05).drop("Intercept")  # 95% confidence intervals

        sorted_indices = np.argsort(np.abs(coefficients))
        return (coefficients.iloc[sorted_indices], conf_int.iloc[sorted_indices])

    def plot_standardized_coefficients(self) -> None:
        """Plots standardized coefficients in a horizontal bar plot."""
        coefficients, confidence_intervals = self._get_fitted_standardized_data_coefficients_and_confidence_intervals()

        # Plot the parameter output with confidence intervals (horizontal bar plot)
        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot the horizontal bars for coefficients
        ax.barh(
            coefficients.index,
            coefficients.values,
            xerr=(coefficients - confidence_intervals[0], confidence_intervals[1] - coefficients),
            capsize=5,
            align="center",
            alpha=0.8,
            ecolor="black",
            color="blue",
        )
        ax.set_xlabel("Coefficient Value")
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=15)  # Increase the font size for tick labels
        plt.show()

    def _fit(self) -> None:
        self.y, self.X, self.result = self._return_fit(formula=self.formula, data=self.df)
        self.df["resid"] = self.result.resid
        self.df["y_predict"] = self.result.fittedvalues
        self._fit_noise_dist()

    def drop_term(self, term: str) -> None:
        """Drops the given term from the formula and refits the model.

        Parameters
        ----------
        term : str
            Term to drop.
        """
        self.formula = self.formula + " - " + term
        self._fit()

    def print_p_sorted_terms(self) -> None:
        """Prints the sorted p-values of the model."""
        p_values = self.result.pvalues.drop("Intercept")
        display(p_values.sort_values().apply(lambda x: f"{x:.3f}"))

    @staticmethod
    def is_predictor_hierarchically_dependent(predictor: str, predictors: List[str]) -> bool:
        """Returns if the predictor is hierarchically dependent on the list of predictors.

        Parameters
        ----------
        predictor : str
            Predictor to be analyzed for hierarchical dependency
        predictors : List[str]
            List of predictors

        Returns
        -------
        bool
            If the predictor is hierarchically dependent on the predictors.
        """
        predictors.remove(predictor)
        return any(
            [
                ConvenientOlsFit.__is_subterm(potential_subterm=predictor, potential_superterm=potential_superterm)
                for potential_superterm in predictors
            ]
        )

    @staticmethod
    def __is_subterm_of_other_terms_with_lower_value(
        term_to_check: str, value_to_term_to_check: float, other_terms: pd.Series
    ) -> bool:
        a = type(other_terms)
        for other_term, other_value in other_terms.items():
            if (
                ConvenientOlsFit.__is_subterm(potential_subterm=term_to_check, potential_superterm=other_term)
                and other_value < value_to_term_to_check
            ):
                return True
        return False

    @staticmethod
    def __is_subterm(potential_subterm: str, potential_superterm: str) -> bool:
        # Checks if all elements of list1 are contained in list2 with the same occurrence ignoring categories
        list1 = potential_subterm.split(":")
        list2 = potential_superterm.split(":")
        list1_without_categories = [ConvenientOlsFit.__remove_category_from_string(element) for element in list1]
        list2_without_categories = [ConvenientOlsFit.__remove_category_from_string(element) for element in list2]
        # Hint: The occurrence is important for terms like "A*A*B"
        for element in list1_without_categories:
            if list1_without_categories.count(element) > list2_without_categories.count(element):
                return False
        return True

    @staticmethod
    def __remove_category_from_string(string: str) -> str:
        return re.sub(r"\[.*?\]", "", string)

    @staticmethod
    def least_significant(p_values: pd.Series) -> Tuple[str, float]:
        """Gets the least significant term without categories and the p-value of said term.

        Parameters
        ----------
        p_values : pd.Series
            Series of p-values.

        Returns
        -------
        Tuple[str, float]
            Tuple of least significant term and its p-value.
        """
        # Extract elements that are not subterms of any other term
        removable_p_values = dict()
        for term, p_value in p_values.items():
            if not ConvenientOlsFit.__is_subterm_of_other_terms_with_lower_value(
                term_to_check=term, value_to_term_to_check=p_value, other_terms=p_values
            ):
                removable_p_values.update({term: p_value})
        if len(removable_p_values) == 0:
            return (None, -np.inf)
        least_significant_term = pd.Series(removable_p_values).idxmax()
        # Remove [T.xxx] and return with p-value
        least_significant_term_without_categories = ConvenientOlsFit.__remove_category_from_string(
            least_significant_term
        )
        return (
            least_significant_term_without_categories,
            removable_p_values[least_significant_term],
        )

    def backward_elimination(self, alpha: float = 0.05) -> None:
        """Performs backward elimination on the model.

        Parameters
        ----------
        alpha : float, optional
            Threshold for the backward elimination.
        """
        linear_combinations = _get_linearly_dependent_predictors(self.X.drop(["Intercept"], axis=1))
        if linear_combinations:
            warnings.warn(
                f"Some of the predictors are linearly dependent, which is not allowed in OLS regression. See:\n {LINES_SEPARATOR.join(linear_combinations)}"
            )

        sorted_p_values = self.result.pvalues.drop("Intercept").sort_values()
        step = 0
        df_log = pd.DataFrame(
            {step: sorted_p_values.apply(lambda x: f"{x:.3f}")}, index=sorted_p_values.index.tolist()
        )  # tolist required?

        term, p_value = self.least_significant(self.result.pvalues.drop("Intercept"))
        while p_value >= alpha:
            step += 1
            self.drop_term(term)
            new_pvalues = self.result.pvalues.drop("Intercept")
            term, p_value = self.least_significant(new_pvalues)
            df_log[step] = new_pvalues.apply(lambda x: f"{x:.3f}")
        df_log = df_log.fillna("")
        df_log.index.name = None  # Remove index label
        display(df_log)

    def combine_categories(self, column_name: str, categories: List[str], combined_category: str) -> None:
        """Combines the categories for the given column name of the IFV DataFrame and fits the model again.

        Parameters
        ----------
        column_name : str
            Column name in the IFV DataFrame.
        categories : List[str]
            List of categories.
        combined_category : str
            Name of the combined category.
        """
        for category in categories:
            self.df[column_name].replace(to_replace=category, value=combined_category, inplace=True)
        self._fit()

    def print_outlier_session_ids(self, n: int = 10, ascending: bool = False, csv_export: bool = False) -> None:
        """Prints outlier session IDs and writes session IDs with extreme residuals to a file if specified.

        Parameters
        ----------
        n : int, optional
            Number of session IDs to print.
        ascending : bool, optional
            Wether to sort the IFV DataFrame in ascending or descending order.
        csv_export : bool, optional
            Wether to export the session IDs with extreme residuals as a csv.
        """
        df = self.df.copy()
        df = df[["resid"] + [col for col in df.columns if col != "resid"]]  # Move resid column to the front
        df = df.sort_values(by=["resid"], ascending=ascending).head(n)
        display(df)
        if csv_export:
            file = self.residual_write_file
            df["session_id"].to_csv(file, index=False, header=False)
            print("Session IDs with extreme residuals written to " + file)

    def export_model(self, nr_of_digits_after_comma: Optional[int] = NR_OF_DIGITS_AFTER_COMMA) -> None:
        """Exports the model as a json file.

        Parameters
        ----------
        nr_of_digits_after_comma : int or None, optional
            Number of digits after the comma in the json file.
        """
        self.__create_folder(os.path.dirname(self.model_write_file))
        SpmModel.from_linear_model_result(
            model_result=self.result,
            name=self.response_name,
            total_failure_rate=self.posterior_total_failure_probability,
            timestamp_input_data=self.timestamp_input_data,
            timestamp_export=self.timestamp_export,
            custom_noise_distribution=self.noise_dist,
        ).as_json(self.model_write_file, nr_of_digits_after_comma)
        print("Model written to " + self.model_write_file + " on " + (self.timestamp_export or get_now_string()))

    def __formula_rhs(self) -> str:
        return self.formula.split("~")[1].strip()  # Use right hand side of formula (y ~ (this part))

    def export_tests(self, input_data: Dict[str, List[Any]], seed: str) -> None:
        """Generates and exports tests for the model as a json file.

        Parameters
        ----------
        input_data : Dict[str, List[Any]]
            Input data to generate the tests.
        seed : str
            Seed for generating the tests.
        """
        X = dmatrix(self.__formula_rhs(), data=pd.DataFrame(input_data), return_type="dataframe")
        test_data = generate_test_data(
            model=self.result,
            input_data=input_data,
            dmatrix=X,
            seed=seed,
            inv_trafo=self.inverse_output_trafo,
            noise_distribution=self.noise_dist,
            timestamp_input_data=self.timestamp_input_data,
            timestamp_export=self.timestamp_export,
        )
        self.__create_folder(os.path.dirname(self.tests_write_file))
        if self.timestamp_input_data is None or self.timestamp_export is None:
            warnings.warn(
                f"Either input data timestamp or export timestamp was not provided while exporting tests", UserWarning
            )
        with open(self.tests_write_file, "w") as f:
            f.write(json.dumps(test_data, indent=4))
            f.write("\n")
            print("Tests written to " + self.tests_write_file + " on " + (self.timestamp_export or get_now_string()))

    @staticmethod
    def __reduce_rows_keeping_rank(matrix: pd.DataFrame) -> Tuple[int, List[int], pd.DataFrame]:
        matrix = np.array(matrix)
        num_rows, num_columns = matrix.shape
        rank = 0
        reduced_matrix = np.empty((0, num_columns))
        indices = []

        for i in range(num_rows):
            # Temporary add new row
            temp_matrix = np.vstack((reduced_matrix, matrix[i]))
            temp_rank = np.linalg.matrix_rank(temp_matrix)

            if temp_rank > rank:
                # Keep if rank is increased
                indices.append(i)
                reduced_matrix = temp_matrix
                rank = temp_rank

            if rank == num_columns:
                # Break if full rank is already achieved
                break

        return rank, indices, reduced_matrix

    def factor_names_from_formula(self) -> Tuple[List[str], List[str]]:
        """Gets the factor names from the formula.

        Returns
        -------
        Tuple[List[str], List[str]]
            Tuple of terms of the left hand side and right hand side.
        Notes
        -----
        This function cannot extract variables from expressions like "np.power(x1,2)".

        """
        # Caveat: cannot extract variables from expressions like "np.power(x1,2)"

        # X_temp = dmatrix(self.__formula_rhs(), data=data, return_type="dataframe")
        # return [
        #     factor_info.code for factor_info in X_temp.design_info.factor_infos
        # ]  # see https://patsy.readthedocs.io/en/latest/API-reference.html#patsy.FactorInfo
        def get_params_from_terms(term_list: List[Term]) -> List[str]:
            return sorted(
                list({factor.name() for term in term_list for factor in term.factors if factor.name() != "Intercept"})
            )

        model_description = ModelDesc.from_formula(self.formula)
        return get_params_from_terms(model_description.lhs_termlist), get_params_from_terms(
            model_description.rhs_termlist
        )

    def export_tests_from_input_data(self, seed: str = 1234, substitute_columns: Dict[str, str] = {}) -> None:
        """Substitutes columns in the regression formula and exports a test json-file.

        Parameters
        ----------
        seed : str, optional
            Seed for test data generation
        substitute_columns : Dict[str, str], optional
            Dictionary for which the keys are the old and the values the new factor names.
        """
        # TODO: This part needs refactoring and tests!
        # Add column is a workaround when you have expressions like "np.power(x1,2)"".
        # I (Moritz) did not find a way how to extract "x1" easily from the design matrix.
        _, test_rows_indices, _ = self.__reduce_rows_keeping_rank(self.X)
        test_rows_dataframe = self.df.reset_index(drop=True).iloc[test_rows_indices]

        # Reduce Design Matrix to these rows...
        _, factor_names = self.factor_names_from_formula()

        if substitute_columns:
            for old, new in substitute_columns.items():
                if old in factor_names:
                    index_to_replace = factor_names.index(old)
                    factor_names[index_to_replace] = new
                else:
                    warnings.warn(f"{old} was not found in the regression formula!")

        factor_names = list(dict.fromkeys(factor_names))  # Remove duplicates

        # ... and also the original data frame
        test_rows_dataframe = test_rows_dataframe[factor_names]  # Take only columns that are found in the Design Matrix

        # use this instead: patsy.ModelDesc.from_formula
        formula_rhs = (
            self.__formula_rhs()
        )  # Use right hand side of formula (y ~ (this part)), TODO: Use design_info.factor_infos instead!
        X = dmatrix(formula_rhs, data=test_rows_dataframe, return_type="dataframe")
        test_data = generate_test_data(
            model=self.result,
            input_data=test_rows_dataframe.to_dict("list"),
            dmatrix=X,
            seed=seed,
            inv_trafo=self.inverse_output_trafo,
            noise_distribution=self.noise_dist,
            timestamp_input_data=self.timestamp_input_data,
            timestamp_export=self.timestamp_export,
        )
        self.__create_folder(os.path.dirname(self.tests_write_file))
        self.__create_file(test_data)

    def __create_folder(self, folder: str) -> None:
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"Error occurred: {str(e)}")

    def __create_file(self, test_data: dict) -> None:
        if self.timestamp_input_data is None or self.timestamp_export is None:
            warnings.warn(
                f"Either input data timestamp or export timestamp was not provided while exporting tests from input data",
                UserWarning,
            )
        with open(self.tests_write_file, "w") as f:
            f.write(json.dumps(test_data, indent=4))
            f.write("\n")
            print("Tests written to " + self.tests_write_file + " on " + (self.timestamp_export or get_now_string()))

    def to_marginal_distribution(self) -> ConvenientMarginalDistributionFitSciPyFrozen:
        """Converts the model to a frozen marginal distribution.

        Returns
        -------
        ConvenientMarginalDistributionFitSciPyFrozen
            Frozen marginal distribution.
        """
        return ConvenientMarginalDistributionFitSciPyFrozen(
            dist=self.noise_dist, ds=self.df["resid"].rename("resid_" + self.response_name)
        )

    def _display_model_info(self) -> None:
        display_markdown(f"Summary for Model: {self.model_name}", 18)
        display_markdown("General information:", 14)
        if self.inverse_output_trafo:
            display_markdown(
                f"This model predicts {self._endog_part_of_formula}. To get the value for {self.response_name}, the output needs to be transformed with np.{self.inverse_output_trafo.__name__}().",
                10,
            )
        display_dataframe(
            pd.DataFrame(
                {
                    "Property": [
                        "Number of observations",
                        "R-Squared",
                        "R-Squared adj.",
                        "Noise distribution",
                        "Posterior total failure probability with uniform prior",
                        "Observed total failures",
                    ],
                    "Value": [
                        int(self.result.nobs),
                        self.result.rsquared,
                        self.result.rsquared_adj,
                        self.noise_dist_family.name,
                        self.posterior_total_failure_probability,
                        self.observed_total_failures,
                    ],
                }
            )
        )
        display_markdown("P-values", 14)

        df = self.result.pvalues.to_frame(name="p-value")
        df["coefficient"] = self.result.params
        df_pval = df.reset_index().rename(columns={"index": "Exog name"})
        display_dataframe(df_pval)

        display_markdown("Original output vs. Model output", 14)

    def _get_model_prediction(self, seed: int) -> pd.DataFrame:
        uniform_noise = st.uniform.rvs(size=self.df.shape[0], random_state=seed)

        model_output = predict_statsmodels(
            model=self.result,
            X=self.X,
            U=uniform_noise,
            inv_trafo=self.inverse_output_trafo,
            noise_distribution=self.noise_dist,
        )
        df_plot = self.df.copy()
        df_plot["model_output"] = model_output
        return df_plot

    def plot_model_vs_original_histogram(self, bin_size: float = 0.05, seed: int = 123) -> None:
        """Plots data sampled from the model and training data for comparison.

        Parameters
        ----------
        bin_size : float, optional
            Size of the bins.
        seed : int, optional
            Seed for the model prediction.
        """
        df_plot = self._get_model_prediction(seed)
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        xbins = dict(
            start=min([df_plot["model_output"].min(), df_plot[self.response_name].min()]),
            end=max([df_plot["model_output"].max(), df_plot[self.response_name].max()]),
            size=bin_size,
        )
        fig.add_trace(go.Histogram(x=df_plot[self.response_name], name="original output", xbins=xbins), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_plot["model_output"], name="model output", xbins=xbins), row=1, col=1)

        fig.update_traces(opacity=0.5)
        fig.update_layout(barmode="overlay")
        fig.update_layout(
            title_text=f"Result {self.response_name} model vs. original output",  # title of plot
        )

        fig.show()

    def print_model_summary(self, bin_size: float = 0.05, seed: int = 123) -> None:
        """Prints a summary of the model and plots data sampled from the model vs the training data.

        Parameters
        ----------
        bin_size : float, optional
            Size of the bins.
        seed : int, optional
            Seed for the model prediction.
        """
        self._display_model_info()
        self.plot_model_vs_original_histogram(bin_size=bin_size, seed=seed)


class ConvenientOlsFit(ConvenientRegressionFit):
    """A class for ordinary least squares (OLS) regression models.

    Parameters
    ----------
    model_name : str
        Name of the model.
    scenario_name : str
        Name of the scenario.
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Regression formula.
    timestamp_input_data : str or None
        Timestamp of the input data.
    timestamp_export : str or None
        Timestamp of the model export.
    total_failure_info : TotalFailureInfo or None
        Total failure info received from remove_invalid_sessions.
    inverse_output_trafo : Callable or None, optional
        Inverse output transformation of the SPV.

    Attributes
    ----------
    noise_dist_family : st.rv_continuous
        Normal distribution.
    """

    def __init__(
        self,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        formula: str,
        timestamp_input_data: Optional[str] = None,
        timestamp_export: Optional[str] = None,
        total_failure_info: Optional[TotalFailureInfo] = None,
        inverse_output_trafo: Optional[Callable] = None,
    ) -> None:
        self.noise_dist_family = st.norm
        super().__init__(
            model_name=model_name,
            scenario_name=scenario_name,
            df=df,
            formula=formula,
            timestamp_input_data=timestamp_input_data,
            timestamp_export=timestamp_export,
            total_failure_info=total_failure_info,
            inverse_output_trafo=inverse_output_trafo,
        )

    @property
    def noise_dist_name(self) -> str:
        return self.noise_dist_family.name

    def _fit_noise_dist(self) -> None:
        if self.noise_dist_name == "norm":
            self.noise_dist = st.norm(0.0, np.sqrt(self.result.mse_resid))
        else:
            pars = self.noise_dist_family.fit(self.df["resid"])
            self.noise_dist = self.noise_dist_family(*pars)

            try:
                test_quantiles = np.linspace(1e-10, 1 - 1e-10, 100_000)
                resids = self.noise_dist.ppf(q=test_quantiles)
                if not (np.any(np.isinf(resids)) or np.any(np.isnan(resids))):
                    print(f"1e-10 percentile of the fitted residual distribution: {min(resids)}")
                    print(f"1-1e-10 percentile of the fitted residual distribution: {max(resids)}")
                elif np.any(np.isinf(resids)):
                    warn_msg = "Inf values occurred for the quantiles:"
                    for q in test_quantiles[np.isinf(resids)]:
                        warn_msg += f"\n{q}"
                    warnings.warn(warn_msg, Warning)
                elif np.any(np.isnan(resids)):
                    warn_msg = "NaN values occurred for the quantiles:"
                    for q in test_quantiles[np.isnan(resids)]:
                        warn_msg += f"\n{q}"
                    warnings.warn(warn_msg, Warning)
            except RuntimeError as err:
                warnings.warn(f"Some example percentile threw the following error:\n{err}", Warning)

    def _return_fit(self, formula: str, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, RegressionResultsWrapper]:
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        return y, X, sm.OLS(y, X).fit()

    def nongaussian_resid_fit(self, dist: st.rv_continuous) -> None:
        """Fits custom distribution on the residuals of the model.

        Parameters
        ----------
        dist : st.rv_continuous
            Custom distribution family to fit.
        """
        self.noise_dist_family = dist
        self._fit_noise_dist()


class ConvenientGlmFit(ConvenientRegressionFit):
    """A class to fit generalized linear model (GLM) regressions.

    Parameters
    ----------
    model_name : str
        Name of the model.
    scenario_name : str
        Name of the scenario.
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Regression formula.
    family : sm.genmod.families.family.Family
        Family of the noise distribution.
    total_failure_info : TotalFailureInfo or None
        Total failure info received from remove_invalid_sessions.
    timestamp_input_data : str or None
        Timestamp of the input data.
    timestamp_export : str or None
        Timestamp of the model export.

    Attributes
    ----------
    noise_dist_family : sm.genmod.families.family.Family
        Family of the noise distribution.

    Notes
    -----
    Do not use this class yet as it is in an experimental stage.
    """

    def __init__(
        self,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        formula: str,
        family: sm.genmod.families.family.Family,
        total_failure_info: Optional[TotalFailureInfo] = None,
        timestamp_input_data: Optional[str] = None,
        timestamp_export: Optional[str] = None,
    ) -> None:
        self.noise_dist_family = family
        super().__init__(
            model_name=model_name,
            scenario_name=scenario_name,
            df=df,
            formula=formula,
            total_failure_info=total_failure_info,
            timestamp_input_data=timestamp_input_data,
            timestamp_export=timestamp_export,
        )

    @property
    def noise_dist_name(self) -> str:
        return self.noise_dist.dist.name

    def _fit_noise_dist(self) -> None:
        # WARNING! This is not the noise dist, refactor!
        pars = st.chi2.fit(self.result.resid_deviance, f0=self.result.df_resid)
        self.noise_dist = st.chi2(*pars)

    def _return_fit(self, formula: str, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, RegressionResultsWrapper]:
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        result = sm.GLM(y, X, family=self.noise_dist_family).fit()
        result.resid = result._results.resid = result.resid_deviance  # todo refactor
        return y, X, result


class ConvenientMleFit(ConvenientRegressionFit):
    """A class to fit a regression using the maximum likelihood estimator.

    Parameters
    ----------
    model_name : str
        Name of the model.
    scenario_name : str
        Name of the scenario.
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Regression formula.
    noise_dist : st.rv_continuous
        Family of the noise distribution.
    total_failure_info : TotalFailureInfo or None
        Total failure info received from remove_invalid_sessions.
    timestamp_input_data : str or None
        Timestamp of the input data.
    timestamp_export : str or None
        Timestamp of the model export.

    Attributes
    ----------
    noise_dist_family : st.rv_continuous
        Family of the noise distribution.

    Notes
    -----
    Do not use this class yet as it is in an experimental stage.
    """

    def __init__(
        self,
        model_name: str,
        scenario_name: str,
        df: pd.DataFrame,
        formula: str,
        noise_dist: st.rv_continuous,
        total_failure_info: Optional[TotalFailureInfo] = None,
        timestamp_input_data: Optional[str] = None,
        timestamp_export: Optional[str] = None,
    ) -> None:
        self.noise_dist_family = noise_dist
        super().__init__(
            model_name=model_name,
            scenario_name=scenario_name,
            df=df,
            formula=formula,
            total_failure_info=total_failure_info,
            timestamp_input_data=timestamp_input_data,
            timestamp_export=timestamp_export,
        )

    @property
    def noise_dist_name(self) -> str:
        return "todo"

    def _return_fit(self, formula: str, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, RegressionResultsWrapper]:
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        return y, X, GenericLinearLikelihoodModel(endog=y, exog=X, noise_dist=self.noise_dist_family).fit()

    def _fit_noise_dist(self) -> None:
        arg = self.result.noise_params[:-1]
        scale = self.result.noise_params[-1]
        self.noise_dist = self.result.model.noise_zero_mean(arg=arg, scale=scale)


class GenericLinearLikelihoodModel(GenericLikelihoodModel):
    """
    Notes
    -----
    Do not use this class yet as it is in an experimental stage.
    """

    def __init__(self, endog: pd.DataFrame, exog: pd.DataFrame, noise_dist: st.rv_continuous, **kwds) -> None:
        self.size_beta = exog.shape[1]  # todo: rename
        self.noise_dist_type = noise_dist
        super().__init__(endog=endog, exog=exog, **kwds)

    def nloglikeobs(self, params: np.ndarray) -> np.ndarray:
        y = self.endog
        X = self.exog
        y_hat = self.predict(params=params, exog=X)
        dist_par = params[self.size_beta :]

        arg = dist_par[:-1]
        scale = dist_par[-1]
        ll = self.noise_zero_mean(arg=arg, scale=scale).logpdf(y - y_hat)
        return -ll

    def noise_zero_mean(self, arg: Iterable[float], scale: float) -> st._distn_infrastructure.rv_continuous_frozen:
        """Choses location, so that mean=0"""
        mean = self.noise_dist_type(*arg, 0, scale).mean()
        return self.noise_dist_type(*arg, -mean, scale)

    def predict(self, params: np.ndarray, exog: np.ndarray, **pred_kwds_linear) -> np.ndarray:
        beta = params[: self.size_beta]
        X = self.exog
        return np.dot(X, beta)

    def fit(
        self, start_params: Optional[np.ndarray] = None, maxiter: int = 10000, maxfun: int = 5000, **kwds
    ) -> RegressionResultsWrapper:
        if start_params == None:
            # compute approximate least squares fit
            ols_result = sm.OLS(self.endog, self.exog).fit()
            params_ols_fit = ols_result.params

            # compute fit to residuals
            params = self.noise_dist_type.fit(ols_result.resid)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # use both as start guess
            params_dist = (*arg, scale)
            # The noise dist below will have set the location to zero, since it is considered in the intercept.
            params_ols_fit[0] += loc
            start_params = np.append(params_ols_fit, params_dist)  # add distribution's initial guess

        # add for summary
        # self.exog_names.append('shape')
        # self.exog_names.append('scale')
        result = super().fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

        # Build OLS-style result
        # Remark: This should all move to ConvenientMleFit to be aligned with GLM
        result_ols_style = types.SimpleNamespace()  # Initialize an empty object, todo: namedtuple
        prediction = result.get_prediction(self.exog).predicted
        result_ols_style.fittedvalues = prediction
        result_ols_style.predict = result.predict
        y = self.endog
        result_ols_style.resid = y - prediction
        result_ols_style.mse_resid = mean_squared_error(y, prediction)
        result_ols_style.aic = result.aic

        # Isolate noise parameters
        result_ols_style.params = pd.Series(result.params[: self.size_beta], index=self.exog_names[: self.size_beta])
        result_ols_style.pvalues = pd.Series(result.pvalues[: self.size_beta], index=self.exog_names[: self.size_beta])
        result_ols_style.noise_params = pd.Series(
            result.params[self.size_beta :], index=self.exog_names[self.size_beta :]
        )
        result_ols_style.noise_pvalues = pd.Series(
            result.pvalues[self.size_beta :], index=self.exog_names[self.size_beta :]
        )

        # result_ols_style.complete_noise_params = result.model.get_noise_dist()

        def conf_int_data_frame(alpha=0.05):
            # return panda data frame instead of matrix
            df = pd.DataFrame(result.conf_int(alpha))
            df.index = self.exog_names
            return df

        result_ols_style.conf_int = conf_int_data_frame
        result_ols_style.model = result.model
        result_ols_style.summary = result.summary
        return result_ols_style


def _get_linearly_dependent_predictors(
    df: pd.DataFrame, tolerance_linear_dependency: Optional[float] = 1e-12
) -> List[str]:
    null_space_matrix = null_space(np.array(df))
    linear_combinations = []
    for col in null_space_matrix.T:
        combination = ""
        for idx in np.where(np.abs(col) > tolerance_linear_dependency)[0]:
            combination += f" {round(col[idx], 3)}*{df.columns[idx]} +"
        linear_combinations.append(combination.lstrip(" ").rstrip(" +").replace("+ -", "- ") + " = 0")
    return linear_combinations


def backward_stepwise_selection(base_model: ConvenientRegressionFit) -> ConvenientRegressionFit:
    """Algorithm for removing statistically insignificant predictors.

    Parameters
    ----------
    base_model : ConvenientRegressionFit
        Model to remove predictors from.

    Returns
    -------
    ConvenientRegressionFit
        Model without statistically insignificant predictors.

    See Also
    --------
    get_polynomial_of_parameters_and_order : Generate a polynomial of a specific order for the parameters.

    Notes
    -----
    For further information see https://cc-github.bmwgroup.net/moritzwerling/reliability-engineering/tree/main/Model_Fits/tools/regression_models#backward-stepwise-selection.
    """
    number_of_predictors = len(base_model.predictors)

    if base_model.y.size <= number_of_predictors:
        raise ValueError(
            "Backward stepwise selection: The number of samples has to be greater than the number of predictors."
        )
    linear_combinations = _get_linearly_dependent_predictors(base_model.X.drop(["Intercept"], axis=1))
    if linear_combinations:
        raise ValueError(
            f"Some of the predictors are linearly dependent, which is not allowed in OLS regression. See:\n {LINES_SEPARATOR.join(linear_combinations)}"
        )

    final_models: Dict[int, ConvenientRegressionFit] = {number_of_predictors: base_model}

    loocv_index = "LOOCV mean error"
    df_log = _create_log_dataframe(base_model, loocv_index)
    df_log.loc[number_of_predictors, base_model.predictors] = ["included" for _ in base_model.predictors]
    df_log.loc[number_of_predictors, loocv_index] = base_model.loocv

    for k in reversed(range(1, number_of_predictors + 1)):
        final_models[k - 1] = _get_best_model(final_models[k])

        df_log.loc[k - 1, final_models[k - 1].predictors] = ["included" for _ in final_models[k - 1].predictors]
        df_log.loc[k - 1, loocv_index] = final_models[k - 1].loocv

    def _get_color_for_bool(val: str) -> str:
        color = "green" if val == "included" else "red"
        return "background-color: %s" % color

    final_model_idx = (
        df_log[loocv_index].iloc[::-1].idxmin()
    )  # use reversed dataframe to get model with least predictors in case of a tie
    df_log[loocv_index] = df_log[loocv_index].map("{:.6e}".format)
    display(
        df_log.transpose()
        .style.applymap(_get_color_for_bool, subset=(base_model.predictors, range(0, number_of_predictors + 1)))
        .applymap(
            lambda x: "background-color: yellow; color: black", subset=pd.IndexSlice[[loocv_index], [final_model_idx]]
        )
    )
    print(f"Backward stepwise selection has chosen the model for k = {final_model_idx}.")

    return final_models[final_model_idx]


def _create_log_dataframe(base_model: ConvenientRegressionFit, loocv_index: str) -> pd.DataFrame:
    number_of_predictors = len(base_model.predictors)
    log_data = {predictor: ["removed"] * (number_of_predictors + 1) for predictor in base_model.predictors}
    log_data[loocv_index] = [0 for _ in range(0, number_of_predictors + 1)]
    return pd.DataFrame(
        log_data, index=pd.Index(reversed(range(0, number_of_predictors + 1)), name="k"), columns=base_model.predictors
    )


def _get_best_model(last_model: ConvenientRegressionFit) -> ConvenientRegressionFit:
    models: List[ConvenientRegressionFit] = []
    for predictor in last_model.predictors:
        is_predictor_hierarchically_dependent = ConvenientRegressionFit.is_predictor_hierarchically_dependent(
            predictor=predictor, predictors=last_model.predictors
        )
        if not is_predictor_hierarchically_dependent:
            model = copy.deepcopy(last_model)
            model.drop_term(predictor)
            models.append(model)
    return max(models, key=lambda model: model.result.rsquared)


def full_run(
    df: pd.DataFrame,
    formula: str,
    model_name: str,
    scenario_name: str,
    noise_dist: Optional[st.rv_continuous] = None,
    total_failure_info: Optional[TotalFailureInfo] = None,
    timestamp_input_data: Optional[str] = None,
    timestamp_export: str = get_now_string(),
    drop_terms: List[str] = [],
    run_find_best_fit: bool = False,
    number_of_printed_outliers: int = 10,
    print_outlier_session_ids: bool = False,
) -> ConvenientOlsFit:
    """Creates and fits an OLS model and prints and plots information regarding the model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of training data.
    formula : str
        Regression formula.
    model_name : str
        Name of the model.
    scenario_name : str
        Name of the scenario.
    noise_dist : st.rv_continuous or None, optional
        Distribution family to fit the noise.
    total_failure_info : TotalFailureInfo or None, optional
        Total failure info received from remove_invalid_sessions.
    timestamp_input_data : str or None, optional
        Timestamp of the input data.
    timestamp_export : str or None, optional
        Timestamp of the model export.
    drop_terms : List[str], optional
        Terms to drop from the model.
    run_find_best_fit : bool, optional
        Wether to find the best fit for the residuals.
    number_of_printed_outliers : int, optional
        Number of outliers to be printed.
    print_outlier_session_ids : bool, optional
        Wether to print the outlier session ids.

    Returns
    -------
    ConvenientOlsFit
        The fitted model.
    """
    fit = ConvenientOlsFit(
        scenario_name=scenario_name,
        model_name=model_name,
        df=df,
        formula=formula,
        total_failure_info=total_failure_info,
        timestamp_input_data=timestamp_input_data,
        timestamp_export=timestamp_export,
    )
    font_size = 18
    for term in drop_terms:
        fit.drop_term(term)
    display_markdown(f"Dropped terms: {drop_terms}", font_size)
    display_markdown("Performed backward stepwise selection:", font_size)
    fit = backward_stepwise_selection(fit)
    display_markdown("Regression summary:", font_size)
    fit.print_summary()
    display_markdown("Residuals:", font_size)
    fit.plot_residuals()
    if run_find_best_fit:
        display_markdown("Best fit for residuals:", font_size)
        display(find_best_fit(fit.df["resid"]))
    if noise_dist:
        display_markdown(f"Performed nongaussian resid fit on distribution: {noise_dist.name}", font_size)
        fit.nongaussian_resid_fit(noise_dist)
    display_markdown("QQ-Plots:", font_size)
    fit.resid_qqplot()
    display_markdown("Partial regression plots:", font_size)
    fit.plot_partregress_grid()
    display_markdown("Component-component plus residual plots:", font_size)
    fit.plot_ccpr_grid()
    if print_outlier_session_ids:
        display_markdown("Outlier session ids:", font_size)
        fit.print_outlier_session_ids(number_of_printed_outliers)
    fit.print_model_summary()
    return fit


def get_polynomial_of_parameters_and_order(parameters: List[str], order: int) -> str:
    """Generate a polynomial of a specific order for the parameters.

    Parameters
    ----------
    parameters : List[str]
        Parameters for the polynomial.
    order : int
        Order of the polynomial.

    Returns
    -------
    str
        The generated polynomial.

    """
    if order < 1:
        raise ValueError(f"The order of the polynomial as to be greater than 1. Given {order}")

    all_combinations = []
    for i in range(1, order + 1):
        all_combinations += list(combinations_with_replacement(parameters, i))
    factors = ["*".join(combination) for combination in all_combinations]
    return " + ".join(factors)
