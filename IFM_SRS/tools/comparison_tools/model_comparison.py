#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import pandas as pd
import plotly.figure_factory as ff
import scipy.stats as st
from tools.comparison_tools.divergence_metrics import emd, empirical_js_divergence, ks_test, mse
from tools.general_utility.custom_eval import evaluate_formula
from tools.general_utility.helpers import display_dataframe, display_markdown
from tools.models_export.statsmodels_exporter import SpmModel
from tools.plot_tools.plot_tools import plot_ks_test_statistic
from tools.regression_models.regression_helpers import predict_statsmodels
from tools.regression_models.regression_models import ConvenientRegressionFit


@dataclass
class ComparisonInformation:
    """A wrapper containing information about the compared models."""

    model_names: Tuple[str, str]
    """The names of the compared models."""
    data: Tuple[pd.Series, pd.Series]
    """The predictions of the compared models on the training data."""
    title: str
    """A title for the comparison."""


class ModelComparison:
    """Compare two iterations of a model to decide whether it needs to be updated in MCS.

    Parameters
    ----------
    new_model : ConvenientRegressionFit
        The new model.
    path_old_model : str
        The path to the json file describing the old model.
    inverse_output_trafo : Callable or None, optional
        The inverse of the transformation of the explained variable
        (i.e. if our model was ln(y) ~ x, the inverse_output_trafo would be math.exp)
    seed : int or None, optional
        A seed used to sample the noise when generating predictions of the models.
    name_new_model : str, optional
        A name for the new model. The tool will use this name to refer to the new model in plots.
    name_old_model : str, optional
        A name for the old model. The tool will use this name to refer to the old model in plots.

    Attributes
    ----------
    new_model : ConvenientRegressionFit
        The new model.
    old_model : SpmModel
        The old model.
    df : pd.DataFrame
        The training data of the new model.
    y_observed : pd.Series
        The ground-truth values of the explained variable in the training data of the new model.
    """

    def __init__(
        self,
        new_model: ConvenientRegressionFit,
        path_old_model: str,
        inverse_output_trafo: Optional[Callable] = None,
        seed: Optional[int] = 123,
        name_new_model: str = "New model",
        name_old_model: str = "Old model",
    ) -> None:
        self.new_model = new_model
        self.old_model = SpmModel.from_json(path_old_model)
        self.df = self.new_model.df
        self._inverse_output_trafo = inverse_output_trafo
        self._seed = seed
        self._uniform_noise = st.uniform.rvs(size=self.df.shape[0], random_state=self._seed)
        self._regression_formula = None
        self.y_observed = self.df[self.new_model.response_name]
        self._name_new_model = name_new_model
        self._name_old_model = name_old_model
        self._check_transformation_of_dependent_variable()

    @property
    def dependent_variable(self) -> str:
        """str : The name of the dependent variable in the old model."""
        return self.old_model.dependent_variable

    def _check_transformation_of_dependent_variable(self) -> None:
        matches = []
        supported_patterns = [
            re.compile("np\.\w+\((\w+)\s*,\s*\d+\)"),
            re.compile("np\.\w+\((\w+)\)"),
        ]
        for pattern in supported_patterns:
            matches.extend(re.findall(pattern, self.dependent_variable))

        if matches and not self._inverse_output_trafo:
            warnings.warn(
                "The dependent variable has been transformed and no inverse transformation has been given. The Regression output will correspond to the transformed variable!"
            )
        if not matches and self._inverse_output_trafo:
            warnings.warn(
                f"An inverse output transformation was given but no transformation of the dependent variable {self.dependent_variable} was detected. At the moment, only the transformations of the type np.(...) are recognized. In case the dependent variable was transformed by a different function and the correct inverse transformation has been given, this warning can be ignored."
            )

    def _find_ifv_in_model_param(self, model_param: str) -> str:
        for colname in self.df.columns:
            if colname in model_param:
                return colname
        raise ValueError(f"No matching column found in the data for old model parameter {model_param}")

    def _get_parameters_with_categories(self) -> Set[Tuple[str, str]]:
        categorical_params = []
        category_pattern = r"\[T\.(.*?)\]"
        for param in [k for k in self.old_model.params.keys() if k != "Intercept"]:
            for single_param in param.split(":"):
                ifv = self._find_ifv_in_model_param(single_param)
                match = re.search(category_pattern, single_param)
                if match:
                    categorical_params += [(ifv, match.group(1))]
                else:
                    categorical_params += [(ifv, None)]
        return set(categorical_params)

    def _get_formula_from_parameter_list(self) -> str:
        standard_replace_pattern = "df['{param}']"
        categorical_replace_pattern = "(df['{param}']=='{category}')"
        formula = " + ".join(
            [str(self.old_model.params["Intercept"])]
            + [
                f"{name.replace(':', ' * ')} * {value}"
                for name, value in self.old_model.params.items()
                if name != "Intercept"
            ]
        )
        for param in self._get_parameters_with_categories():
            if param[1] == None:
                formula = formula.replace(param[0], standard_replace_pattern.format(param=param[0]))
            else:
                formula = formula.replace(
                    f"{param[0]}[T.{param[1]}]", categorical_replace_pattern.format(param=param[0], category=param[1])
                )
        self._regression_formula = formula
        return formula

    @property
    def old_model_result(self) -> pd.Series:
        """pd.Series : The predictions of the old model on the training data of the new model."""
        result = evaluate_formula(
            self._get_formula_from_parameter_list(), subscriptable_dict={"df": self.df}
        ) + self.old_model.get_noise_distribution().ppf(self._uniform_noise)
        if self._inverse_output_trafo:
            return self._inverse_output_trafo(result)
        else:
            return result

    @property
    def new_model_result(self) -> pd.Series:
        """pd.Series : The predictions of the new model on its training data."""
        return predict_statsmodels(
            model=self.new_model.result,
            X=self.new_model.X,
            U=self._uniform_noise,
            inv_trafo=self.new_model.inverse_output_trafo,
            noise_distribution=self.new_model.noise_dist,
        )

    def compare_predictions(self, bin_size: float = 0.5) -> None:
        """Displays histogram plots of the predictions of both models on the training data of the new model.

        Parameters
        ----------
        bin_size : float, optional
            The desired width of the histogram bins.
        """
        display_markdown("Distplot: Compare predictions of the models", font_size=16)
        fig = ff.create_distplot(
            hist_data=[self.y_observed, self.new_model_result, self.old_model_result],
            group_labels=["Original output", self._name_new_model, self._name_old_model],
            bin_size=bin_size,
            show_hist=True,
            show_rug=False,
            curve_type="normal",
        )
        fig.update_layout(title_text="Predictions", xaxis_title=self.new_model.response_name, yaxis_title="Probability")
        fig.update_layout()
        fig.show()
        print(
            f"Used regression formula parsed from the input file (not including inverse output transformation if given): \n {self._regression_formula}"
        )

    def compare_params(self) -> None:
        """Displays a tabular comparison of the model parameters."""
        cm_keys = set(self.old_model.params.keys())
        bm_keys = set(self.new_model.result.params.keys())
        display_markdown("Comparison of model parameters", font_size=16)
        display_dataframe(
            pd.DataFrame(
                [
                    [
                        key,
                        self.new_model.result.params.get(key),
                        self.old_model.params.get(key),
                        abs(self.new_model.result.params[key] - self.old_model.params[key])
                        if key in bm_keys & cm_keys
                        else None,
                    ]
                    for key in bm_keys | cm_keys
                ],
                columns=["Parameter", self._name_new_model, self._name_old_model, "Absolute Difference"],
            ),
            style_properties={"text-align": "left", "index": False},
            table_styles=[{"selector": "th", "props": [("text-align", "left")]}],
        )

    def _create_comparison_information_list(self) -> List[ComparisonInformation]:
        model_information = {
            name: data
            for name, data in zip(
                [self._name_new_model, self._name_old_model, "Actual observations"],
                [self.new_model_result, self.old_model_result, self.y_observed],
            )
        }
        return [
            ComparisonInformation(
                model_names=(comparison[0], comparison[1]),
                data=(model_information[comparison[0]], model_information[comparison[1]]),
                title=f"{comparison[0]} vs {comparison[1]}",
            )
            for comparison in [
                [self._name_new_model, self._name_old_model],
                ["Actual observations", self._name_new_model],
                ["Actual observations", self._name_old_model],
            ]
        ]

    def model_similarity_quantities(self, nbins_for_distribution_discretization: int = 50) -> None:
        """Displays different model similarity metrics in a tabular fashion. Also plots the empirical cumulative
        distribution functions together with the Kolmogorov-Smirnov test statistic.

        Parameters
        ----------
        nbins_for_distribution_discretization : int, optional
            The number of bins to be used to calculate the Jensen-Shannon divergence.
        """
        display_markdown("Quantities indicating model similarity or lack of similarity", font_size=16)
        comparison_information_list = self._create_comparison_information_list()
        ks_test_results = [ks_test(*comparison.data) for comparison in comparison_information_list]
        display_dataframe(
            pd.DataFrame(
                data={
                    "MSE": [mse(*comparison.data) for comparison in comparison_information_list],
                    "EMD": [emd(*comparison.data) for comparison in comparison_information_list],
                    "JS-Divergence": [
                        empirical_js_divergence(*comparison.data, nbins=nbins_for_distribution_discretization)
                        for comparison in comparison_information_list
                    ],
                    "KS test statistic": [res.statistic for res in ks_test_results],
                },
                index=[comparison.title for comparison in comparison_information_list],
            ),
            display_index=True,
        )
        print(
            "Notes: \n 1) JS divergence is calculated from a discretized distribution \n 2) The KS test column shows the \033[1mtest statistic\033[0m of the Kolmogorov-Smirnov hypothesis test. The KS test statistic can be interpreted as the highest possible deviation between the two empirical cumulative distribution functions of the provided data, see figures below. As such, it can take values ranging from 0 (models are identical) to 1 (not similar at all)."
        )

        display_markdown("Empirical CDF plots with KS test statistic", font_size=16)

        for comparison, ks_result in zip(comparison_information_list, ks_test_results):
            plot_ks_test_statistic(
                data=comparison.data,
                names=comparison.model_names,
                ks_result=ks_result,
                title=comparison.title,
                xlabel=self.dependent_variable,
            )
