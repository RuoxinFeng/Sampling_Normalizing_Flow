#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import os
from typing import List, Optional, Type

import numpy as np
import scipy.stats as st
from pydantic import BaseModel
from scipy.stats import rv_continuous, uniform
from tools.general_utility.helpers import NR_OF_DIGITS_AFTER_COMMA, round_output
from tools.models_export.residual_distribution_model import CompositeDistribution
from tools.situation_model.situation_model import MarginalDistributionTestCase, SituationModelParameter


def save_model_as_json_file(
    model: Type[BaseModel],
    file_path: str,
    exclude_unset: bool = False,
    nr_of_digits_after_comma: Optional[int] = NR_OF_DIGITS_AFTER_COMMA,
) -> None:
    """Exports a pydantic model to a json file.

    Parameters
    ----------
    model : Type[pydantic.BaseModel]
        The model to be exported.
    file_path : str
        The path where the json file should be saved.
    exclude_unset : bool, optional
        Whether fields of the model that were not explicitly set should also be exported.
    nr_of_digits_after_comma : int, optional
        How many digits after the comma of floats should be exported.
    """
    with open(file=file_path, mode="w", encoding="utf8") as f:
        json.dump(
            round_output(json.loads(model.json(exclude_unset=exclude_unset)), nr_of_digits_after_comma), f, indent=4
        )
        f.write("\n")


def _generate_u_noise(n_test_cases: int, seed: int) -> List[float]:
    np.random.seed(seed)
    return list(uniform.rvs(size=n_test_cases))


def export_composite_distribution_with_test_cases(
    base_file_name: str,
    distribution: rv_continuous,
    n_test_cases: int,
    seed: int,
    nr_of_digits_after_comma: Optional[int] = NR_OF_DIGITS_AFTER_COMMA,
) -> None:
    """Exports a composite distribution together with test cases (to unittest its integration in MCS) to json files.

    Parameters
    ----------
    base_file_name : List[str]
        A base for the names of the generated json files.
        The whole names will be <base_file_name>_composite_distribution.json and <base_file_name>_tests.json
    distribution : scipy.stats.rv_continuous
        The composite distribution to be exported.
    n_test_cases : int
        How many test cases should be exported.
    seed : int
        A seed used to sample the noise for the test cases.
    nr_of_digits_after_comma : int or None, optional
        How many digits after the comma of floats should be exported.
    """
    distribution_export = CompositeDistribution.from_scipy_distribution(distribution=distribution)
    save_model_as_json_file(
        model=distribution_export,
        file_path=os.path.join("models", base_file_name + "_composite_distribution.json"),
        nr_of_digits_after_comma=nr_of_digits_after_comma,
    )

    u_noise = _generate_u_noise(n_test_cases=n_test_cases, seed=seed)
    test_cases = MarginalDistributionTestCase(u_noise=u_noise, y_predict=list(distribution.ppf(u_noise)))
    save_model_as_json_file(
        model=test_cases,
        file_path=os.path.join("models", base_file_name + "_tests.json"),
        nr_of_digits_after_comma=nr_of_digits_after_comma,
    )


def get_situation_model_parameter_from_distribution(
    distribution: st.distributions.rv_frozen,
    name: str,
    codebeamer_reference: str,
    transformation_function: str,
    n_test_cases: int,
    seed: int,
) -> SituationModelParameter:
    """Creates a SituationModelParameter object from a distribution.

    Parameters
    ----------
    distribution : st.distributions.rv_frozen
        The distribution to create the SituationModelParameter from.
    name : str
        The name of the distribution.
    codebeamer_reference : str
        A reference to a codebeamer item of the fitted model.
    transformation_function : str or None, optional
        A function to transform the data sampled from the distribution.
    n_test_cases : int
        The number of test cases to be generated.
    seed : int
        The seed used to generate noise for the test cases.

    Returns
    -------
    SituationModelParameter
        The created SituationModelParameter object.
    """
    u_noise = _generate_u_noise(n_test_cases=n_test_cases, seed=seed)

    if hasattr(distribution, "pdf"):  # closed-form scipy distribution
        return SituationModelParameter(
            name=name,
            transformation_function=transformation_function,
            marginal_distribution=distribution.dist.name,
            marginal_distribution_parameters=distribution.args,
            marginal_distribution_test_cases=MarginalDistributionTestCase(
                u_noise=u_noise, y_predict=list(distribution.ppf(u_noise))
            ),
            codebeamer_reference=codebeamer_reference,
        )

    # Kernel density
    return SituationModelParameter(
        name=name,
        transformation_function=transformation_function,
        marginal_distribution="Kernel Density",
        marginal_distribution_parameters=distribution.icdf.tolist(),
        marginal_distribution_test_cases=MarginalDistributionTestCase(
            u_noise=u_noise,
            y_predict=np.interp(
                u_noise,
                np.linspace(0, 1, num=distribution.icdf.size),
                distribution.icdf,
            ).tolist(),
        ),
        codebeamer_reference=codebeamer_reference,
    )
