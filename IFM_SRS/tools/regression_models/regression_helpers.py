#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm, rv_continuous, uniform
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tools.general_utility.helpers import round_output


def predict_statsmodels(
    model: RegressionResultsWrapper,
    X: pd.DataFrame,
    U: np.ndarray,
    inv_trafo: Optional[Callable] = None,
    noise_distribution: Optional[rv_continuous] = None,
) -> pd.Series:
    """Predicts the statsmodels model for the given input X based on uniform samples. If no noise distribution is provided the residuals are calculated with a normal distribution.

    Parameters
    ----------
    model : RegressionResultsWrapper
        The model used for the prediction.
    X : pd.DataFrame
        The sample data.
    U : np.ndarray
        The uniformly distributed sample data for the residuals.
    inv_trafo : Callable or None, optional
        The inverse transformation of the prediction.
    noise_distribution : rv_continuous or None, optional
        The distribution used to calculate the residuals.

    Returns
    -------
    pd.Series
        The prediction.
    """
    fit = model.predict(X)
    if noise_distribution:
        noise = noise_distribution.ppf(U)
    else:
        noise = norm.ppf(U, loc=0.0, scale=np.sqrt(model.mse_resid))
    y = fit + noise
    if inv_trafo:
        return inv_trafo(y)
    else:
        return y


def generate_test_data(
    model: RegressionResultsWrapper,
    input_data: Dict[str, List[Any]],
    dmatrix: dmatrix,
    seed: int,
    inv_trafo: Optional[Callable] = None,
    noise_distribution: Optional[rv_continuous] = None,
    timestamp_input_data: Optional[str] = None,
    timestamp_export: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates test data for the model including X_input, U_noise, Y_predict and timestamps for the input data and export.

    Parameters
    ----------
    model : RegressionResultsWrapper
        The model that test data should be generated for.
    input_data : Dict[str, List[Any]]
        The input data.
    dmatrix : dmatrix
        The input data as a dmatrix.
    seed : int
        A seed for randomization.
    inv_trafo : Callable or None, optional
        The inverse transformation of the prediction.
    noise_distribution : rv_continuous or None, optional
        The distribution used to calculate the residuals.
    timestamp_input_data : str or None, optional
        The timestamp of the input data.
    timestamp_export : str or None, optional
        The timestamp of the export.

    Returns
    -------
    Dict[str, Any]
        The generated test data.

    Raises
    ------
    ValueError
        If there are less than two samples in the input data.
    """
    np.random.seed(seed)
    number_samples = len(next(iter(input_data.values())))
    if any([len(l) != number_samples for l in input_data.values()]):
        raise ValueError("Input data columns do not have the same length!")
    # Todo: The user still has to specify for categorical variables at least TWO levels, otherwise the design matrix will still degenerate. Fix needed.
    if number_samples < 2:
        raise ValueError(
            "Define at least two test vectors!"
        )  # This will cause problems for categorical variables otherwise
    input_noise = list(uniform.rvs(size=number_samples))
    test_data = {
        "X_input": input_data,
        "U_noise": input_noise,
        "Y_predict": round_output(
            list(
                predict_statsmodels(
                    model=model,
                    X=dmatrix,
                    U=np.array(input_noise),
                    inv_trafo=inv_trafo,
                    noise_distribution=noise_distribution,
                )
            )
        ),
        "timestamp_input_data": timestamp_input_data,
        "timestamp_export": timestamp_export,
    }
    return test_data
