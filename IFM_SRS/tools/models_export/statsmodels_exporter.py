#
# Copyright (C) 2022-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import warnings
from typing import Dict, Optional

import pandas as pd
from pydantic import Extra, confloat
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from statsmodels import __version__ as sm_version
from statsmodels.regression.linear_model import OLSResults
from tools.general_utility.helpers import NR_OF_DIGITS_AFTER_COMMA, round_output
from tools.models_export.residual_distribution_model import CompositeDistribution
from tools.situation_model.situation_model import ModelTimestamps


def get_actual_statsmodels_version() -> str:
    return sm_version


class SpmModel(ModelTimestamps):
    name: str
    model_class: str
    params: Dict[str, float]
    dependent_variable: str
    mse_residuals: confloat(strict=True, ge=0.0)
    custom_noise_distribution: CompositeDistribution
    total_failure_rate: Optional[float]

    class Config:
        extra = Extra.forbid

    @classmethod
    def from_linear_model_result(
        cls,
        model_result: OLSResults,
        name: str,
        total_failure_rate: Optional[float],
        timestamp_input_data: Optional[str],
        timestamp_export: Optional[str],
        custom_noise_distribution: rv_continuous_frozen,
    ) -> "SpmModel":
        cls.raise_if_statsmodels_version_incorrect()
        params: pd.Series = model_result.params

        model_dict = dict(
            timestamp_input_data=timestamp_input_data,
            timestamp_export=timestamp_export,
            name=name,
            model_class=str(model_result.model.__class__),
            params=params.to_dict(),
            dependent_variable=model_result.model.endog_names,
            mse_residuals=model_result.mse_resid,
            custom_noise_distribution=CompositeDistribution.from_scipy_distribution(custom_noise_distribution),
            total_failure_rate=total_failure_rate,
        )

        return cls(**model_dict)

    @classmethod
    def raise_if_statsmodels_version_incorrect(cls) -> None:
        compatible_versions = ["0.13.2", "0.14.0"]
        actual_version = get_actual_statsmodels_version()
        if actual_version not in compatible_versions:
            raise RuntimeError(
                f"The class {cls.__name__} was developed for statsmodels {compatible_versions}, not {actual_version}!"
            )

    def as_json(self, file_path: str, nr_of_digits_after_comma: Optional[int] = NR_OF_DIGITS_AFTER_COMMA) -> None:
        if self.timestamp_input_data is None or self.timestamp_export is None:
            warnings.warn(
                f"Either input data timestamp or export timestamp was not provided while exporting the model",
                UserWarning,
            )
        with open(file_path, mode="w", encoding="utf8") as fp:
            fp.write(self.as_json_str(nr_of_digits_after_comma))
            fp.write("\n")

    def as_json_str(self, nr_of_digits_after_comma: Optional[int] = NR_OF_DIGITS_AFTER_COMMA) -> str:
        return json.dumps(round_output(json.loads(self.json(exclude_unset=True)), nr_of_digits_after_comma), indent=4)

    def get_noise_distribution(self) -> rv_continuous_frozen:
        return self.custom_noise_distribution.frozen

    @classmethod
    def from_json(cls, file_path: str) -> "SpmModel":
        return cls.parse_file(file_path)
