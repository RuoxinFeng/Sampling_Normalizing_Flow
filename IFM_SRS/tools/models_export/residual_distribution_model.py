#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import math
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Extra, confloat, conlist, constr, root_validator, validator
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from tools.regression_models.advanced_residual_distributions import piecewise_distribution, truncated_distribution


def _get_distribution_from_name(distribution_name: str) -> stats.rv_continuous:
    return getattr(stats, distribution_name)


class NoiseDistributionType(str, Enum):
    STANDARD = "standard"
    TRUNCATED = "truncated"
    PIECEWISE = "piecewise"


class SingleDistribution(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    parameters: conlist(float, min_items=1)
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    probability_mass: confloat(gt=0, le=1) = 1

    class Config:
        extra = Extra.forbid

    @root_validator
    def validate_truncation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        lower_bound = values.get("lower_bound")
        upper_bound = values.get("upper_bound")
        if lower_bound >= upper_bound:
            raise ValueError(f"Lower bound {lower_bound} needs to be smaller than upper bound {upper_bound}")
        return values

    @root_validator
    def validate_truncated_if_probability_mass_given(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        lower_bound = values.get("lower_bound")
        upper_bound = values.get("upper_bound")
        probability_mass = values.get("probability_mass")
        if probability_mass < 1.0 and lower_bound == -math.inf and upper_bound == math.inf:
            raise ValueError("Distribution needs to be truncated if probability mass is not equal to 1.")
        return values

    @validator("name")
    def validate_base_distribution_known(cls, name: str) -> str:
        try:
            _get_distribution_from_name(name)
        except AttributeError as exc:
            raise ValueError(f"Distribution {name} is not defined in scipy package.") from exc
        return name

    @property
    def is_truncated(self) -> bool:
        return self.lower_bound > -math.inf or self.upper_bound < math.inf

    @property
    def base_scipy_distribution(self) -> stats.rv_continuous:
        return _get_distribution_from_name(self.name)

    @property
    def frozen(self) -> rv_continuous_frozen:
        if self.is_truncated:
            return truncated_distribution(self.base_scipy_distribution, self.lower_bound, self.upper_bound)(
                *self.parameters
            )
        return self.base_scipy_distribution(*self.parameters)


class CompositeDistribution(BaseModel):
    distributions: conlist(SingleDistribution, min_items=1, max_items=2)

    class Config:
        extra = Extra.forbid

    @validator("distributions")
    def validate_probability_mass(cls, distributions: List[SingleDistribution]) -> List[SingleDistribution]:
        mass = sum([d.probability_mass for d in distributions])
        if not math.isclose(mass, 1.0):
            raise ValueError("Probability masses don't sum up to 1.")
        return distributions

    @validator("distributions")
    def validate_bounds_matching(cls, distributions: List[SingleDistribution]) -> List[SingleDistribution]:
        lower_bounds = [d.lower_bound for d in distributions]
        upper_bounds = [d.upper_bound for d in distributions]

        if len(distributions) > 1 and not all(
            math.isclose(ub, lb) for ub, lb in zip(upper_bounds[:-1], lower_bounds[1:])
        ):
            raise ValueError(
                "The upper bound of the distributions need to coincide with the lower bound of the subsequent distribution."
            )
        return distributions

    @validator("distributions")
    def validate_support_real_numbers_if_piecewise(
        cls, distributions: List[SingleDistribution]
    ) -> List[SingleDistribution]:
        if len(distributions) > 1:
            if distributions[0].lower_bound > -math.inf or distributions[-1].upper_bound < math.inf:
                raise ValueError("The support of a piecewise distribution needs to be all real numbers.")
        return distributions

    @property
    def type(self) -> NoiseDistributionType:
        if len(self.distributions) > 1:
            return NoiseDistributionType.PIECEWISE
        if self.distributions[0].is_truncated:
            return NoiseDistributionType.TRUNCATED
        return NoiseDistributionType.STANDARD

    @property
    def thresholds(self) -> List[float]:
        if self.type == NoiseDistributionType.PIECEWISE:
            return [d.lower_bound for d in self.distributions[1:]]
        raise ValueError(f"{self.type} distribution does not have thresholds.")

    @property
    def parameters(self) -> List[float]:
        parameters = [p for d in self.distributions for p in d.parameters]
        if self.type == NoiseDistributionType.PIECEWISE:
            return parameters + [d.probability_mass for d in self.distributions[:-1]]
        return parameters

    @property
    def frozen(self) -> rv_continuous_frozen:
        if self.type == NoiseDistributionType.PIECEWISE:
            return piecewise_distribution(
                *[d.base_scipy_distribution for d in self.distributions],
                *self.thresholds,
            )(*self.parameters)
        return self.distributions[0].frozen

    @classmethod
    def from_scipy_distribution(cls, distribution: rv_continuous_frozen) -> "CompositeDistribution":
        if isinstance(distribution.dist, piecewise_distribution):
            return cls(distributions=cls._get_piecewise_distribution(distribution))
        if isinstance(distribution.dist, truncated_distribution):
            bounds_args: Dict[str, float] = {}
            if distribution.dist.lower_bound > -np.inf:
                bounds_args["lower_bound"] = distribution.dist.lower_bound
            if distribution.dist.upper_bound < np.inf:
                bounds_args["upper_bound"] = distribution.dist.upper_bound
            return cls(
                distributions=[
                    SingleDistribution(name=distribution.dist.dist.name, parameters=distribution.args, **bounds_args)
                ]
            )
        return cls(distributions=[SingleDistribution(name=str(distribution.dist.name), parameters=distribution.args)])

    @classmethod
    def _get_piecewise_distribution(cls, distribution: rv_continuous_frozen) -> List[SingleDistribution]:
        args = distribution.args
        d: piecewise_distribution = distribution.dist
        nr_args_lower_dist = d.lower_dist.numargs + 2
        nr_args_upper_dist = d.upper_dist.numargs + 2
        lower_dist = SingleDistribution(
            name=d.lower_dist.dist.name,
            parameters=args[:nr_args_lower_dist],
            upper_bound=d.threshold,
            probability_mass=args[-1],
        )
        upper_dist = SingleDistribution(
            name=d.upper_dist.dist.name,
            parameters=args[nr_args_lower_dist : nr_args_lower_dist + nr_args_upper_dist],
            lower_bound=d.threshold,
            probability_mass=1 - args[-1],
        )
        return [lower_dist, upper_dist]
