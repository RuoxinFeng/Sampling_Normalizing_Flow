#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import math
from typing import Dict, List, Union

import pytest
from tools.general_utility.helpers import OUTPUT_TOLERANCE


def assert_rounded(
    param: Union[Dict[str, float], List[float], float],
    tolerance: float = OUTPUT_TOLERANCE,
) -> None:
    if isinstance(param, list):
        for value in param:
            assert_rounded(value)
    elif isinstance(param, dict):
        for value in param.values():
            assert_rounded(param)
    elif not math.isinf(param):
        assert (param % tolerance == pytest.approx(0)) or (
            param % tolerance == pytest.approx(tolerance)
        ), f"{param} is not rounded to {-int(math.log10(tolerance))} digits"
