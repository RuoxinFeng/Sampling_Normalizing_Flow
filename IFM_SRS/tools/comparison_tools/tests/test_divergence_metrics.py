#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import pandas as pd
import pytest
from tools.comparison_tools.divergence_metrics import emd, empirical_js_divergence, ks_test, mse


@pytest.fixture
def data_1() -> pd.Series:
    return pd.Series([1, 2, 3, 4])


@pytest.fixture
def data_2() -> pd.Series:
    return pd.Series([2, 2, 4, 5, 5])


@pytest.fixture
def data_3() -> pd.Series:
    return pd.Series([5, 4, 4, 3])


def test_empirical_js_divergence(data_1: pd.Series, data_2: pd.Series):
    assert empirical_js_divergence(data_1, data_2, nbins=5) == pytest.approx(0.5674859751930244)


def test_ks_test(data_1: pd.Series, data_2: pd.Series):
    res = ks_test(data_1, data_2)

    assert res.statistic == pytest.approx(
        max(abs(1 / 4 - 0 / 5), abs(2 / 4 - 2 / 5), abs(3 / 4 - 2 / 5), abs(4 / 4 - 3 / 5), abs(4 / 4 - 5 / 5))
    )
    assert res.pvalue == pytest.approx(0.7460317460317459)


def test_mse(data_1: pd.Series, data_3: pd.Series):
    assert mse(data_1, data_3) == pytest.approx((4**2 + 2**2 + 1**2 + 1**2) / 4)


def test_emd(data_1: pd.Series, data_2: pd.Series):
    assert emd(data_1, data_2) == (
        (1 / 4 - 0 / 5) + (2 / 4 - 2 / 5) + (3 / 4 - 2 / 5) + (4 / 4 - 3 / 5) + (4 / 4 - 5 / 5)
    )
