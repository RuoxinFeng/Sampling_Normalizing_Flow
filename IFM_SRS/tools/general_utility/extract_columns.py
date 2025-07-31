#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


from typing import List

import pandas as pd


def get_columns_with_IFVs(df: pd.DataFrame) -> List[str]:
    """Extracts all column names that include an IFV from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns should be extracted.

    Returns
    -------
    List[str]
        The list of column names that include an IFV.
    """
    pattern = r"[0-9]{7,}$"
    return _get_special_columns_by_regex(df=df, pattern=pattern)


def get_columns_with_SPVs(df: pd.DataFrame) -> List[str]:
    """Extracts all column names that include an SPV from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns should be extracted.

    Returns
    -------
    List[str]
        The list of column names that include an SPV.
    """
    pattern = r"_[0-9]{7,}[a-zA-Z]$"
    return _get_special_columns_by_regex(df=df, pattern=pattern)


def _get_special_columns_by_regex(df: pd.DataFrame, pattern: str) -> List[str]:
    return df.filter(regex=pattern).columns.tolist()
