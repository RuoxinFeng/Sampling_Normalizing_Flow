#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import Dict, List, Optional

import pandas as pd
from IPython.display import display

CB_URL = "https://codebeamer.bmwgroup.net/cb/item/"


def extract_and_rename_columns(df: pd.DataFrame, ids: Dict[str, str], helper_data_list: List[str] = []) -> pd.DataFrame:
    """Extracts columns of IFVs and SPVs from a DataFrame and prints a table of all codebeamer pages of these variables.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose column names should be formatted.
    ids : Dict[str, str]
        A dictionary containing the codebeamer ids of all IFVs and SPVs as keys and their desired names as values.
        The columns will be renamed according to these values, but non-alphanumeric characters will be replaced by an underscore.
    helper_data_list : List[str], optional
        List of additional column names that should be included in the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with renamed columns.

    Raises
    ------
    ValueError
        If multiple column names in the DataFrame include the same CB ID.
    ValueError
        If a key in the ids dictionary has multiple sequences of digits (e.g. 1234A1234).
    """
    matching_col_names: List[str] = []
    new_col_names: List[str] = []
    cb_links: List[str] = []

    for key in ids.keys():
        matching_col = _get_matching_col(df, key)
        if matching_col:
            cb_id = _extract_cb_id(key)
            cb_links.append(_get_cb_link(cb_id))
            matching_col_names.append(matching_col)
            new_col_names.append(_get_new_col_name(ids, key))

    df_new = _create_new_df(
        df_old=df, matching_col_names=matching_col_names, new_col_names=new_col_names, helper_data_list=helper_data_list
    )
    _display_info(matching_col_names=matching_col_names, new_col_names=new_col_names, cb_links=cb_links)
    return df_new


def _get_matching_col(df: pd.DataFrame, key: str) -> Optional[str]:
    matching_cols = [col for col in df.columns if col.startswith(key) or col.endswith(key)]
    if len(matching_cols) > 1:
        raise ValueError(
            f"Multiple columns may not have the same CB ID! Please review the following columns: {matching_cols}"
        )
    return matching_cols[0] if matching_cols else None


def _extract_cb_id(key: str) -> str:
    cb_id_candidates = re.findall(r"\d+\.?\d*", key)
    if len(cb_id_candidates) > 1:
        raise ValueError(f"Id {key} has multiple sequences of digits: {cb_id_candidates}")
    if len(cb_id_candidates) == 0:
        return key  # SRS-08 has keys like XXXXW
    return cb_id_candidates[0]


def _get_cb_link(cb_id: str) -> str:
    return CB_URL + cb_id


def _get_new_col_name(ids: Dict[str, str], key: str) -> str:
    new_name = ids[key].lower()
    new_name = re.sub(r"[^a-zA-Z0-9]+", "_", new_name)  # Remove non-characters and non-numbers
    return new_name + "_" + key


def _display_info(matching_col_names: List[str], new_col_names: List[str], cb_links: List[str]) -> None:
    display_df = _get_df_to_display(
        matching_col_names=matching_col_names, new_col_names=new_col_names, cb_links=cb_links
    )
    pd.options.display.max_rows = pd.options.display.max_columns = pd.options.display.width = None
    display(display_df.style.format({"Codebeamer": lambda x: x}))


def _get_df_to_display(matching_col_names: List[str], new_col_names: List[str], cb_links: List[str]) -> pd.DataFrame:
    display_df = pd.DataFrame(
        {
            "Old": matching_col_names,
            "New": new_col_names,
            "Codebeamer": cb_links,
        }
    )
    display_df["Codebeamer"] = display_df["Codebeamer"].apply(lambda x: _make_clickable(x))
    return display_df


def _make_clickable(url: str) -> str:
    return f'<a href="{url}">{url}</a>'


def _create_new_df(
    df_old: pd.DataFrame, matching_col_names: List[str], new_col_names: List[str], helper_data_list: List[str]
) -> pd.DataFrame:
    df_new = df_old[matching_col_names].rename(columns=dict(zip(matching_col_names, new_col_names)))

    if helper_data_list:
        for entry in helper_data_list:
            df_new[entry.lower()] = df_old[entry]

    # Add session_id in first column if it exists in the original frame
    if "session_id" in df_old.columns:
        df_new.insert(0, "session_id", df_old["session_id"])

    return df_new
