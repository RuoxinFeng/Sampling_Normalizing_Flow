#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import json
import os
from typing import Any, List, Optional

import pandas as pd
from tools.remove_invalid_sessions.remove_invalid_sessions import SRS
from tools.sessions_to_manually_change.sessions_to_manually_change_schema import SessionsToChange


def _open_and_validate_file(srs: SRS) -> SessionsToChange:
    with open(os.path.join(os.getcwd(), "sessions_to_manually_change", srs, "manual_changes.json")) as f:
        return SessionsToChange(**json.load(f))


def _get_column_name_by_cb_id(df: pd.DataFrame, cb_id: str) -> Optional[str]:
    found_column = None
    for column_name in df.columns:
        if column_name.upper().endswith(cb_id.upper()):
            if found_column is not None:
                raise ValueError(f"Found multiple columns for CB id {cb_id}!")
            found_column = column_name
    return found_column


def _remove_duplicates(in_list: List[Any]) -> List[Any]:
    return list(set(in_list))


def manually_change_values(df: pd.DataFrame, srs: SRS) -> pd.DataFrame:
    """Changes the value of cells based on the `manual_changes.json` in the `sessions_to_manually_change/SRS` folder for the given SRS.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing IFVs and SPVs with their respective Codebeamer IDs.
    srs : SRS
        The SRS that is analyzed.

    Returns
    -------
    pd.DataFrame
        The provided dataframe with exchanged values per specified column.

    Raises
    ------
    ValueError
        If there are multiple columns with the same Codebeamer ID.

    Examples
    --------
    >>> # Let's look at an example for SRS04:
    >>> # We need the file manual_changes.json in <current working directory>/sessions_to_manually_change/SRS04. This file looks like this:
    {
        "1234567A": {
            "some_reason_CB_ID": {
                "abc1": 0.4594
            }
        },
        "76543210": {
            "some_other_reason_CB_ID": {
                "abc3": "new_value""
            },
            "another_reason_CB_ID": {
                "abc4": "test"
            }
        }
    }
    >>> df = pd.DataFrame({"A_1234567A": [1, 2, 3, 4], "B_7654321B": ["a", "b", "c", "d"], "session_id": ["abc1", "abc2", "abc3", "abc4"]})
       A_1234567A B_76543210 session_id
    0           1          a       abc1
    1           2          b       abc2
    2           3          c       abc3
    3           4          d       abc4
    >>> result_df = manually_change_values(df=df, srs=SRS.SRS04)
       A_1234567A B_76543210 session_id
    0      0.4594          a       abc1
    1           2          b       abc2
    2           3  new_value       abc3
    3           4       test       abc4
    """
    values_replaced = dict()
    values_not_found = []
    df_copy = df.copy()
    sessions_to_change = _open_and_validate_file(srs=srs)
    for cb_id, mapping in sessions_to_change.mappings_by_cb_id.items():
        column = _get_column_name_by_cb_id(df_copy, cb_id)
        if column is None:
            print(f"No matching column found for codebeamer id {cb_id}!")
        else:
            values_replaced[column] = dict()
            for session_id, value in mapping.items():
                if session_id in df_copy["session_id"].tolist():
                    df_copy.loc[df_copy["session_id"] == session_id, column] = value
                    values_replaced[column][session_id] = value
                else:
                    values_not_found.append(session_id)
    info_str = f"Successfully replaced the values for the following columns and session ids:\n{values_replaced}\n"
    if values_not_found:
        values_not_found = _remove_duplicates(values_not_found)
        info_str += f"The following session ids were not found in the Dataframe: {values_not_found}"
    print(info_str)
    return df_copy
