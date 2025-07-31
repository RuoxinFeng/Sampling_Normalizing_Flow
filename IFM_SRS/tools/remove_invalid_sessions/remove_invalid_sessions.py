#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


# ------------------------------------------------------------------------------------------------------------------------------------------------
# With this module, it should be possible to remove invalid sessions.
# To do this, a SRS and the sensor names of interest have to be provided.
# Each srs has its own jsons of invalid sessions from the individual sensors.
# Optionally, you can specify the reasons_to_remove parameter to remove any of the session types "degraded_sessions",
# "invalid_sessions" or "total_failure_sessions" instead of removing all types (default).
# ------------------------------------------------------------------------------------------------------------------------------------------------

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import pandas as pd
from IPython.display import Markdown, display
from tools.remove_invalid_sessions.invalid_sessions_schema import InvalidSession


class ReasonsToRemove(str, Enum):
    DegradedSessions = "degraded_sessions"
    InvalidSessions = "invalid_sessions"
    TotalFailureSessions = "total_failure_sessions"


class SRS(str, Enum):
    SRS04 = "SRS04"
    SRS06 = "SRS06"
    SRS08 = "SRS08"
    SRS11_MV = "SRS11_MV"
    SRS11_ST = "SRS11_ST"
    SRS12 = "SRS12"
    SRS16 = "SRS16"


@dataclass
class InvalidSessionsSpec:
    """A wrapper containing information about files that specify which sessions need to be removed from the input data."""

    srs: SRS
    """The SRS that is analyzed."""
    sensor_names: List[str]
    """List of sensors whose invalid sessions should be removed."""
    reasons_to_remove: List[ReasonsToRemove] = field(
        default_factory=lambda: [
            ReasonsToRemove.DegradedSessions,
            ReasonsToRemove.InvalidSessions,
            ReasonsToRemove.TotalFailureSessions,
        ]
    )
    """List of reasons why certain sessions should be removed."""
    file_prefix: str = ""
    """File prefix of invalid sessions jsons."""


@dataclass
class TotalFailureInfo:
    """A wrapper containing information about total failures of a sensor in the input data."""

    posterior_total_failure_probability: float
    """The posterior probability of a total failure of the sensor with a uniform prior."""
    observed_total_failures: int
    """The number of total failures of the sensor observed in the input data."""


def _open_and_validate_file(
    srs: SRS, reason_to_remove: ReasonsToRemove, file_prefix: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    with open(
        os.path.join(os.getcwd(), "sessions_to_exclude", srs, f"{file_prefix}{reason_to_remove.value}.json")
    ) as f:
        return dict(InvalidSession(**json.load(f)))


def _check_for_duplicate_sessions(df: pd.DataFrame) -> None:
    duplicates = df["session_id"].duplicated()
    if any(duplicates):
        raise ValueError(
            f"Duplicates of the following session IDs exist in the dataframe: {df.loc[duplicates, 'session_id'].drop_duplicates().values}"
        )
    else:
        print("No duplicate sessions detected in the dataframe.")


def _check_for_duplicate_ids_in_total_failure_sessions(srs: SRS, file_prefix: str) -> None:
    invalid_sessions = _open_and_validate_file(
        srs=srs, reason_to_remove=ReasonsToRemove.TotalFailureSessions, file_prefix=file_prefix
    )
    ids = [
        set([id for category in sensor.values() for id in category["value"]]) for sensor in invalid_sessions.values()
    ]
    ids_count = Counter(id for set in ids for id in set)
    common_ids = [id for id, count in ids_count.items() if count > 1]
    if common_ids:
        ids_str = "<br>".join(sorted(common_ids))
        text = f"Correlated Total Failures detected. Please urgently contact Stanislav Braun, Moritz Werling or Felix Modes for deeper investigation. Affected sessions:<br>{ids_str}"
        display(Markdown(f"**<span style='background-color: red'>{text}</span>**"))
    else:
        print("No duplicate session ids detected in total failure sessions file.")


def remove_invalid_sessions(
    df: pd.DataFrame,
    spec: InvalidSessionsSpec,
) -> Tuple[pd.DataFrame, TotalFailureInfo]:
    """Removes invalid sessions (degraded, invalid, total failure) given by json files for the specified SRS and sensors and calculates the posterior total failure probability with a uniform prior and the number of observed total failures.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing IFVs and SPVs with their respective Codebeamer IDs.
    spec : InvalidSessionsSpec
        An object containing specifications about the files that define which sessions need to be removed.

    Returns
    -------
    pd.DataFrame
        The dataframe without the sessions that should be removed.
    TotalFailureInfo
        Information about the posterior total failure probability and the observed total failures.

    Raises
    ------
    ValueError
        If 'global' is part of the sensor_names list.
    ValueError
        If duplicates of session ids exist in the DataFrame.

    Notes
    -----
    This function requires a json file per SRS for every ReasonsToRemove in `reasons_to_remove`.
    """
    if not isinstance(spec.srs, SRS):
        raise ValueError(f"SRS {spec.srs} has to be a member of the SRS Enum.")
    if not all(isinstance(reason, ReasonsToRemove) for reason in spec.reasons_to_remove):
        raise ValueError(
            f"Reasons to remove {list(filter(lambda reason: not isinstance(reason,ReasonsToRemove),spec.reasons_to_remove))} have to be a member of the ReasonsToRemove Enum."
        )
    if not isinstance(spec.sensor_names, list):
        raise ValueError(f"sensor_names parameter {spec.sensor_names} is not a valid list")
    if "global" in spec.sensor_names:
        raise ValueError(
            "'global' invalid sessions get removed by default. Please only provide additional sensors whose invalid sessions should be removed."
        )
    _check_for_duplicate_sessions(df)
    _check_for_duplicate_ids_in_total_failure_sessions(spec.srs, file_prefix=spec.file_prefix)

    df_copy = df.copy()

    removed_sessions = {}
    should_removed = 0
    total_removed = 0

    for reason_to_remove in spec.reasons_to_remove:
        should_removed_per_reason = 0
        total_removed_per_reason = 0
        removed_sessions[reason_to_remove.value] = {}

        sessions_dict = _open_and_validate_file(
            srs=spec.srs, reason_to_remove=reason_to_remove, file_prefix=spec.file_prefix
        )

        for sensor in [*spec.sensor_names, "global"]:
            if sensor not in sessions_dict:
                if sensor == "global":
                    continue
                raise ValueError(f"No {reason_to_remove.value} for sensor {sensor} defined.")

            for category in list(sessions_dict[sensor].keys()):
                sessions_to_remove = sessions_dict[sensor][category]["value"]
                removed_sessions[reason_to_remove.value][category] = list(
                    df_copy.loc[df_copy["session_id"].isin(sessions_to_remove)]["session_id"]
                )
                df_copy = df_copy.loc[~df_copy["session_id"].isin(sessions_to_remove)]
                should_removed += len(sessions_to_remove)
                total_removed += len(removed_sessions[reason_to_remove.value][category])
                should_removed_per_reason += len(sessions_to_remove)
                total_removed_per_reason += len(removed_sessions[reason_to_remove.value][category])

        print(
            f"Removed {total_removed_per_reason}/{should_removed_per_reason} {reason_to_remove.value} successfully from df."
        )

    # Total failure rate. See https://codebeamer.bmwgroup.net/cb/issue/14717910
    observed_total_failure_rate_values = {
        "removed_sessions": sum([len(l) for l in removed_sessions.get(ReasonsToRemove.TotalFailureSessions).values()])
        if removed_sessions.get(ReasonsToRemove.TotalFailureSessions) is not None
        else 0,
    }
    observed_total_failure_rate_values["all_sessions"] = (
        df_copy.shape[0] + observed_total_failure_rate_values["removed_sessions"]
    )
    posterior_total_failure_rate_values = {
        "removed_sessions": observed_total_failure_rate_values["removed_sessions"] + 1,
        "all_sessions": observed_total_failure_rate_values["all_sessions"] + 2,
    }

    posterior_total_failure_rate_values["total_failure_probability"] = (
        posterior_total_failure_rate_values["removed_sessions"]
    ) / (posterior_total_failure_rate_values["all_sessions"])

    total_failure_info = TotalFailureInfo(
        posterior_total_failure_probability=posterior_total_failure_rate_values["total_failure_probability"],
        observed_total_failures=observed_total_failure_rate_values["removed_sessions"],
    )

    print(f"Removed a total of {total_removed}/{should_removed} sessions successfully from df.")
    print(f"Dataframe now has {df_copy.shape[0]} entries, had {df_copy.shape[0]+total_removed}.")
    print(
        f"Following sessions are removed:\n"
        + "\n".join([f"{reason} : {removed_sessions[reason]}" for reason in removed_sessions.keys()])
    )
    print(
        f"Observed total failure rate: {observed_total_failure_rate_values['removed_sessions']} per {observed_total_failure_rate_values['all_sessions']} sessions."
    )
    print(
        f"Posterior total failure probability with uniform prior: {posterior_total_failure_rate_values['removed_sessions']} per {posterior_total_failure_rate_values['all_sessions']} sessions ({posterior_total_failure_rate_values['total_failure_probability']:.3%})."
    )

    return df_copy.reset_index().drop("index", axis=1), total_failure_info
