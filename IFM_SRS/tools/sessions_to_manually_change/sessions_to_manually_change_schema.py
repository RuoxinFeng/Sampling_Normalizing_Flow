#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import re
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator


def _extract_mappings_by_cb_id(
    sessions_to_change_dict: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    mappings_by_cb_id = dict()
    for cb_id, mappings in sessions_to_change_dict.items():
        mappings_by_cb_id[cb_id] = {
            session: replacement for mapping in mappings.values() for session, replacement in mapping.items()
        }
    return mappings_by_cb_id


def _extract_session_list(cb_id_to_change_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    return [session for mappings in cb_id_to_change_dict.values() for session in mappings.keys()]


def _get_all_data_types_by_variable(
    sessions_to_change_dict: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, List[type]]:
    mappings_by_cb_id = _extract_mappings_by_cb_id(sessions_to_change_dict)
    datatypes_by_cb_id = dict()
    for cb_id, mappings in mappings_by_cb_id.items():
        datatypes_by_cb_id[cb_id] = {type(replacement) for replacement in mappings.values()}
    for cb_id, datatypes in datatypes_by_cb_id.items():
        if {int, float}.issubset(datatypes):
            datatypes_by_cb_id[cb_id].remove(int)
    return datatypes_by_cb_id


class ReasonToChange(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def lowest_hierarchy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for value in values.values():
            if isinstance(value, list) or isinstance(value, dict):
                raise ValueError(
                    f"json file of sessions to manually change can have a maximum hierarchy of 3 nested dictionaries and no lists! Got {value} but expected a float, int or string."
                )
        return values


class CBIDToChange(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def clusters_have_valid_schema(cls, values: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        for reason_to_change_dict in values.values():
            ReasonToChange(**reason_to_change_dict)
        return values


class SessionsToChange(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def cb_id_dicts_have_valid_schema(
        cls, values: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        for cb_id_dict in values.values():
            CBIDToChange(**cb_id_dict)
        return values

    @root_validator
    def cb_ids_have_valid_schema(
        cls, values: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        valid_pattern = r"[0-9]{7,}[a-zA-Z]?$"
        for cb_id in values.keys():
            match = re.match(pattern=valid_pattern, string=cb_id)
            if match is None or match.string != cb_id:
                raise ValueError(f'Invalid codebeamer id "{cb_id}" provided!')
        return values

    @root_validator
    def consistent_datatypes(cls, values: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        datatypes_by_variable = _get_all_data_types_by_variable(values)
        for cb_id, datatypes in datatypes_by_variable.items():
            if len(datatypes) > 1:
                raise ValueError(f"Variable {cb_id} got an inconsistent set of datatypes for replacements: {datatypes}")
        return values

    @root_validator
    def sessions_are_unique_per_cb_id(
        cls, values: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        for cb_id, cb_id_dict in values.items():
            session_list = _extract_session_list(cb_id_dict)
            if len(session_list) > len(set(session_list)):
                raise ValueError(f"Some sessions have been defined multiple times for cb id {cb_id}")
        return values

    @property
    def mappings_by_cb_id(self) -> Dict[str, Dict[str, Any]]:
        return _extract_mappings_by_cb_id(self.dict())
