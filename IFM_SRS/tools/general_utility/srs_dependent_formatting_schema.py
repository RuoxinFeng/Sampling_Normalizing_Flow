#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Dict

from pydantic import BaseModel, Extra, root_validator


class FormattingModel(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def validate_srs_names(cls, values: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        for srs in values:
            if not srs.startswith("SRS-"):
                raise ValueError(f"Key '{srs}' must start with 'SRS-'")
        return values

    @root_validator
    def validate_json_structure(cls, values: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        for srs in values:
            if not isinstance(values[srs], dict):
                raise ValueError(f"Renaming should be defined as a dictionary, but '{values[srs]}' was given for {srs}")
            if not all(isinstance(key, str) and isinstance(val, str) for key, val in values[srs].items()):
                raise ValueError(f"Column names should be strings, but {values[srs]} was given")
        return values
