#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator


class CategoryModel(BaseModel):
    value: List[str]

    class Config:
        extra = Extra.allow


class SensorModel(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def categories_have_valid_json_schema(cls, values: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        for category in values.values():
            CategoryModel(**category)
        return values


class InvalidSession(BaseModel):
    class Config:
        extra = Extra.allow

    @root_validator
    def sensors_have_valid_json_schema(
        cls, values: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        for sensor in values.values():
            SensorModel(**sensor)
        return values
