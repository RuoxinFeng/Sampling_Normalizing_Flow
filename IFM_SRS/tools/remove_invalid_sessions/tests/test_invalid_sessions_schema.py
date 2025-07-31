#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from typing import Any, Dict

import pydantic
import pytest
from tools.remove_invalid_sessions.invalid_sessions_schema import InvalidSession


@pytest.fixture
def valid_config() -> Dict[str, Dict[str, Any]]:
    return {
        "global": {
            "rt_range_issues_automatic": {
                "__comment": "Do not make any manual changes, the field is created automatically! AT range fault. Sessions are marked with \u201cRT_RANGE_CHECK_\u201c or marked \u201cEM_SCENECAT_DEFECT_RTRANGE_START.\u201d ",
                "value": [
                    "0bf3e484-6fcd-4c1a-aa43-018801682cfa",
                    "7ae24366-1460-4f43-a7e2-0186cb9b4e67",
                    "1c818f8d-0c4c-4ac1-ad51-0186cb69bad9",
                ],
            }
        },
        "adcam": {
            "some_category": {"value": ["3445fcd2-2b45-499a-acc9-0186dace0a76", "1afe8e1a-065f-42d1-ad84-0186df31c23a"]}
        },
        "lidar": {},
        "frr": {},
    }


def test__valid_config(valid_config: Dict[str, Dict[str, Any]]):
    InvalidSession(**valid_config)


def test__raises_when_no_value_in_invalid_sessions_dict(valid_config: Dict[str, Dict[str, Any]]):
    del valid_config["global"]["rt_range_issues_automatic"]["value"]
    with pytest.raises(pydantic.error_wrappers.ValidationError, match="field required"):
        InvalidSession(**valid_config)


def test__raises_when_value_is_not_a_list(valid_config: Dict[str, Dict[str, Any]]):
    valid_config["global"]["rt_range_issues_automatic"]["value"] = "not a list"
    with pytest.raises(pydantic.error_wrappers.ValidationError, match="value is not a valid list"):
        InvalidSession(**valid_config)


def test__raises_when_value_is_not_a_list_of_strings(valid_config: Dict[str, Dict[str, Any]]):
    valid_config["global"]["rt_range_issues_automatic"]["value"] = [[]]
    with pytest.raises(pydantic.error_wrappers.ValidationError, match="str type expected"):
        InvalidSession(**valid_config)
