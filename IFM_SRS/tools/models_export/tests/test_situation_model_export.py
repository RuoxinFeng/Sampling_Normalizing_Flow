#
# Copyright (C) 2023-2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import os
from typing import Any, Dict, Optional

import pytest
from tools.models_export.situation_model_export import save_model_as_json_file
from tools.situation_model.situation_model import ConditionalSituationModel, SituationModel
from tools.situation_model.tests.test_situation_model import (
    get_valid_conditional_situation_model_entries,
    get_valid_situation_model_entries,
)
from tools.test_helpers.file_creation_helpers import assert_file_content


@pytest.mark.parametrize("exclude_unset", [True, False])
@pytest.mark.parametrize("rounding_strategy", [{"nr_of_digits_after_comma": 4}, {"nr_of_digits_after_comma": None}, {}])
@pytest.mark.parametrize(
    "entries, model_type",
    [
        (get_valid_situation_model_entries(), SituationModel),
        (get_valid_conditional_situation_model_entries(), ConditionalSituationModel),
    ],
)
def test__save_model_as_json_file_correctly_saves_passed_situation_model_entries(
    tmpdir, exclude_unset: bool, rounding_strategy: Dict[str, Optional[int]], entries: Dict[str, Any], model_type: type
):
    model = model_type(**entries)
    file_path = os.path.join(tmpdir, "model.json")
    save_model_as_json_file(model=model, file_path=file_path, exclude_unset=exclude_unset, **rounding_strategy)

    assert_file_content(expected_content=entries, file_path=file_path, **rounding_strategy)
