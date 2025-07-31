#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


import re
from os import listdir, walk
from os.path import isdir, join
from typing import List


def get_all_config_folder_paths(data_path: str, config_folder_pattern: str) -> List[str]:
    config_folders = []
    config_folder_pattern = config_folder_pattern
    for dir_name in listdir(data_path):
        dir_path = join(data_path, dir_name)
        if isdir(dir_path) and re.match(config_folder_pattern, dir_name):
            config_folders.append(dir_path)
    return config_folders


def get_all_file_paths(data_path: str, config_file_pattern: str) -> List[str]:
    configs = []
    for root, dirs, files in walk(data_path):
        for file in files:
            if re.match(config_file_pattern, file):
                configs.append(join(root, file))
    return configs
