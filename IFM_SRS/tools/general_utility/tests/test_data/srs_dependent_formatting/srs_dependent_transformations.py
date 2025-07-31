#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

from collections import defaultdict

import pandas as pd
from tools.general_utility.srs_formatting_categories import SRSFormattingCategories


def apply_data_transformation(df: pd.DataFrame, srs: SRSFormattingCategories) -> pd.DataFrame:
    def srs04_1_data_transformation(df: pd.DataFrame) -> pd.DataFrame:
        df["combined_column"] = df["column_2"] * df["column_3"].abs()
        df["some_other_column_new"] = df["some_other_column_new"] * 100
        df["categorical_column"] = df["categorical_column"].replace(["car", "motorcycle"], "car or motorcycle")
        return df

    def srs_template_data_transformation(df: pd.DataFrame) -> pd.DataFrame:
        df["branch"] = df["branch"].replace("sop/2311", "sop/2312")
        return df

    def no_transformation(df: pd.DataFrame) -> pd.DataFrame:
        return df

    transformation_dict = defaultdict(lambda: no_transformation)
    transformation_dict[SRSFormattingCategories.SRS04_1] = srs04_1_data_transformation
    transformation_dict[SRSFormattingCategories.SRS_template] = srs_template_data_transformation
    return transformation_dict[srs](df.copy())
