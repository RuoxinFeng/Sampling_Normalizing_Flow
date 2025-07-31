#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


from enum import Enum


class SRSFormattingCategories(str, Enum):
    SRS_template = "SRS-TEMPLATE"
    SRS_template_ifm = "SRS-TEMPLATE-IFM"
    SRS04_1 = "SRS-04-1"
    SRS04_1_adcam_brake_trigger = "SRS-04-1-adcam_brake_trigger"
    SRS06 = "SRS-06"
    SRS06_adcam_brake_trigger = "SRS-06-adcam_brake_trigger"
    SRS06_ifm = "SRS-06-ifm"
    SRS06_ifm_observer = "SRS-06-ifm-observer"
    SRS08 = "SRS-08"
    SRS08_ifm = "SRS-08-ifm"
    SRS08_plan = "SRS-08-plan"
    SRS11_moving = "SRS-11-moving"
    SRS11_stationary = "SRS-11-stationary"
    SRS11_ifm = "SRS-11-ifm"
    SRS12 = "SRS-12"
    SRS12_ifm = "SRS-12-ifm"
