#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#


from dataclasses import dataclass
from enum import Enum

import numpy as np


class TrafficParticipant(str, Enum):
    host = "host"
    opponent = "opponent"


@dataclass
class InjuryRiskLevels:
    il1_plus: np.ndarray
    il2_plus: np.ndarray
    il3: np.ndarray


def injury_level_probability_frontal_collision(
    deltaV_kmh: np.ndarray, traffic_participant: TrafficParticipant
) -> InjuryRiskLevels:

    # https://codebeamer.bmwgroup.net/cb/issue/7859753
    if traffic_participant == TrafficParticipant.host:
        p_il1 = 1 / (1 + np.exp(-1 * (-5.1323 + 0.086538 * (deltaV_kmh + 15) / 0.7 - 1.5850)))
        p_il2 = 1 / (1 + np.exp(-1 * (-5.4928 + 0.095705 * (deltaV_kmh) / 0.7 - 1.5850)))
        p_il3 = 1 / (1 + np.exp(-1 * (-4.9090 + 0.095 * (deltaV_kmh - 33) / 0.7 - 1.5850)))

    elif traffic_participant == TrafficParticipant.opponent:
        p_il1 = 1 / (1 + np.exp(-1 * (-5.0901 + 0.0889514 * (deltaV_kmh + 15) / 0.7 - 1.5850)))
        p_il2 = 1 / (1 + np.exp(-1 * (-5.3205 + 0.095865 * (deltaV_kmh) / 0.7 - 1.5850)))
        p_il3 = 1 / (1 + np.exp(-1 * (-4.9090 + 0.095 * (deltaV_kmh - 33) / 0.7 - 1.5850)))

    else:
        raise ValueError(
            f"Invalid traffic participant, expected {TrafficParticipant.host} or {TrafficParticipant.opponent}, got {traffic_participant} instead."
        )

    # https://codebeamer.bmwgroup.net/cb/issue/5876933 (Low Speed Crashes)
    p_il1[deltaV_kmh <= 10] = 0
    p_il2[deltaV_kmh <= 10] = 0
    p_il3[deltaV_kmh <= 10] = 0

    return InjuryRiskLevels(il1_plus=p_il1, il2_plus=p_il2, il3=p_il3)


def injury_level_probability_pedestrian_collision(collision_velocity_kmh: np.ndarray) -> InjuryRiskLevels:

    # https://codebeamer.bmwgroup.net/cb/issue/3048704
    p_il1 = 1 - 1 / (1 + np.exp(-(0.001 + collision_velocity_kmh * (-0.7214381))))
    p_il2 = 1 - 1 / (1 + np.exp(-(1.911387 + collision_velocity_kmh * (-0.07877268))))
    p_il3 = 1 - 1 / (1 + np.exp(-(5.93883298 + collision_velocity_kmh * (-0.07833647))))

    # https://codebeamer.bmwgroup.net/cb/issue/3048704
    # Adaption to very small collision speeds only valid for collision speeds < 1kmh (0.2777777 m/s)
    p_il2_adapt = 0.13792713 * collision_velocity_kmh
    p_il3_adapt = 0.00284173 * collision_velocity_kmh

    idx = collision_velocity_kmh < 1
    p_il2[idx] = p_il2_adapt[idx]
    p_il3[idx] = p_il3_adapt[idx]

    return InjuryRiskLevels(il1_plus=p_il1, il2_plus=p_il2, il3=p_il3)


def injury_level_probability_smalloverlap_collision(
    deltaV_kmh: np.ndarray, traffic_participant: TrafficParticipant
) -> InjuryRiskLevels:

    # https://codebeamer.bmwgroup.net/cb/issue/2897626
    if traffic_participant == TrafficParticipant.host:
        p_il1 = 1 / (1 + np.exp(-1 * (-4.6281 + 0.077868 * (deltaV_kmh + 18) / 0.7 - 1.5850)))
        p_il2 = 1 / (1 + np.exp(-1 * (-5.3213 + 0.091549 * (deltaV_kmh + 3) / 0.7 - 1.5850)))
        p_il3 = 1 / (1 + np.exp(-1 * (-4.9090 + 0.095 * (deltaV_kmh - 25) / 0.7 - 1.5850)))

    elif traffic_participant == TrafficParticipant.opponent:
        p_il1 = 1 / (1 + np.exp(-1 * (-4.6551 + 0.082058 * (deltaV_kmh + 18) / 0.7 - 1.5850)))
        p_il2 = 1 / (1 + np.exp(-1 * (-5.1817 + 0.092518 * (deltaV_kmh + 3) / 0.7 - 1.5850)))
        p_il3 = 1 / (1 + np.exp(-1 * (-4.9090 + 0.095 * (deltaV_kmh - 25) / 0.7 - 1.5850)))

    else:
        raise ValueError(
            f"Invalid traffic participant, expected {TrafficParticipant.host} or {TrafficParticipant.opponent}, got {traffic_participant} instead."
        )

    # https://codebeamer.bmwgroup.net/cb/issue/5876933 (Low Speed Crashes)
    p_il1[deltaV_kmh <= 10] = 0
    p_il2[deltaV_kmh <= 10] = 0
    p_il3[deltaV_kmh <= 10] = 0

    return InjuryRiskLevels(il1_plus=p_il1, il2_plus=p_il2, il3=p_il3)
