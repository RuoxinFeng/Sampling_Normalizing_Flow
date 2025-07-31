#
# Copyright (C) 2024, Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
#

import numpy as np
import pytest
from tools.injury_risk_level_functions.srs07_irls import (
    InjuryRiskLevels,
    TrafficParticipant,
    injury_level_probability_frontal_collision,
    injury_level_probability_pedestrian_collision,
    injury_level_probability_smalloverlap_collision,
)

TEST_DATA_HIGH_VELOCITIES = np.linspace(10.1, 100, 3)
TEST_DATA_SMALL_VELOCITIES = np.linspace(0.1, 9.9, 3)
ZERO_INJURY_RISK = InjuryRiskLevels(np.zeros(3), np.zeros(3), np.zeros(3))


def assert_irls_equal(act: InjuryRiskLevels, exp: InjuryRiskLevels):
    np.testing.assert_almost_equal(act.il1_plus, exp.il1_plus)
    np.testing.assert_almost_equal(act.il2_plus, exp.il2_plus)
    np.testing.assert_almost_equal(act.il3, exp.il3)


def test__injury_level_probability_frontal_collision__yields_expected_results_for_high_velocities():
    p_il_host_exp = InjuryRiskLevels(
        il1_plus=[0.02622941, 0.87464642, 0.99944703],
        il2_plus=[0.00334508, 0.61033359, 0.99863378],
        il3=[6.75985014e-05, 2.92695813e-02, 9.30787991e-01],
    )
    p_il_opponent_exp = InjuryRiskLevels(
        il1_plus=[0.02972598, 0.90259476, 0.99964333],
        il2_plus=[0.00398072, 0.65330474, 0.99887573],
        il3=[6.75985014e-05, 2.92695813e-02, 9.30787991e-01],
    )
    assert_irls_equal(
        injury_level_probability_frontal_collision(TEST_DATA_HIGH_VELOCITIES, TrafficParticipant.host), p_il_host_exp
    )
    assert_irls_equal(
        injury_level_probability_frontal_collision(TEST_DATA_HIGH_VELOCITIES, TrafficParticipant.opponent),
        p_il_opponent_exp,
    )


def test__injury_level_probability_frontal_collision__yields_no_injuries_for_small_velocities():
    for traffic_participant in [TrafficParticipant.host, TrafficParticipant.opponent]:
        assert_irls_equal(
            injury_level_probability_frontal_collision(TEST_DATA_SMALL_VELOCITIES, traffic_participant),
            ZERO_INJURY_RISK,
        )


def test__injury_level_probability_frontal_collision__raises_for_invalid_traffic_participant():
    with pytest.raises(
        ValueError,
        match="Invalid traffic participant, expected host or opponent, got not_a_traffic_participant instead.",
    ):
        injury_level_probability_frontal_collision(TEST_DATA_HIGH_VELOCITIES, "not_a_traffic_participant")


def test__injury_level_probability_pedestrian_collision__yields_expected_results_for_high_velocities():
    test_data_high_velocities = np.linspace(1.1, 100, 3)
    p_il_pedestrian_exp = InjuryRiskLevels(
        il1_plus=[0.68838545, 1.0, 1.0],
        il2_plus=[0.13886643, 0.88800986, 0.99744178],
        il3=[0.00286401, 0.12143354, 0.86930345],
    )
    assert_irls_equal(injury_level_probability_pedestrian_collision(test_data_high_velocities), p_il_pedestrian_exp)


def test__injury_level_probability_pedestrian_collision__yields_expected_results_for_small_velocities():
    test_data_small_velocities = np.linspace(0.1, 0.9, 3)
    p_il_pedestrian_exp = InjuryRiskLevels(
        il1_plus=[0.51777845, 0.58897242, 0.65662598],
        il2_plus=0.13792713 * test_data_small_velocities,
        il3=0.00284173 * test_data_small_velocities,
    )
    assert_irls_equal(injury_level_probability_pedestrian_collision(test_data_small_velocities), p_il_pedestrian_exp)


def test__injury_level_probability_smalloverlap_collision__yields_expected_results_for_high_velocities():
    p_il_host_exp = InjuryRiskLevels(
        il1_plus=[0.043636, 0.87135379, 0.99900644],
        il2_plus=[0.00552448, 0.66501588, 0.99859243],
        il3=[2.00170101e-04, 8.19768503e-02, 9.75507035e-01],
    )
    p_il_opponent_exp = InjuryRiskLevels(
        il1_plus=[0.04992318, 0.91078344, 0.99949605],
        il2_plus=[0.00646227, 0.71211649, 0.99893813],
        il3=[2.00170101e-04, 8.19768503e-02, 9.75507035e-01],
    )
    assert_irls_equal(
        injury_level_probability_smalloverlap_collision(TEST_DATA_HIGH_VELOCITIES, TrafficParticipant.host),
        p_il_host_exp,
    )
    assert_irls_equal(
        injury_level_probability_smalloverlap_collision(TEST_DATA_HIGH_VELOCITIES, TrafficParticipant.opponent),
        p_il_opponent_exp,
    )


def test__injury_level_probability_smalloverlap_collision__yields_no_injuries_for_small_velocities():
    for traffic_participant in [TrafficParticipant.host, TrafficParticipant.opponent]:
        assert_irls_equal(
            injury_level_probability_smalloverlap_collision(TEST_DATA_SMALL_VELOCITIES, traffic_participant),
            ZERO_INJURY_RISK,
        )


def test__injury_level_probability_smalloverlap_collision__raises_for_invalid_traffic_participant():
    with pytest.raises(
        ValueError,
        match="Invalid traffic participant, expected host or opponent, got not_a_traffic_participant instead.",
    ):
        injury_level_probability_smalloverlap_collision(TEST_DATA_HIGH_VELOCITIES, "not_a_traffic_participant")
