"""TRAIN_SCENARIO_V1 的确定性、支持、隔离与环境兼容性测试。"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from auv_risk_rl.env.scenario_generator import (
    LocalTrainingScenarioGenerator,
    load_training_scenario_config,
    make_local_navigation_env,
)
from auv_risk_rl.frames import relative_position_body, rotation_body_to_ned
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.sensors.rays import ray_directions_body
from auv_risk_rl.sensors.visibility import (
    body_bearing_elevation_rad,
    is_obstacle_geometrically_visible,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/train_scenario_v1.yaml"


@pytest.fixture(scope="module")
def scenario_config():
    return load_training_scenario_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def generator(scenario_config, project_config):
    return LocalTrainingScenarioGenerator(scenario_config, project_config)


def test_sg_01_same_seed_index_identical(generator) -> None:
    left = generator.generate(86021, 18).to_dict()
    right = generator.generate(86021, 18).to_dict()
    assert left == right


def test_sg_02_independent_process_identical(generator) -> None:
    expected = json.dumps(generator.generate(86022, 19).to_dict(), sort_keys=True)
    code = (
        "import json;"
        "from auv_risk_rl.config import load_project_config;"
        "from auv_risk_rl.env.scenario_generator import "
        "LocalTrainingScenarioGenerator,load_training_scenario_config;"
        "c=load_training_scenario_config('configs/train_scenario_v1.yaml');"
        "p=load_project_config('configs/stage0.yaml');"
        "print(json.dumps(LocalTrainingScenarioGenerator(c,p).generate(86022,19).to_dict(),"
        "sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected


def test_sg_03_04_stable_distinct_ids(generator) -> None:
    first = generator.generate(77, 0)
    second = generator.generate(77, 1)
    assert first.scenario_id == "train-v1-seed-77-idx-0"
    assert second.scenario_id == "train-v1-seed-77-idx-1"
    assert first.scenario_id != second.scenario_id


def test_sg_05_06_start_goal_support(generator) -> None:
    for index in range(200):
        scenario = generator.generate(1000, index)
        start = scenario.initial_auv_state.position_ned_m
        goal = scenario.goal_position_ned_m
        assert 15.0 <= start[0] <= 25.0
        assert 20.0 <= start[1] <= 80.0
        assert 8.0 <= start[2] <= 32.0
        assert 75.0 <= goal[0] <= 85.0
        assert 20.0 <= goal[1] <= 80.0
        assert 8.0 <= goal[2] <= 32.0


def test_sg_07_08_even_indices_force_vertical_separation(generator) -> None:
    forced = [generator.generate(1001, index) for index in range(0, 100, 2)]
    assert len(forced) == 50
    assert all(abs(item.goal_position_ned_m[2]-item.initial_auv_state.position_ned_m[2])
               >= 4.0 for item in forced)
    assert all(item.metadata.vertical_separation_stratum == "forced_vertical_separation"
               for item in forced)


def test_sg_09_goal_line_of_sight_orientation(generator) -> None:
    for index in range(100):
        scenario = generator.generate(1002, index)
        state = scenario.initial_auv_state
        delta = scenario.goal_position_ned_m-state.position_ned_m
        expected = delta/np.linalg.norm(delta)
        actual = rotation_body_to_ned(state.yaw_rad, state.pitch_rad)[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_sg_10_initial_response_state(generator) -> None:
    state = generator.generate(1003, 0).initial_auv_state
    assert state.surge_speed_mps == 0.3
    assert state.yaw_rate_rad_s == 0.0
    assert state.pitch_rate_rad_s == 0.0


def test_sg_11_12_count_support_and_single_draw(generator) -> None:
    for index in range(1000):
        scenario = generator.generate(1004, index)
        assert scenario.metadata.obstacle_count in {1, 2, 3, 4}
        assert len(scenario.initial_obstacle_states) == scenario.metadata.obstacle_count
        assert scenario.metadata.count_draw_count == 1


def test_sg_13_20_obstacle_support_finiteness_and_cv(generator) -> None:
    for index in range(500):
        scenario = generator.generate(1005, index)
        assert scenario.metadata.motion_model == "CV"
        assert scenario.metadata.current == "zero"
        for state, metadata in zip(
            scenario.initial_obstacle_states, scenario.metadata.obstacles, strict=True
        ):
            assert 8.0 <= metadata.distance_m <= 23.0
            assert -50.0 <= metadata.horizontal_angle_deg <= 50.0
            assert -25.0 <= metadata.vertical_angle_deg <= 25.0
            assert 0.5 <= state.radius_m <= 1.0
            assert 0.2 <= metadata.speed_mps <= 0.8
            assert np.all(np.isfinite(state.position_ned_m))
            assert np.all(np.isfinite(state.velocity_ned_mps))
            np.testing.assert_allclose(
                np.linalg.norm(state.velocity_ned_mps), metadata.speed_mps,
                rtol=1.0e-14, atol=1.0e-14,
            )


def test_sensor_support_consistency(generator, project_config) -> None:
    for index in range(200):
        scenario = generator.generate(1006, index)
        for obstacle in scenario.initial_obstacle_states:
            visible, _ = is_obstacle_geometrically_visible(
                scenario.initial_auv_state, obstacle, project_config.sensor
            )
            assert visible


def test_angle_convention_roundtrip_1000_samples(generator) -> None:
    checked = 0
    index = 0
    while checked < 1000:
        scenario = generator.generate(1007, index)
        state = scenario.initial_auv_state
        for obstacle, metadata in zip(
            scenario.initial_obstacle_states, scenario.metadata.obstacles, strict=True
        ):
            relative = relative_position_body(
                state.position_ned_m, obstacle.position_ned_m,
                state.yaw_rad, state.pitch_rad,
            )
            bearing, elevation = body_bearing_elevation_rad(relative)
            np.testing.assert_allclose(np.linalg.norm(relative), metadata.distance_m,
                                       rtol=0.0, atol=1.0e-12)
            np.testing.assert_allclose(np.degrees(bearing), metadata.horizontal_angle_deg,
                                       rtol=0.0, atol=1.0e-12)
            np.testing.assert_allclose(np.degrees(elevation), metadata.vertical_angle_deg,
                                       rtol=0.0, atol=1.0e-12)
            checked += 1
            if checked == 1000:
                break
        index += 1


def test_body_direction_matches_existing_ray_convention(generator, project_config) -> None:
    directions = ray_directions_body(project_config.sensor)
    bearings = (np.arange(9)+0.5)*np.deg2rad(120.0)/9-np.deg2rad(60.0)
    elevations = (np.arange(5)+0.5)*np.deg2rad(60.0)/5-np.deg2rad(30.0)
    expected = np.array([
        generator._body_direction(bearing, elevation)
        for elevation in elevations for bearing in bearings
    ])
    np.testing.assert_allclose(directions, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("namespace", ["sensor_noise", "dropout", "environment"])
def test_sg_21_23_rng_isolation(generator, namespace) -> None:
    path_a = SeedManager(2001)
    expected = path_a.get_rng(namespace).normal(size=20)
    path_b = SeedManager(2001)
    for index in range(25):
        generator.generate(2001, index)
    actual = path_b.get_rng(namespace).normal(size=20)
    np.testing.assert_array_equal(actual, expected)


def test_sg_24_no_ood_leakage_10000(generator) -> None:
    for index in range(10_000):
        scenario = generator.generate(3001, index)
        assert 1 <= scenario.metadata.obstacle_count <= 4
        assert scenario.metadata.motion_model == "CV"
        assert scenario.metadata.current == "zero"
        assert all(metadata.speed_mps <= 0.8 for metadata in scenario.metadata.obstacles)


def test_no_hidden_rejection_contract(generator) -> None:
    for index in range(500):
        metadata = generator.generate(3002, index).metadata
        assert metadata.count_draw_count == 1
        assert metadata.generation_retry_count == (
            metadata.goal_depth_retry_count+metadata.velocity_direction_retry_count
        )
        if index % 2:
            assert metadata.goal_depth_retry_count == 0


def test_generator_has_no_policy_or_screening_imports() -> None:
    path = ROOT / "src/auv_risk_rl/env/scenario_generator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("auv_risk_rl.rl", "auv_risk_rl.risk", "auv_risk_rl.safety",
                 "auv_risk_rl.env.local_task.task_reward", "auv_risk_rl.costs")
    assert not any(module and module.startswith(forbidden) for module in modules)
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert calls.isdisjoint({"policy", "Q", "risk", "validator", "TTC", "task_reward"})


def test_factory_preserves_exact_scenario(generator, project_config) -> None:
    scenario = generator.generate(4001, 2)
    env = make_local_navigation_env(project_config, scenario)
    assert env.scenario_id == scenario.scenario_id
    actual_auv = env._initial_auv
    expected_auv = scenario.initial_auv_state
    np.testing.assert_array_equal(actual_auv.position_ned_m, expected_auv.position_ned_m)
    assert actual_auv.yaw_rad == expected_auv.yaw_rad
    assert actual_auv.pitch_rad == expected_auv.pitch_rad
    assert actual_auv.surge_speed_mps == expected_auv.surge_speed_mps
    assert actual_auv.yaw_rate_rad_s == expected_auv.yaw_rate_rad_s
    assert actual_auv.pitch_rate_rad_s == expected_auv.pitch_rate_rad_s
    assert len(env._initial_obstacles) == len(scenario.initial_obstacle_states)
    for actual, expected in zip(
        env._initial_obstacles, scenario.initial_obstacle_states, strict=True
    ):
        assert actual.obstacle_id == expected.obstacle_id
        np.testing.assert_array_equal(actual.position_ned_m, expected.position_ned_m)
        np.testing.assert_array_equal(actual.velocity_ned_mps, expected.velocity_ned_mps)
        assert actual.radius_m == expected.radius_m
    np.testing.assert_array_equal(env.goal_position_ned_m, scenario.goal_position_ned_m)


def test_fixed_environment_reset_regression(generator, project_config) -> None:
    scenario = generator.generate(4002, 3)
    env = make_local_navigation_env(project_config, scenario)
    first_obs, first_info = env.reset(seed=991)
    second_obs, second_info = env.reset(seed=991)
    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_info == second_info


def test_100_scenario_legal_warmup(generator, project_config) -> None:
    for index in range(100):
        scenario = generator.generate(5001, index)
        env = make_local_navigation_env(project_config, scenario)
        observation, info = env.reset(seed=6000+index)
        assert observation.shape == (234,)
        assert np.all(np.isfinite(observation))
        assert info["warmup_duration_s"] == 1.0
        assert info["scenario_id"] == scenario.scenario_id


def test_scenario_arrays_are_read_only(generator) -> None:
    scenario = generator.generate(7001, 0)
    assert not scenario.initial_auv_state.position_ned_m.flags.writeable
    assert not scenario.goal_position_ned_m.flags.writeable
    assert all(not state.position_ned_m.flags.writeable
               and not state.velocity_ned_mps.flags.writeable
               for state in scenario.initial_obstacle_states)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("obstacle_count_values", [1, 2, 3, 4, 5]),
        ("obstacle_speed_range_mps", [0.2, 1.2]),
        ("motion_model", "CT"),
        ("current", "unknown"),
    ],
)
def test_config_rejects_ood_values(tmp_path, field, value) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_scenario_config(path)
