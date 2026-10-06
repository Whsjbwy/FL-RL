"""Phase A 外置一致性审计：只调用原项目，不修改生产代码，不执行 RL 或 F0/F1。"""
from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from auv_risk_rl.config import load_project_config
from auv_risk_rl.costs.finite_horizon import (
    finite_horizon_discount_normalizer,
    normalized_finite_horizon_cost,
)
from auv_risk_rl.dynamics.auv_kinematics import rollout_constant_command
from auv_risk_rl.env.geometry import moving_sphere_minimum_clearance
from auv_risk_rl.env.world import AUVWorld, _operation_boundary_fraction
from auv_risk_rl.logging_utils import ResearchLoggerAdapter
from auv_risk_rl.prediction.predictor import predict_position_distribution
from auv_risk_rl.risk.gaussian_bounds import (
    aggregate_union_bounds,
    point_collision_upper_bound,
    segment_projection_direction,
)
from auv_risk_rl.safety.candidates import build_candidate_actions
from auv_risk_rl.safety.validator import validate_nominal_action
from auv_risk_rl.sensors.delay_queue import DetectionDelayQueue
from auv_risk_rl.tracking.kalman_filter import predict_cv_state, update_position_measurement
from auv_risk_rl.types import (
    AUVState, ControlCommand, GroundTruthObstacleState, KFTrackState,
    SensorDetection, TrackedObstacle,
)

ROOT = Path(os.environ['AUV_AUDIT_PROJECT_ROOT'])


@pytest.fixture(scope='session')
def cfg():
    """读取上传的冻结 Stage 0 配置，不改动配置文件。"""
    return load_project_config(ROOT / 'configs' / 'stage0.yaml')


def state(position=(50.0, 50.0, 20.0), speed=0.8):
    """构造明确米制诊断初值，不使用训练场景池。"""
    return AUVState(np.asarray(position, dtype=np.float64), 0., 0., speed, 0., 0.)


def test_PA01_segment_direction_equals_native_LOCAL_equation_38(cfg):
    """原式（38）使用两端均值之和的归一化，不是最近点方向。"""
    m0 = np.array([2., 0., 0.]); m1 = np.array([2., 2., 0.])
    expected = (m0 + m1) / np.linalg.norm(m0 + m1)
    actual = segment_projection_direction(m0, m1, cfg.risk.mean_norm_tolerance_m)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)


def test_PA02_positive_tiny_variance_bound_dominates_exact_rank1_ball_probability(cfg):
    """半正定秩一高斯可用一维区间 CDF 精确计算球事件，正方差不得当零。"""
    std = 1e-7
    mean = np.array([1. + .5 * std, 0., 0.])
    covariance = np.diag([std * std, 0., 0.])
    actual = point_collision_upper_bound(
        mean, covariance, 1., cfg.risk.mean_norm_tolerance_m,
        cfg.risk.variance_tolerance_m2, cfg.kf.covariance_symmetry_tolerance,
        cfg.kf.covariance_psd_tolerance,
    )
    exact = float(ndtr((1. - mean[0]) / std) - ndtr((-1. - mean[0]) / std))
    assert actual + 1e-12 >= exact, (actual, exact)


def test_PA03_invalid_covariance_returns_recorded_unverified_fallback(cfg):
    """原协议要求无效协方差记录并进入未验证执行路径，不从入口未处理抛出。"""
    track = KFTrackState(1, np.array([70., 50., 20., .2, 0., 0.]),
                         np.diag([-1., 1., 1., 1., 1., 1.]), 0., 0.)
    nominal = ControlCommand(.8, 0., 0.)
    result = validate_nominal_action(state(), 0., nominal, nominal,
                                     [TrackedObstacle(track, .5)], cfg)
    assert result.decision_type == 'fallback'
    assert 'numerical' in result.reason or 'unverified' in result.reason


def test_PA04_backup_obeys_latest_violation_then_U_distance_id(cfg):
    """无硬约束可行候选但有有效预测时，检查已冻结 backup 字典序。"""
    initial = state((98.8, 50., 20.), 1.5)
    nominal = ControlCommand(1.5, 0., 0.)
    candidates = build_candidate_actions(nominal, nominal, cfg.validator)
    ranking = []
    for candidate_id, action in enumerate(candidates):
        path = rollout_constant_command(initial, action, np.zeros(3),
            cfg.validator.validation_horizon_s, cfg.dynamics.integration_dt_s, cfg.dynamics)
        violation_time = None
        for m, (a, b) in enumerate(zip(path, path[1:])):
            fraction = _operation_boundary_fraction(a, b, cfg)
            if fraction is not None:
                violation_time = (m + fraction) * cfg.dynamics.integration_dt_s
                break
        assert violation_time is not None
        vector = np.array([
            2 * (action.surge_speed_command_mps - cfg.dynamics.min_surge_speed_mps)
              / (cfg.dynamics.max_surge_speed_mps - cfg.dynamics.min_surge_speed_mps) - 1,
            action.yaw_rate_command_rad_s / cfg.dynamics.max_yaw_rate_rad_s,
            action.pitch_rate_command_rad_s / cfg.dynamics.max_pitch_rate_rad_s,
        ])
        distance = float(np.linalg.norm(vector - np.array([1., 0., 0.])))
        # 本诊断无障碍，各候选模型风险均为零，时间最迟是首排序键。
        ranking.append(((-violation_time, 0., distance, candidate_id), action))
    expected = min(ranking, key=lambda item: item[0])[1]
    actual = validate_nominal_action(initial, 0., nominal, nominal, [], cfg)
    assert actual.executed_action == expected, (actual, expected)


def test_PA05_minimum_clearance_covers_all_control_subsegments(cfg):
    """周期最小净间距不能仅返回最后一个积分小步。"""
    initial = state(speed=.3)
    obstacle = GroundTruthObstacleState(1, np.array([50.03, 51.75, 20.]),
                                        np.array([-.8, 0., 0.]), .5)
    world = AUVWorld(cfg, initial, (obstacle,), np.array([90., 90., 20.]))
    result = world.step(ControlCommand(.3, 0., 0.))
    expected, _ = moving_sphere_minimum_clearance(
        initial.position_ned_m, result.auv_state.position_ned_m,
        obstacle.position_ned_m, result.obstacle_states[0].position_ned_m,
        cfg.risk.auv_radius_m + obstacle.radius_m,
    )
    assert result.event.reason == 'none'
    assert result.event.minimum_clearance_m == pytest.approx(expected, abs=1e-12)


def test_PA06_terminal_clearance_excludes_unexecuted_suffix(cfg):
    """首次接触处停止时，不能把该小步后续未执行部分的穿透计为真实净间距。"""
    initial = state(speed=.3)
    obstacle = GroundTruthObstacleState(3, np.array([51.3, 50., 20.]),
                                        np.array([-.8, 0., 0.]), .5)
    result = AUVWorld(cfg, initial, (obstacle,), np.array([90., 90., 20.])).step(
        ControlCommand(.3, 0., 0.))
    assert result.event.reason == 'collision'
    assert abs(result.event.minimum_clearance_m) <= 1e-12


def test_PA07_context_logger_accepts_documented_context():
    """配置中 module 字段不得与标准 LogRecord 保留字段冲突。"""
    logger = logging.getLogger('phase_a_test_logger')
    logger.setLevel(logging.INFO); logger.propagate = False
    logger.addHandler(logging.NullHandler())
    adapter = ResearchLoggerAdapter(logger, dict(stage_id='S0', run_id='audit', seed=1,
                                   scenario_id='unit', module='risk'))
    adapter.info('独立审计日志')


def test_PA08_one_step_delay_handles_accumulated_control_clock(cfg):
    """0.2 秒加延迟与八次 0.05 秒积分的舍入差，不能增加整控制步的延迟。"""
    queue = DetectionDelayQueue()
    detection = SensorDetection(1, np.ones(3), np.eye(3),
                                cfg.dynamics.control_dt_s, 2 * cfg.dynamics.control_dt_s)
    queue.enqueue_many((detection,))
    clock = 0.
    for _ in range(8):
        clock += cfg.dynamics.integration_dt_s
    arrived = queue.pop_arrived(clock)
    assert len(arrived) == 1, (clock, detection.arrival_timestamp_s)


def test_PA09_KF_projection_matches_independent_blocks(cfg):
    """独立块公式核对六维预测到三维位置边缘，包含过程噪声。"""
    mean = np.arange(6, dtype=float)
    covariance = np.diag([1., 1.2, 1.4, .4, .5, .6])
    covariance[0, 3] = covariance[3, 0] = .1
    qa = np.diag(cfg.kf.acceleration_spectral_density_m2_s3)
    horizon = 2.3
    track = KFTrackState(1, mean, covariance, 0., 0.)
    actual = predict_position_distribution(track, horizon, qa, 1e-10, 1e-10)
    expected = (covariance[:3, :3]
                + horizon * (covariance[:3, 3:] + covariance[3:, :3])
                + horizon ** 2 * covariance[3:, 3:]
                + horizon ** 3 / 3 * qa)
    np.testing.assert_allclose(actual.position_covariance_ned_m2, expected,
                               atol=1e-12, rtol=0.)
    assert actual.position_covariance_ned_m2.shape == (3, 3)
    assert actual.position_covariance_ned_m2.dtype == np.float64


def test_PA10_Joseph_matches_gaussian_conditional_covariance(cfg):
    """用独立高斯条件协方差公式核对 Joseph 结果。"""
    qa = np.diag(cfg.kf.acceleration_spectral_density_m2_s3)
    pm, pc = predict_cv_state(np.arange(6, dtype=float), np.eye(6), .2, qa, 1e-10, 1e-10)
    h = np.c_[np.eye(3), np.zeros((3, 3))]
    r = np.diag([.04, .04, .09])
    _, actual, _, _ = update_position_measurement(pm, pc, np.array([1., 2., 3.]),
                                                  r, 1e-10, 1e-10)
    expected = pc - pc @ h.T @ np.linalg.solve(h @ pc @ h.T + r, h @ pc)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)


def test_PA11_cost_ledger_matches_explicit_success_and_failure_sequences(cfg):
    """只验数学账本，不把它冒充未实现的神经 cost Bellman target。"""
    gamma = cfg.cost.discount_gamma; horizon = cfg.cost.planned_horizon_control_steps
    beta = finite_horizon_discount_normalizer(gamma, horizon)
    for step in (0, 37, 999):
        costs = [.25] * step + [1.]
        actual = normalized_finite_horizon_cost(costs, gamma, horizon, step)
        sequence = [.25] * step + [1.] * (horizon - step)
        expected = beta * sum(gamma ** i * c for i, c in enumerate(sequence))
        assert abs(actual - expected) < 1e-12
    success = normalized_finite_horizon_cost([.25, 1.], gamma, horizon)
    assert abs(success - beta * (.25 + gamma)) < 1e-12


def test_PA12_union_aggregation_retains_unclipped_sum():
    """已实现的并集聚合确实保存 U，不改成最大单项。"""
    result = aggregate_union_bounds([.4, .7, .2])
    assert abs(result.untruncated_union_bound - 1.3) <= 1e-12
    assert result.risk_upper_bound == 1.


def test_PA13_timeout_is_real_termination(cfg):
    """无 RL 世界的任务到期使用真实终止；不代表 sampler 截断接口已实现。"""
    short = replace(cfg, environment=replace(cfg.environment, max_episode_control_steps=1),
                    cost=replace(cfg.cost, planned_horizon_control_steps=1))
    result = AUVWorld(short, state(), (), np.array([90., 90., 20.])).step(
        ControlCommand(.8, 0., 0.))
    assert result.is_terminated
    assert result.event.reason == 'timeout'
