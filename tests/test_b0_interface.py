"""B0特权当前状态与不执行风险过滤；全部为非学习短接口验收。"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.b0_navigation import B0NavigationEnv, B0ObservationBuilder
from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.local_task import normalized_to_command
from auv_risk_rl.env.observation import ObservationBuilder
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.sensors.rays import RayFrame
from auv_risk_rl.types import AUVState, GroundTruthObstacleState


def make_env(config: ProjectConfig, *, state: AUVState | None = None,
             obstacles: tuple[GroundTruthObstacleState, ...] = (),
             goal: np.ndarray | None = None) -> B0NavigationEnv:
    """固定合法测试场景；不改变train-v1范围或进行随机训练分布筛选。"""

    own = state or AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    return B0NavigationEnv(config, own, obstacles,
                           np.array([80., 50., 20.]) if goal is None else goal, 'b0-interface')


def empty_rays(config: ProjectConfig) -> RayFrame:
    """无回波固定射线；用于独立编码检查，不作为真实环境采样。"""

    return RayFrame(np.full(45, config.sensor.range_m), np.zeros(45))


def test_b0_own_and_ray_encoding_unchanged(project_config: ProjectConfig) -> None:
    """B0只换目标槽；自身/任务及45+45射线逐值复用原接口。"""

    state = AUVState(np.array([20., 50., 20.]), .2, -.1, .6, .1, -.1)
    rays = RayFrame(np.linspace(0., 25., 45), np.arange(45) % 2)
    args = (state, np.array([80., 50., 22.]), np.array([-.2, .3, -.4]), 955, rays, (), 2.)
    original = ObservationBuilder(project_config).build(*args)
    actual = B0ObservationBuilder(project_config).build(*args)
    assert actual.dtype == np.float32 and actual.shape == (234,)
    np.testing.assert_array_equal(actual[:108], original[:108])
    np.testing.assert_array_equal(actual[108:], np.zeros(126))


def test_b0_current_position_ground_velocity_and_zero_covariance(
    project_config: ProjectConfig,
) -> None:
    """解析Body旋转参考：yaw=pi/2时Body分量为(E,-N,D)，不减自身地速。"""

    state = AUVState(np.array([20., 50., 20.]), np.pi/2, 0., .8, .15, 0.)
    obstacle = GroundTruthObstacleState(3, np.array([24., 60., 22.]),
                                        np.array([.4, -.2, .6]), .75)
    actual = B0ObservationBuilder(project_config).build(
        state, np.array([80., 50., 20.]), np.zeros(3), 1000,
        empty_rays(project_config), (obstacle,), 1.,
    )[108:129]
    np.testing.assert_allclose(actual[:3], np.array([10., -4., 2.])/25,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual[3:6], [-.2, -.4, .6], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actual[6:18], np.zeros(12))
    np.testing.assert_array_equal(actual[18:], [.75, 0., 1.])


class PoisonFutureObstacle:
    """允许当前真值地速，未来属性及查询一旦被读取就失败。"""

    obstacle_id = 1
    position_ned_m = np.array([30., 50., 20.])
    velocity_ned_mps = np.array([-.4, .2, 0.])
    radius_m = .6

    def __init__(self, future_marker: object) -> None:
        self.future_marker = future_marker

    @property
    def future_position_ned_m(self) -> np.ndarray:
        raise AssertionError('B0 accessed future truth')

    def query_future(self, _: float) -> np.ndarray:
        raise AssertionError('B0 queried a future trajectory')


def test_b0_future_query_change_cannot_change_current_observation(
    project_config: ProjectConfig,
) -> None:
    """毒化未来对象可任意替换；同一当前快照的B0输入精确不变。"""

    state = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    builder = B0ObservationBuilder(project_config)
    args = (state, np.array([80., 50., 20.]), np.zeros(3), 1000,
            empty_rays(project_config))
    first = builder.build(*args, (PoisonFutureObstacle(object()),), 1.)
    second = builder.build(*args, (PoisonFutureObstacle({'arbitrary': 'future'}),), 1.)
    np.testing.assert_array_equal(first, second)


def test_b0_velocity_change_changes_only_velocity_encoding(project_config: ProjectConfig) -> None:
    """当前地速不是未来均值，改变地速只改变槽中三维速度编码。"""

    state = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    obstacle = GroundTruthObstacleState(1, np.array([30., 50., 20.]), np.zeros(3), .5)
    builder = B0ObservationBuilder(project_config)
    args = (state, np.array([80., 50., 20.]), np.zeros(3), 1000,
            empty_rays(project_config))
    first = builder.build(*args, (obstacle,), 1.)
    second = builder.build(*args, (replace(obstacle, velocity_ned_mps=np.array([.8, 0., 0.])),), 1.)
    assert np.flatnonzero(first != second).tolist() == [111]
    assert second[111] == pytest.approx(.8, abs=1e-7)


def test_b0_current_distance_and_id_sort_padding(project_config: ProjectConfig) -> None:
    """距离并列以稳定ID排序，最多六槽；半径标识证明顺序而不使用risk/TTC。"""

    state = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    obstacles = tuple(GroundTruthObstacleState(
        index, np.array([30., 50., 20.]), np.zeros(3), .5+index*.01,
    ) for index in reversed(range(8)))
    builder = B0ObservationBuilder(project_config)
    args = (state, np.array([80., 50., 20.]), np.zeros(3), 1000,
            empty_rays(project_config))
    actual = builder.build(*args, obstacles, 1.)
    np.testing.assert_allclose(actual[126::21], .5+np.arange(6)*.01, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actual[128::21], np.ones(6))
    actual = builder.build(*args, obstacles[:2], 1.)
    np.testing.assert_array_equal(actual[150:], np.zeros(84))


@pytest.mark.parametrize('position', [[10., 50., 20.], [70., 50., 20.]])
def test_b0_truth_slot_not_limited_by_fov_range_or_dropout(
    project_config: ProjectConfig, position: list[float],
) -> None:
    """后方/量程外目标且dropout=1仍进入B0槽；射线规则没有扩展。"""

    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=1.))
    obstacle = GroundTruthObstacleState(2, np.array(position), np.zeros(3), .5)
    env = make_env(config, obstacles=(obstacle,))
    actual, _ = env.reset(seed=900001)
    assert actual[128] == 1
    assert not env.perception.generated and not env.perception.tracks(1.)
    np.testing.assert_array_equal(actual[63:108], np.zeros(45))
    np.testing.assert_array_equal(actual[114:126], np.zeros(12))


def test_b0_step_reads_current_truth_not_initial_snapshot(project_config: ProjectConfig) -> None:
    """warm-up与正式控制后均读取真实CV当前状态，不重复使用场景物理t=0位置。"""

    obstacle = GroundTruthObstacleState(2, np.array([40., 50., 20.]),
                                        np.array([-.4, .2, 0.]), .6)
    env = make_env(project_config, obstacles=(obstacle,))
    initial, _ = env.reset(seed=900002)
    actual, _, _, _, _ = env.step(np.array([-1., 0., 0.]))
    expected_position = obstacle.position_ned_m+obstacle.velocity_ned_mps*1.2
    np.testing.assert_allclose(env.world.obstacle_states[0].position_ned_m,
                               expected_position, rtol=0, atol=1e-12)
    expected_relative = (expected_position-env.world.auv_state.position_ned_m)/25
    np.testing.assert_allclose(actual[108:111], expected_relative, rtol=0, atol=1e-7)
    assert not np.array_equal(initial[108:111], actual[108:111])


def test_b0_reset_step_never_calls_risk_filter_or_future_predictor(
    project_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行时异常注入，不以文本搜索代替执行边界证明。"""

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError('disabled B0 risk/filter/future path called')

    for name in (
        'auv_risk_rl.env.local_navigation.validate_nominal_action',
        'auv_risk_rl.safety.validator.validate_nominal_action',
        'auv_risk_rl.safety.validator.evaluate_candidate_action',
        'auv_risk_rl.safety.validator.build_candidate_actions',
        'auv_risk_rl.safety.candidates.build_candidate_actions',
        'auv_risk_rl.risk.gaussian_bounds.segment_collision_upper_bound',
        'auv_risk_rl.prediction.predictor.predict_position_distribution',
        'auv_risk_rl.env.observation.predict_position_distribution',
    ):
        monkeypatch.setattr(name, forbidden)
    obstacle = GroundTruthObstacleState(1, np.array([30., 50., 20.]),
                                        np.array([-.4, 0., 0.]), .5)
    env = make_env(project_config, obstacles=(obstacle,))
    actual, info = env.reset(seed=900003)
    assert actual[128] == 1 and info['warmup_duration_s'] == 1
    actual, _, _, _, info = env.step(np.zeros(3))
    assert np.all(np.isfinite(actual))
    assert info['unclipped_risk_U'] is None and info['cost_components']['c_risk'] is None
    assert info['intervention'] is None and info['fallback'] is None
    assert not info['risk_training'] and not info['safety_validation']


def test_b0_nominal_equals_executed_with_real_actuator_response(
    project_config: ProjectConfig,
) -> None:
    """指令不经筛选；真实surge仍由共同一阶响应/变化率模型推进。"""

    env = make_env(project_config)
    initial, _ = env.reset(seed=900004)
    action = np.array([1., .3, -.2])
    command = normalized_to_command(action, project_config.dynamics)
    reference = AUVWorld(project_config, env.world.auv_state, (), env.goal_position_ned_m, 1.)
    expected = reference.step(command)
    observation, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_array_equal(env.world.auv_state.position_ned_m,
                                  expected.auv_state.position_ned_m)
    np.testing.assert_array_equal(info['nominal_action_normalized'], action)
    np.testing.assert_array_equal(info['executed_action_normalized'], action)
    assert .3 < env.world.auv_state.surge_speed_mps < command.surge_speed_command_mps
    np.testing.assert_allclose(observation[13:16], action, rtol=0, atol=1e-7)
    assert reward == pytest.approx(sum(info['reward_components'].values()))
    difference = action-initial[13:16]
    assert info['reward_components']['smoothness'] == pytest.approx(-.02*(difference@difference))
    assert not terminated and not truncated and info['cost'] == 0


@pytest.mark.parametrize('event', ['success', 'collision', 'boundary'])
def test_b0_real_events_are_not_filtered_or_hidden(
    project_config: ProjectConfig, event: str,
) -> None:
    """真实最早事件仍发生；0.3m验证裕量不替代物理碰撞，终止后不再推进。"""

    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=1.))
    goal = np.array([22.34, 50., 20.]) if event == 'success' else None
    obstacles = (() if event != 'collision' else (
        GroundTruthObstacleState(1, np.array([22.09, 50., 20.]), np.zeros(3), 1.),))
    state = (AUVState(np.array([98.91, 50., 20.]), 0., 0., .3, 0., 0.)
             if event == 'boundary' else None)
    env = make_env(config, state=state, obstacles=obstacles, goal=goal)
    env.reset(seed=900005)
    _, _, terminated, truncated, info = env.step(np.array([-1., 0., 0.]))
    names = {'success': 'goal_success', 'collision': 'collision',
             'boundary': 'operational_boundary_failure'}
    assert terminated and not truncated and info['failure_type'] == names[event]
    assert 0 < info['elapsed_s'] < .2
    assert info['cost'] == int(event != 'success')
    assert info['cost_components']['c_risk'] is None
    timestamp = env.world.timestamp_s
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))
    assert env.world.timestamp_s == timestamp


def test_b0_external_cutoff_and_true_task_timeout_are_distinct(
    project_config: ProjectConfig,
) -> None:
    """外部工程停止不是物理timeout；10步短时域仅为单测夹具。"""

    env = make_env(project_config)
    env.reset(seed=900006, options={'external_max_steps': 1})
    _, _, terminated, truncated, info = env.step(np.array([-1., 0., 0.]))
    assert not terminated and truncated and info['failure_type'] == 'external_truncation'
    config = replace(project_config, environment=replace(
        project_config.environment, max_episode_control_steps=10,
    ))
    env = make_env(config)
    env.reset(seed=900007)
    for _ in range(10):
        observation, _, terminated, truncated, info = env.step(np.array([-1., 0., 0.]))
    assert terminated and not truncated and info['failure_type'] == 'task_horizon'
    assert info['task_control_step'] == 10 and observation[16] == 0


def test_b0_does_not_disable_original_local_navigation_filter(
    project_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通有限感知/B5环境仍调用原执行验证；B0不全局改写原类。"""

    calls: list[str] = []

    def original_filter(*_: Any, **__: Any) -> Any:
        calls.append('original_filter')
        raise RuntimeError('sentinel original filter')

    monkeypatch.setattr('auv_risk_rl.env.local_navigation.validate_nominal_action', original_filter)
    b0 = make_env(project_config)
    finite = LocalNavigationEnv(project_config, b0._initial_auv, (),
                                b0.goal_position_ned_m, 'finite-interface')
    b0.reset(seed=900008)
    b0.step(np.zeros(3))
    assert not calls
    finite.reset(seed=900008)
    with pytest.raises(RuntimeError, match='sentinel original filter'):
        finite.step(np.zeros(3))
    assert calls == ['original_filter']


@pytest.mark.parametrize('action', [np.array([1.001, 0., 0.]), np.array([np.nan, 0., 0.])])
def test_b0_invalid_action_is_rejected_without_world_advance(
    project_config: ProjectConfig, action: np.ndarray,
) -> None:
    """不clip越界动作，不以非法reset或推进隐藏输入错误。"""

    env = make_env(project_config)
    env.reset(seed=900009)
    timestamp = env.world.timestamp_s
    with pytest.raises(ValueError):
        env.step(action)
    assert env.world.timestamp_s == timestamp and env.world.control_step_index == 0
