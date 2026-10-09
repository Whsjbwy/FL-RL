"""R2单因素奖励注入、物理不变与恢复身份；不执行SAC梯度或科研训练。"""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.local_task import LocalTaskConfig
from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.fixed_validation import FixedValidationPool
from auv_risk_rl.training.harness import B0TrainingHarness, states_equal
from auv_risk_rl.training.scenarios import B0ScenarioSource
from auv_risk_rl.types import AUVState, GroundTruthObstacleState
from test_b0_fixed_validation import ValidationHarnessFixture
from test_b0_harness import FixtureAgent, FixtureEnv

R2_ID = 'STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET'


@pytest.fixture(scope='module')
def scenario_config() -> Any:
    """读取冻结训练分布，不修改原1–4障碍生成器。"""
    path = Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml'
    return load_training_scenario_config(path)


def r2_config(w_goal: float = 100.0, **overrides: Any) -> B0HarnessConfig:
    """只建立独立R2登记配置，不启动实际科研run。"""
    values = dict(run_kind='scientific_training', task_profile='obstacle_free',
                  training_seed=11, transition_budget=300000, num_envs=2,
                  validation_episodes=30, research_registration=R2_ID,
                  task=LocalTaskConfig(w_goal=w_goal))
    values.update(overrides)
    return B0HarnessConfig(**values)


def mock_harness(project: ProjectConfig, scenario: Any, config: B0HarnessConfig,
                 ) -> B0TrainingHarness:
    """纯Agent/Env夹具用于恢复检查，不产生环境推进或梯度。"""
    return B0TrainingHarness(config, project, scenario, code_version='git-r2-unit-fixture',
                             agent=FixtureAgent(config.sac), env_factory=FixtureEnv)


def test_r2_preserves_global_default_and_original_r1_exception() -> None:
    """goal200不改变LocalTaskConfig默认，也不削弱R1唯一lr1e-4登记。"""
    assert LocalTaskConfig() == LocalTaskConfig(w_goal=100.0)
    assert B0HarnessConfig().task == LocalTaskConfig()
    assert r2_config(200.0).task.w_goal == 200.0
    r1 = B0HarnessConfig(
        run_kind='scientific_training', task_profile='obstacle_free', training_seed=11,
        transition_budget=100000, validation_episodes=30,
        research_registration='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
        sac=SACConfig(learning_rate=1e-4))
    assert r1.task == LocalTaskConfig() and r1.sac.learning_rate == 1e-4


@pytest.mark.parametrize('overrides', [
    dict(research_registration='arbitrary'), dict(task_profile='cv_train_v1'),
    dict(training_seed=44), dict(transition_budget=100000), dict(num_envs=1),
    dict(sac=SACConfig(learning_rate=1e-4)), dict(task=LocalTaskConfig(w_goal=201.0)),
    dict(task=LocalTaskConfig(w_goal=200.0, w_progress=2.0)),
    dict(task=LocalTaskConfig(w_goal=200.0, w_time=.02)),
    dict(task=LocalTaskConfig(w_goal=200.0, w_smooth=.03)),
    dict(task=LocalTaskConfig(w_goal=200.0, d_C=.1)),
])
def test_nonregistered_reward_or_second_factor_is_rejected(overrides: dict[str, Any]) -> None:
    """不把R2单因素许可变成任意奖励、学习率或预算改写入口。"""
    with pytest.raises(ValueError):
        r2_config(200.0, **overrides)


def test_paired_reward_configs_do_not_change_explicit_random_streams(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """组名及奖励不进入随机派生；当前几何、环境seed和SAC三流精确配对。"""
    control, goal200 = r2_config(), r2_config(200.0)
    assert control.sac == goal200.sac
    assert derived_sac_config(control.sac, training_seed=11, run_kind=control.run_kind) \
        == derived_sac_config(goal200.sac, training_seed=11, run_kind=goal200.run_kind)
    left = B0ScenarioSource(control, project_config, scenario_config)
    right = B0ScenarioSource(goal200, project_config, scenario_config)
    for split in ('train', 'validation'):
        for index in (0, 1, 7):
            assert left.split_seed(split) == right.split_seed(split)
            assert left.environment_seed(index, split) == right.environment_seed(index, split)
            assert states_equal(left.scenario(index, split), right.scenario(index, split))


def test_environment_factory_receives_registered_reward_before_reset(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """训练和固定验证共同调用的工厂显式注入任务，不执行reset/warm-up。"""
    config = r2_config(200.0)
    source = B0ScenarioSource(config, project_config, scenario_config)
    harness = B0TrainingHarness(config, project_config, scenario_config,
                                 code_version='git-r2-unit-fixture',
                                 agent=FixtureAgent(config.sac), env_factory=source.make_env)
    pool = FixedValidationPool(project_config, scenario_config)
    for scenario in (source.scenario(0), pool.scenario('obstacle_free', 0)):
        env = harness._make_env(scenario)
        assert type(env) is B0NavigationEnv and env.task_config == config.task


def test_wrong_environment_factory_reward_is_rejected_before_reset(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """自定义工厂不能把默认100环境接进登记200运行后静默通过。"""
    config = r2_config(200.0)
    wrong_source = B0ScenarioSource(r2_config(), project_config, scenario_config)
    harness = B0TrainingHarness(config, project_config, scenario_config,
                                 code_version='git-r2-unit-fixture',
                                 agent=FixtureAgent(config.sac), env_factory=wrong_source.make_env)
    with pytest.raises(ValueError, match='奖励'):
        harness._make_env(wrong_source.scenario(0))


@pytest.mark.parametrize('event', ['goal_success', 'collision', 'boundary', 'timeout', 'none'])
def test_goal_weight_changes_only_success_component_on_identical_physics(
    project_config: ProjectConfig, event: str,
) -> None:
    """预登记五个短真实物理反例：只在真实成功终点增加100，不修改另三项或事件。"""
    config = project_config
    if event == 'timeout':
        # 独立短时域夹具须覆盖5个暖机控制步；不修改生产1000步任务时域。
        config = replace(config, environment=replace(
            config.environment, max_episode_control_steps=10))
    initial = AUVState(np.array([98.91 if event == 'boundary' else 20., 50., 20.]),
                       0., 0., .3, 0., 0.)
    goal = np.array([22.34 if event == 'goal_success' else 80., 50., 20.])
    obstacles = (() if event != 'collision' else (
        GroundTruthObstacleState(1, np.array([22.09, 50., 20.]), np.zeros(3), 1.),))
    pair = [B0NavigationEnv(config, initial, obstacles, goal, 'r2-reward-physical-fixture',
                            task_config=LocalTaskConfig(w_goal=weight))
            for weight in (100.0, 200.0)]
    first, second = (env.reset(seed=870091)[0] for env in pair)
    np.testing.assert_array_equal(first, second)
    for _ in range(10 if event == 'timeout' else 1):
        results = [env.step(np.array([-1., 0., 0.])) for env in pair]
    left_obs, left_reward, left_terminal, left_truncated, left_info = results[0]
    right_obs, right_reward, right_terminal, right_truncated, right_info = results[1]
    assert states_equal(pair[0].world, pair[1].world)
    np.testing.assert_array_equal(left_obs, right_obs)
    assert (left_terminal, left_truncated) == (right_terminal, right_truncated)
    assert left_info['failure_type'] == right_info['failure_type']
    expected_event = {'boundary': 'operational_boundary_failure', 'timeout': 'task_horizon'}
    assert left_info['failure_type'] == expected_event.get(event, event)
    for component in ('progress', 'time', 'smoothness'):
        assert left_info['reward_components'][component] \
            == right_info['reward_components'][component]
    expected = 100.0 if event == 'goal_success' else 0.0
    assert right_reward-left_reward == pytest.approx(expected, rel=0.0, abs=1e-12)
    delta_goal = right_info['reward_components']['goal']-left_info['reward_components']['goal']
    assert delta_goal == expected
    assert (left_info['failure_type'] == 'goal_success') == (event == 'goal_success')
    left_physical = {key: value for key, value in left_info.items() if key != 'reward_components'}
    right_physical = {key: value for key, value in right_info.items() if key != 'reward_components'}
    assert states_equal(left_physical, right_physical)


def test_checkpoint_wrong_reward_rejected_and_legacy_default_normalized(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """旧缺task的默认续点补100；不可加载成200，也不可删除新200身份来伪装默认。"""
    control = mock_harness(project_config, scenario_config, r2_config())
    goal200 = mock_harness(project_config, scenario_config, r2_config(200.0))
    original = control.state_dict()
    legacy = deepcopy(original)
    del legacy['config']['task']
    control.load_state_dict(legacy)
    assert states_equal(control.state_dict(), original)
    before = goal200.state_dict()
    for wrong in (original, legacy):
        with pytest.raises(ValueError, match='配置'):
            goal200.load_state_dict(wrong)
        assert states_equal(before, goal200.state_dict())
    saved200 = goal200.state_dict()
    del saved200['config']['task']
    with pytest.raises(ValueError, match='配置'):
        goal200.load_state_dict(saved200)


def test_reward_identity_is_present_in_episode_and_fixed_validation_logs(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """仅合成日志夹具；分别记录200实际回报身份，不让统计把两组奖励混淆。"""
    config = r2_config(200.0)
    harness = mock_harness(project_config, scenario_config, config)
    harness.step()
    record = harness._episode_record(harness.slots[0], complete=False, failure_type='none')
    assert record['task_config'] == record['config']['task'] == asdict(config.task)
    fixture = ValidationHarnessFixture(project_config)
    fixture.config.task = config.task
    result = FixedValidationPool(project_config, scenario_config).evaluate(fixture, 'obstacle_free')
    assert result['task_config'] == asdict(config.task)
    assert all(row['task_config'] == asdict(config.task) for row in result['episodes'])


def test_checkpoint_environment_reward_mismatch_is_rejected_before_loading(
    project_config: ProjectConfig, scenario_config: Any,
) -> None:
    """即使外层config写200，内嵌环境仍为100也不能作为200恢复点。"""
    config = r2_config(200.0)
    harness = mock_harness(project_config, scenario_config, config)
    before = harness.state_dict()
    state = deepcopy(before)
    source = B0ScenarioSource(r2_config(), project_config, scenario_config)
    # 只构造环境对象作身份反例，不reset、暖机或推进物理世界。
    state['scheduler']['slots'][0]['env'] = source.make_env(source.scenario(0))
    with pytest.raises(ValueError, match='环境奖励'):
        harness.load_state_dict(state)
    assert states_equal(before, harness.state_dict())
