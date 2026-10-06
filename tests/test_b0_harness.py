"""B0 调度、验证隔离和完整恢复；此文件不执行真实环境的 SAC 更新。"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from auv_risk_rl.env.scenario_generator import TrainingScenario, load_training_scenario_config
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.replay import ReplayBuffer
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.harness import B0TrainingHarness, states_equal
from auv_risk_rl.training.scenarios import B0ScenarioSource


class FixtureAgent:
    """只计模拟调度，不创建神经网络或调用 backward 的纯接口夹具。"""

    def __init__(self, config: SACConfig) -> None:
        """创建无神经网络的独立接口状态。"""
        self.config = config
        self.device = torch.device('cpu')
        self.replay = ReplayBuffer(config.replay_capacity, config.replay_seed)
        self.rng = np.random.default_rng(config.actor_seed)
        self.counters = dict(environment_steps=0, gradient_updates=0, episodes=0,
                             actor=0, q1=0, q2=0, alpha=0)

    def sample_action(self, obs: np.ndarray) -> np.ndarray:
        """显式局部随机流，只产生夹具动作。"""
        return self.rng.uniform(-0.2, 0.2, 3).astype(np.float32)

    def deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        """评估动作不消费训练随机流。"""
        return np.zeros(3, np.float32)

    def store_transition(self, **transition: Any) -> None:
        """夹具仍使用真实 Replay 插入接口。"""
        self.replay.add(**transition)
        self.counters['environment_steps'] += 1

    def eligible(self) -> bool:
        """与现有生产起步语义相同，只用于调度断言。"""
        return (self.counters['environment_steps'] >= self.config.learning_starts
                and len(self.replay) >= self.config.batch_size)

    def update(self) -> dict[str, float]:
        """模拟完整更新的计数，不实施任何梯度计算。"""
        self.counters['gradient_updates'] += 1
        return {'mock_update': 1.0}

    def state_dict(self) -> dict[str, Any]:
        """直接保存全部夹具状态。"""
        return deepcopy(dict(counters=self.counters, replay=self.replay.state_dict(),
                             rng=self.rng.bit_generator.state))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复夹具的明确随机流和计数。"""
        self.counters = state['counters'].copy()
        self.replay.load_state_dict(state['replay'])
        self.rng.bit_generator.state = deepcopy(state['rng'])


class FixtureEnv:
    """合成控制节点环境；不是 AUV 动力学或真实验收轨迹。"""

    def __init__(self, scenario: TrainingScenario) -> None:
        """记录合成夹具的发行身份。"""
        self.scenario = scenario

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """独立传感夹具随机流和工程截断。"""
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.limit = options.get('external_max_steps') or 1000
        self.world = SimpleNamespace(auv_state=SimpleNamespace(
            position_ned_m=self.scenario.initial_auv_state.position_ned_m.copy()))
        return np.zeros(234, np.float32), {'warmup_duration_s': 1.0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """合成 next_obs 具哨兵值，证明先存终态后 reset。"""
        self.steps += 1
        self.world.auv_state.position_ned_m += np.array([0.01, 0.0, 0.0])
        next_obs = np.full(234, self.steps + self.rng.uniform(), np.float32)
        truncated = self.steps >= self.limit
        return next_obs, 0.1, False, truncated, dict(
            executed_action_normalized=action.copy(), cost=0.0, elapsed_s=0.2,
            reward_components={'progress': 0.1}, minimum_clearance=float('inf'),
            task_control_step=self.steps,
            failure_type='external_truncation' if truncated else 'none')


@pytest.fixture
def scenario_config() -> Any:
    """读取原训练分布，不为单测修改采样律。"""
    config_path = Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml'
    return load_training_scenario_config(config_path)


def fixture_harness(project: Any, scenario: Any, **overrides: Any) -> B0TrainingHarness:
    """全部采样和 update 都是明确的调度夹具。"""
    config = replace(B0HarnessConfig(run_kind='engineering_smoke', training_seed=820001,
                                    transition_budget=8), **overrides)
    return B0TrainingHarness(config, project, scenario, code_version='git-unit-fixture',
                             agent=FixtureAgent(config.sac), env_factory=FixtureEnv)


@pytest.mark.parametrize('override', [dict(run_kind='unknown'), dict(transition_budget=None),
                                     dict(num_envs=0), dict(transition_budget=2049),
                                     dict(sac=SACConfig(batch_size=128)),
                                     dict(sac=SACConfig(gamma=0.99))])
def test_config_rejects_invalid_engineering_identity(override: dict[str, Any]) -> None:
    """工程授权不允许改变算法规模或静默预算。"""
    values = dict(run_kind='engineering_smoke', transition_budget=264)
    values.update(override)
    with pytest.raises(ValueError):
        B0HarnessConfig(**values)


def test_production_defaults_and_registration_guard(
    project_config: Any, scenario_config: Any,
) -> None:
    """科研配置可以准备，未登记运行安排时不能采样。"""
    config = B0HarnessConfig(run_kind='scientific_training', transition_budget=300000)
    harness = B0TrainingHarness(config, project_config, scenario_config, code_version='unit',
                                agent=FixtureAgent(config.sac), env_factory=FixtureEnv)
    assert config.sac == SACConfig() and config.num_envs == 2
    with pytest.raises(RuntimeError, match='尚未登记'):
        harness.step()


def test_preflight_cannot_step(project_config: Any, scenario_config: Any) -> None:
    """默认构造不会启动训练。"""
    config = B0HarnessConfig()
    harness = B0TrainingHarness(config, project_config, scenario_config, code_version='unit',
                                agent=FixtureAgent(config.sac), env_factory=FixtureEnv)
    with pytest.raises(RuntimeError, match='preflight'):
        harness.step()


def test_explicit_sac_seed_derivation_isolated() -> None:
    """MVP 不同 seed 和工程命名空间有独立 RL 流，同身份可复现。"""
    first = derived_sac_config(SACConfig(), training_seed=11, run_kind='scientific_training')
    repeated = derived_sac_config(SACConfig(), training_seed=11, run_kind='scientific_training')
    second = derived_sac_config(SACConfig(), training_seed=22, run_kind='scientific_training')
    engineering = derived_sac_config(SACConfig(), training_seed=11, run_kind='engineering_smoke')
    assert first == repeated
    for name in ('initialization_seed', 'actor_seed', 'replay_seed'):
        assert len({getattr(value, name) for value in (first, second, engineering)}) == 3
    assert first.gamma == SACConfig().gamma


def test_scenario_profiles_and_split_isolation(project_config: Any, scenario_config: Any) -> None:
    """无障碍保持起终点，CV 保留原样；验证不占训练发行索引。"""
    cfg = B0HarnessConfig(run_kind='engineering_smoke', transition_budget=4)
    empty = B0ScenarioSource(cfg, project_config, scenario_config)
    cv = B0ScenarioSource(replace(cfg, task_profile='cv_train_v1'), project_config, scenario_config)
    original, clean = cv.scenario(0), empty.scenario(0)
    assert 1 <= len(original.initial_obstacle_states) <= 4 and not clean.initial_obstacle_states
    np.testing.assert_array_equal(clean.initial_auv_state.position_ned_m,
                                  original.initial_auv_state.position_ned_m)
    np.testing.assert_array_equal(clean.goal_position_ned_m, original.goal_position_ned_m)
    assert clean.distribution_version == 'B0_OBSTACLE_FREE_V1'
    assert not clean.scenario_id.startswith('train-v1')
    assert empty.split_seed('train') != empty.split_seed('validation')
    assert empty.scenario(0).to_dict() == clean.to_dict()


def test_fixed_round_robin_new_episode_scenarios(project_config: Any, scenario_config: Any) -> None:
    """同时 pending reset 时按槽次序发行连续索引，真实预算只加一次。"""
    harness = fixture_harness(project_config, scenario_config, external_max_steps=1)
    records = [harness.step() for _ in range(4)]
    assert [r['slot'] for r in records] == [0, 1, 0, 1]
    assert [r['scenario_index'] for r in harness.episode_log] == [0, 1, 2, 3]
    assert harness.transitions == len(harness.agent.replay) == 4
    assert harness.next_scenario_index == 4


def test_terminal_next_observation_stored_before_reset(
    project_config: Any, scenario_config: Any,
) -> None:
    """下一 episode reset 的零观察不能覆盖终态哨兵。"""
    harness = fixture_harness(project_config, scenario_config, num_envs=1, external_max_steps=1)
    harness.step()
    stored = harness.agent.replay.at(np.array([0]))['next_obs'][0].copy()
    harness.step()
    np.testing.assert_array_equal(harness.agent.replay.at(np.array([0]))['next_obs'][0], stored)
    assert np.all(stored > 1) and bool(harness.agent.replay.at(np.array([0]))['truncated'][0])


def test_learning_starts_9999_10000_10001_synthetic(
    project_config: Any, scenario_config: Any,
) -> None:
    """合成预填计数夹具；不实际跑 10000 步或任何神经更新。"""
    config = B0HarnessConfig(run_kind='scientific_training', transition_budget=10001,
                             research_registration='synthetic-interface-fixture', num_envs=1)
    agent = FixtureAgent(config.sac)
    for _ in range(256):
        agent.replay.add(obs=np.zeros(234, np.float32), next_obs=np.ones(234, np.float32),
                         nominal_action=np.zeros(3), executed_action=np.zeros(3), reward=0.0,
                         cost=0.0, terminated=False, truncated=False)
    agent.counters['environment_steps'] = 9998
    harness = B0TrainingHarness(config, project_config, scenario_config, code_version='fixture',
                                agent=agent, env_factory=FixtureEnv)
    harness.transitions = 9998
    metrics = [harness.step()['metrics'] for _ in range(3)]
    assert metrics[0] == {} and metrics[1] and metrics[2]
    assert agent.counters['gradient_updates'] == 2


def test_independent_validation_does_not_change_training(
    project_config: Any, scenario_config: Any,
) -> None:
    """直接比较 Agent/Replay/环境/观察/索引和显式 RNG，没有梯度判断替代。"""
    harness = fixture_harness(project_config, scenario_config, validation_max_steps=2)
    harness.step()
    before = harness._training_state()
    result = harness.evaluate()
    assert result['training_state_unchanged']
    assert states_equal(before, harness._training_state())
    assert harness.validation_scenario_index == 1 and harness.next_scenario_index == 1


def test_validation_detects_hidden_training_rng_mutation(
    project_config: Any, scenario_config: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """故意消费训练策略流的错误评估必须被直接状态比较拒绝。"""
    harness = fixture_harness(project_config, scenario_config, validation_max_steps=1)
    monkeypatch.setattr(harness.agent, 'deterministic_action', harness.agent.sample_action)
    with pytest.raises(RuntimeError, match='改变了训练状态'):
        harness.evaluate()
    with pytest.raises(RuntimeError, match='无失败'):
        harness.state_dict()


def test_checkpoint_callback_cadence(project_config: Any, scenario_config: Any) -> None:
    """调度回调只请求最近恢复点，不默认复制多个完整 Replay。"""
    harness = fixture_harness(project_config, scenario_config, transition_budget=5,
                              checkpoint_interval=2)
    requests = []
    harness.run(checkpoint_callback=lambda current: requests.append(current.transitions))
    assert requests == [2, 4] and harness.next_checkpoint_transition == 6


def test_mid_update_failure_cannot_resume_or_save(
    project_config: Any, scenario_config: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """半次更新故障不能冒充一致控制步 checkpoint，也不能继续采样。"""
    harness = fixture_harness(project_config, scenario_config)
    monkeypatch.setattr(harness.agent, 'eligible', lambda: True)

    def failed_update() -> dict[str, float]:
        """模拟损失异常，不实施梯度更新。"""
        raise FloatingPointError('synthetic failed update')

    monkeypatch.setattr(harness.agent, 'update', failed_update)
    with pytest.raises(FloatingPointError):
        harness.step()
    assert harness.transitions == 1 and harness.failure_metadata is not None
    with pytest.raises(RuntimeError, match='无失败'):
        harness.state_dict()
    with pytest.raises(RuntimeError, match='已有失败'):
        harness.step()


def test_validation_and_save_triggers_restore(project_config: Any, scenario_config: Any) -> None:
    """25000 边界使用纯夹具跳转计数，不实际长跑。"""
    config = B0HarnessConfig(run_kind='scientific_training', transition_budget=25001,
                             research_registration='synthetic-trigger-fixture',
                             validation_max_steps=1)
    agent = FixtureAgent(config.sac)
    harness = B0TrainingHarness(config, project_config, scenario_config, code_version='fixture',
                                agent=agent, env_factory=FixtureEnv)
    harness.transitions = 24999
    result = harness.step()
    assert result['validation'] and result['checkpoint_due']
    state = harness.state_dict()
    restored = B0TrainingHarness(config, project_config, scenario_config, code_version='fixture',
                                 agent=FixtureAgent(config.sac), env_factory=FixtureEnv)
    restored.load_state_dict(state)
    assert restored.next_validation_transition == restored.next_checkpoint_transition == 50000
    assert restored.step()['validation'] is None


@pytest.mark.parametrize('external_limit', [None, 1])
def test_complete_resume_mid_episode_or_pending_reset(
    project_config: Any, scenario_config: Any, tmp_path: Path, external_limit: int | None,
) -> None:
    """恢复中途状态和 episode 结束等待 reset，多环境发行/随机流精确继续。"""
    left = fixture_harness(project_config, scenario_config, external_max_steps=external_limit,
                           validation_interval=3, validation_max_steps=1)
    for _ in range(2):
        left.step()
    checkpoint = tmp_path / 'trusted_local.pt'
    left.save_checkpoint(checkpoint)
    right = fixture_harness(project_config, scenario_config, external_max_steps=external_limit,
                            validation_interval=3, validation_max_steps=1)
    right.load_checkpoint(checkpoint, trusted_local=True)
    for _ in range(4):
        left.step()
        right.step()
    assert states_equal(left.state_dict(), right.state_dict())
    assert left.validation_count == right.validation_count == 2
    assert not checkpoint.with_name(checkpoint.name + '.partial').exists()


def test_untrusted_or_wrong_identity_checkpoint_rejected(
    project_config: Any, scenario_config: Any, tmp_path: Path,
) -> None:
    """来源未确认不反序列化，工程快照不能混为科研续点。"""
    harness = fixture_harness(project_config, scenario_config)
    with pytest.raises(ValueError, match='来源未确认'):
        harness.load_checkpoint(tmp_path / 'unknown.pt')
    state = harness.state_dict()
    state['config']['run_kind'] = 'scientific_training'
    with pytest.raises(ValueError, match='身份'):
        harness.load_state_dict(state)
    state = harness.state_dict()
    state['torch_version'] = 'other-version'
    with pytest.raises(ValueError, match='Torch'):
        harness.load_state_dict(state)


def test_budget_stop_keeps_unfinished_episode(project_config: Any, scenario_config: Any) -> None:
    """全局预算停止不是成功、真实 timeout 或外部截断。"""
    harness = fixture_harness(project_config, scenario_config, transition_budget=3)
    summary = harness.run()
    assert summary['transitions'] == 3 and summary['updates'] == 0
    assert len(harness.budget_stop_records) == 2
    assert all(not r['complete'] and r['budget_stop'] and not r['task_timeout']
               for r in harness.budget_stop_records)
    assert all(not slot['needs_reset'] for slot in harness.slots)
    with pytest.raises(RuntimeError, match='预算'):
        harness.step()


def test_real_b0_environment_resume_without_gradient(
    project_config: Any, scenario_config: Any, tmp_path: Path,
) -> None:
    """真实 B0 仅 3+2 次非学习转移，起步为10000，不执行 SAC 更新。"""
    config = B0HarnessConfig(run_kind='engineering_smoke', training_seed=820008,
                             transition_budget=8)
    left = B0TrainingHarness(config, project_config, scenario_config, code_version='unit-real-env')
    left.step()
    path = tmp_path / 'real_env.pt'
    left.save_checkpoint(path)
    right = B0TrainingHarness(config, project_config, scenario_config, code_version='unit-real-env')
    right.load_checkpoint(path, trusted_local=True)
    for _ in range(2):
        left.step()
        right.step()
    assert left.agent.counters['gradient_updates'] == right.agent.counters['gradient_updates'] == 0
    assert states_equal(left.state_dict(), right.state_dict())
    assert isinstance(left.agent, OrdinarySACAgent)
