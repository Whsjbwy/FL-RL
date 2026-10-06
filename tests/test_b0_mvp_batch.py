"""六job批次发布/恢复的纯调度模拟；合成跳计数不是环境采样或科研训练。"""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.training import curriculum, fixed_validation, mvp_batch
from auv_risk_rl.training.mvp_registration import APPROVED


class MockActor:
    """单CPU标量作模型身份哨兵，没有前向、梯度或优化器。"""

    def __init__(self, seed: int) -> None:
        """不同seed初始身份独立，用于mock恢复对照。"""
        self.value = seed

    def state_dict(self) -> dict[str, torch.Tensor]:
        """真实写小型模型产物，但不称为学习到的Actor。"""
        return {'fixture_only': torch.tensor([self.value], dtype=torch.float32)}


class MockReplay:
    """只模拟有效长度/游标，不分配经验数组。"""

    def __init__(self) -> None:
        """初始无mock transition。"""
        self.length = 0

    def __len__(self) -> int:
        """返回夹具中的合成长度。"""
        return self.length


class MockFixedPool:
    """假验证只核对批次回调，不生成300真实episode。"""

    def __init__(self, project: Any, scenario: Any, root_seed: int) -> None:
        """检查固定root与真实配置已传入，不执行环境。"""
        assert root_seed == 20261006

    def compact_manifest(self) -> dict[str, Any]:
        """fixture_only明确区分真实Val300清单。"""
        return dict(fixture_only=True, root_seed=20261006)

    def evaluate(self, harness: Any, profile: str, full: bool,
                 progress_callback: Any = None) -> dict[str, Any]:
        """回调只返回模拟规模，不产生环境transition。"""
        return dict(fixture_only=True, profile=profile, full=full, episodes=[],
                    count=300 if full else 30, evaluation_env_transitions=0,
                    warmup_control_transitions=0, at_transition=harness.transitions)


class MockCurriculumHarness:
    """一次step跳到下一25k哨兵，测试时间线，绝不实际执行25k步。"""

    trace: list[tuple[Any, ...]] = []
    failure_seed: int | None = None
    failure_transition: int | None = None

    def __init__(self, config: Any, project: Any, scenario: Any, *, code_version: str,
                 evaluation_callback: Any) -> None:
        """保留独立初始化seed、mock Replay和所有调度状态。"""
        self.config, self.code_version = config, code_version
        self.evaluation_callback = evaluation_callback
        self.agent = SimpleNamespace(config=config.sac, actor=MockActor(config.training_seed),
                                     replay=MockReplay(), counters=dict(
                                         environment_steps=0, gradient_updates=0,
                                         actor=0, q1=0, q2=0, alpha=0))
        self.transitions = self.log_sequence = self.evaluation_env_transitions = 0
        self.evaluation_warmup_control_transitions = 0
        self.started_episodes = self.completed_episodes = 0
        self.current_profile = 'obstacle_free'
        self.validation_keys: list[str] = []
        self.log_sink = None
        self.failure_metadata = None
        self.trace.append(('initialize', config.training_seed, config.sac.initialization_seed))

    @property
    def phase_transition_counts(self) -> dict[str, int]:
        """仅mock课程计数接口，不代表实际环境采样。"""
        return dict(obstacle_free=min(self.transitions, 100000),
                    cv_train_v1=max(0, self.transitions - 100000))

    def _emit(self, kind: str, record: dict[str, Any]) -> None:
        """写合成调度事件，每行显式fixture_only，不能当科研证据。"""
        self.log_sequence += 1
        record.update(log_sequence=self.log_sequence, fixture_only=True,
                      code_version=self.code_version, run_kind=self.config.run_kind,
                      method=self.config.method)
        if self.log_sink is not None:
            self.log_sink(kind, record)

    def ensure_validation(self, profile: str, full: bool) -> None:
        """模拟课程幂等键与固定验证回调。"""
        key = f'{profile}:{self.transitions}:{full}'
        if key not in self.validation_keys:
            record = self.evaluation_callback(self, profile, full)
            self._emit('validation', record)
            self.validation_keys.append(key)

    def step(self) -> dict[str, bool]:
        """预先声明的mock计数，不调用env.step或Agent.update。"""
        transition = self.transitions + 25000
        assert transition <= (100000 if self.current_profile == 'obstacle_free' else 300000)
        if self.config.training_seed == self.failure_seed and transition == self.failure_transition:
            self.failure_metadata = dict(fixture_only=True, error='injected partial-step failure')
            self.trace.append(('failed_step', self.config.training_seed, transition))
            raise RuntimeError('injected partial-step failure')
        self.transitions = transition
        updates = max(0, transition - 9999)
        self.agent.counters.update(environment_steps=transition, gradient_updates=updates,
                                   actor=updates, q1=updates, q2=updates, alpha=updates)
        self.agent.replay.length = transition
        self.agent.actor.value = self.config.training_seed + transition
        self._emit('update', dict(transition=transition, task_profile=self.current_profile))
        endpoint = 100000 if self.current_profile == 'obstacle_free' else 300000
        self.ensure_validation(self.current_profile, transition == endpoint)
        self.trace.append(('mock_step', self.config.training_seed, transition))
        return {'checkpoint_due': True}

    def switch_to_cv(self) -> None:
        """固定100k边界，模型/Replay/counters不重新初始化。"""
        assert self.transitions == 100000 and self.current_profile == 'obstacle_free'
        self.current_profile = 'cv_train_v1'
        self.trace.append(('switch', self.config.training_seed, self.transitions))

    def record_final_budget_stop(self) -> None:
        """只记录最终边界，绝不追加第300001步。"""
        assert self.transitions == 300000 and self.current_profile == 'cv_train_v1'
        self.trace.append(('final_stop', self.config.training_seed, self.transitions))

    def _state(self) -> dict[str, Any]:
        """JSON mock恢复状态，包含模型/配置/计数/课程/验证发行位置。"""
        return dict(fixture_only=True, config=asdict(self.config), code_version=self.code_version,
                    transitions=self.transitions, counters=self.agent.counters.copy(),
                    actor_value=self.agent.actor.value, replay_length=len(self.agent.replay),
                    log_sequence=self.log_sequence, profile=self.current_profile,
                    validation_keys=self.validation_keys.copy())

    def save_checkpoint(self, path: Path) -> None:
        """写可信本机的轻量mock JSON，不保存Replay数组或真正模型续点。"""
        assert self.failure_metadata is None
        mvp_batch.atomic_json(path, self._state())
        self.trace.append(('checkpoint', self.config.training_seed, self.transitions,
                           self.current_profile, self.log_sequence))

    def load_checkpoint(self, path: Path, *, trusted_local: bool) -> None:
        """恢复必须同seed、代码、配置；不得将另一seed作为本seed初态。"""
        assert trusted_local
        state = json.loads(path.read_text(encoding='utf-8'))
        assert state['config'] == asdict(self.config) and state['code_version'] == self.code_version
        self.transitions, self.log_sequence = state['transitions'], state['log_sequence']
        self.current_profile, self.validation_keys = state['profile'], state['validation_keys']
        self.agent.counters = state['counters'].copy()
        self.agent.actor.value = state['actor_value']
        self.agent.replay.length = state['replay_length']
        self.trace.append(('restore', self.config.training_seed, self.transitions,
                           self.current_profile, self.agent.actor.value, len(self.agent.replay)))

    def summary(self) -> dict[str, Any]:
        """仅mock故障计数；fixture_only不允许当成实际科研训练统计。"""
        return dict(fixture_only=True, transitions=self.transitions,
                    updates=self.agent.counters['gradient_updates'],
                    failure_metadata=deepcopy(self.failure_metadata))


@pytest.fixture
def mock_batch(monkeypatch: pytest.MonkeyPatch, project_config: Any) -> None:
    """替换设备、网络和环境入口，保留真实小文件/日志/锁的批次行为。"""
    scenario = load_training_scenario_config(Path(__file__).parents[1]
                                             / 'configs/train_scenario_v1.yaml')
    MockCurriculumHarness.trace = []
    MockCurriculumHarness.failure_seed = MockCurriculumHarness.failure_transition = None
    monkeypatch.setattr(curriculum, 'B0CurriculumHarness', MockCurriculumHarness)
    monkeypatch.setattr(fixed_validation, 'FixedValidationPool', MockFixedPool)
    monkeypatch.setattr(mvp_batch, 'resource_identity', lambda output: {'fixture_only': True})
    monkeypatch.setattr(mvp_batch, 'load_project_config', lambda path: project_config)
    monkeypatch.setattr(mvp_batch, 'load_training_scenario_config', lambda path: scenario)
    monkeypatch.setattr(mvp_batch, 'git', lambda root, *args: 'mock-tested-code')
    monkeypatch.setattr(mvp_batch, 'verify_code_identity', lambda root, commit: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)


def test_mock_six_jobs_independent_initialization_then_each_own_resume(
    mock_batch: None, tmp_path: Path,
) -> None:
    """3seed先各100k再各恢复到300k；所有跳计数均mock，真实训练次数为0。"""
    state = mvp_batch.run_batch(tmp_path, deepcopy(APPROVED))
    assert state['status'] == 'COMPLETED'
    assert state['completed_jobs'] == APPROVED['execution_order']
    assert state['actual_training_transitions'] == 900000
    assert state['actual_sac_updates'] == 870003
    trace = MockCurriculumHarness.trace
    starts = [event for event in trace if event[0] == 'initialize']
    assert [event[1] for event in starts] == [11, 22, 33, 11, 22, 33]
    assert len({event[2] for event in starts[:3]}) == 3
    restores = [event for event in trace if event[0] == 'restore']
    assert restores == [('restore', seed, 100000, 'cv_train_v1', seed + 100000, 100000)
                        for seed in (11, 22, 33)]
    assert [event[1:] for event in trace if event[0] == 'final_stop'] \
        == [(11, 300000), (22, 300000), (33, 300000)]
    output = tmp_path / APPROVED['output_directory']
    assert not (output / 'batch.lock').exists()
    for seed in (11, 22, 33):
        row = state['seeds'][str(seed)]
        assert row['transitions'] == 300000 and row['updates'] == 290001
        assert row['optimizer_steps'] == 1160004 and row['profile'] == 'cv_train_v1'
        metadata = json.loads((output / f'seed_{seed}/segments.json').read_text(encoding='utf-8'))
        assert [segment['status'] for segment in metadata] == ['PAUSED', 'COMPLETED']
        assert all(segment['valid_log_sequence'] > 0 for segment in metadata)
        assert metadata[1]['parent_checkpoint'].endswith('latest_resume.pt')
        assert (output / f'models/seed_{seed}_100000.pt').is_file()
        assert (output / f'models/seed_{seed}_300000.pt').is_file()


def test_mock_completed_resume_does_not_repeat_jobs(mock_batch: None, tmp_path: Path) -> None:
    """COMPLETED恢复只是读取批次状态，不能重新初始化/采样或写模型。"""
    first = mvp_batch.run_batch(tmp_path, deepcopy(APPROVED))
    before = deepcopy(MockCurriculumHarness.trace)
    second = mvp_batch.run_batch(tmp_path, deepcopy(APPROVED), resume=True)
    assert second == first and MockCurriculumHarness.trace == before
    with pytest.raises(RuntimeError, match='已有批次状态'):
        mvp_batch.run_batch(tmp_path, deepcopy(APPROVED))


def test_mock_failure_keeps_only_last_valid_checkpoint_and_no_automatic_retry(
    mock_batch: None, tmp_path: Path,
) -> None:
    """seed22半步失败保留0步有效续点，不伪装25000完成或自动重复故障预算。"""
    MockCurriculumHarness.failure_seed = 22
    MockCurriculumHarness.failure_transition = 25000
    with pytest.raises(RuntimeError, match='injected partial-step failure'):
        mvp_batch.run_batch(tmp_path, deepcopy(APPROVED))
    output = tmp_path / APPROVED['output_directory']
    state = json.loads((output / 'batch_state.json').read_text(encoding='utf-8'))
    assert state['status'] == 'FAILED'
    assert state['completed_jobs'] == [{'seed': 11, 'stop': 100000}]
    assert state['failure_actual_counters']['fixture_only']
    last_valid = json.loads((output / 'seed_22/latest_resume.pt').read_text(encoding='utf-8'))
    assert last_valid['transitions'] == 0
    assert not any(event[:3] == ('checkpoint', 22, 25000)
                   for event in MockCurriculumHarness.trace)
    metadata = json.loads((output / 'seed_22/segments.json').read_text(encoding='utf-8'))
    assert metadata[-1]['status'] == 'FAILED'
    assert metadata[-1]['valid_log_sequence'] == last_valid['log_sequence']
    assert not (output / 'batch.lock').exists()
    before = deepcopy(MockCurriculumHarness.trace)
    with pytest.raises(RuntimeError, match='不得自动重试'):
        mvp_batch.run_batch(tmp_path, deepcopy(APPROVED), resume=True)
    assert MockCurriculumHarness.trace == before


def test_mock_resume_without_state_is_rejected(mock_batch: None, tmp_path: Path) -> None:
    """没有自身批次身份不能伪造resume。"""
    with pytest.raises(RuntimeError, match='没有批次状态'):
        mvp_batch.run_batch(tmp_path, deepcopy(APPROVED), resume=True)


def test_segment_resume_cutoff_excludes_unconfirmed_tail(tmp_path: Path) -> None:
    """保留旧日志尾部但权威prefix停在最后安全checkpoint，恢复另开段。"""
    first = mvp_batch.SegmentLog(tmp_path, 11, parent=None, cutoff=None)
    first('update', {'fixture_only': True, 'log_sequence': 1})
    first.confirm(1, 'RUNNING')
    first('update', {'fixture_only': True, 'log_sequence': 2})
    first.close()
    second = mvp_batch.SegmentLog(tmp_path, 11, parent=tmp_path / 'latest_resume.pt', cutoff=1)
    second('update', {'fixture_only': True, 'log_sequence': 2})
    second.confirm(2, 'COMPLETED')
    second.close()
    metadata = json.loads((tmp_path / 'seed_11/segments.json').read_text(encoding='utf-8'))
    assert metadata[0]['valid_log_sequence'] == 1
    assert metadata[1]['valid_log_sequence'] == 2
    assert len((tmp_path / metadata[0]['path'] / 'update.jsonl')
               .read_text(encoding='utf-8').splitlines()) == 2


def test_active_batch_lock_blocks_second_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实自身PID锁阻止重复启动；不依赖工具超时推断进程已结束。"""
    def forbidden_signal(*_: Any, **__: Any) -> None:
        """Windows只读锁检查绝不能向进程发送CTRL事件或终止信号。"""
        raise AssertionError('Windows lock probe sent a process signal')

    if mvp_batch.sys.platform == 'win32':
        monkeypatch.setattr(mvp_batch.os, 'kill', forbidden_signal)
    lock = mvp_batch.BatchLock(tmp_path / 'batch.lock', resume=False)
    try:
        with pytest.raises(RuntimeError, match='已有批次进程'):
            mvp_batch.BatchLock(tmp_path / 'batch.lock', resume=True)
    finally:
        lock.close()
    assert not (tmp_path / 'batch.lock').exists()


def test_stale_lock_requires_explicit_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """纯存活状态夹具证明只有显式resume才清理本批旧锁，不检查/终止其他进程。"""
    path = tmp_path / 'batch.lock'
    path.write_text(json.dumps({'pid': 987654321, 'fixture_only': True}), encoding='utf-8')
    monkeypatch.setattr(mvp_batch, 'process_alive', lambda pid: False)
    with pytest.raises(RuntimeError, match='过期锁'):
        mvp_batch.BatchLock(path, resume=False)
    assert path.is_file()
    lock = mvp_batch.BatchLock(path, resume=True)
    try:
        assert json.loads(path.read_text(encoding='utf-8'))['pid'] == mvp_batch.os.getpid()
    finally:
        lock.close()


def test_fresh_batch_rejects_orphan_checkpoint_before_any_resume(
    mock_batch: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少自身batch身份时不能把旧续点混入新的独立初始化运行。"""
    output = tmp_path / APPROVED['output_directory']
    directory = output / 'seed_11'
    directory.mkdir(parents=True)
    (directory / 'latest_resume.pt').write_text('fixture-only orphan', encoding='utf-8')

    def forbidden_load(*_: Any, **__: Any) -> None:
        """新批次拒绝须早于反序列化旧checkpoint。"""
        raise AssertionError('fresh batch attempted orphan resume')

    monkeypatch.setattr(MockCurriculumHarness, 'load_checkpoint', forbidden_load)
    with pytest.raises(RuntimeError, match='已有.*产物|孤立|现有.*产物'):
        mvp_batch.run_batch(tmp_path, deepcopy(APPROVED))
    assert (directory / 'latest_resume.pt').read_text(encoding='utf-8') == 'fixture-only orphan'
