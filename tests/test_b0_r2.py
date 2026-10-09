"""R2登记、固定双规模评价及恢复；大步数为显式调度夹具，不运行科研预算。"""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts import run_b0_training

from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.training.harness import states_equal
from auv_risk_rl.training.r2_registration import (
    APPROVED,
    CONFIG,
    DOCUMENT,
    OUTPUT,
    REGISTRATION_ID,
    validate_registration,
)
from auv_risk_rl.training.repair_r2 import (
    B0R2Harness,
    harness_config,
    validate_resume_inventory,
)
from auv_risk_rl.training.scenarios import B0ScenarioSource
from test_b0_harness import FixtureAgent, FixtureEnv


def material(root: Path, audit: bool = True) -> Path:
    """只建立明确标记的登记守卫夹具，不当作真实科学审核。"""
    (root / 'configs').mkdir()
    (root / 'docs').mkdir()
    (root / OUTPUT).mkdir(parents=True)
    (root / CONFIG).write_text(yaml.safe_dump(APPROVED), encoding='utf-8')
    (root / DOCUMENT).write_text(REGISTRATION_ID, encoding='utf-8')
    if audit:
        (root / OUTPUT / 'PRETRAIN_MODEL_REWARD_AUDIT.json').write_text(json.dumps(dict(
            fixture_only=True, registration_id=REGISTRATION_ID,
            confirmed_critical_L1_defects=[], unresolved_scientific_conflicts=[],
            training_permitted=True)), encoding='utf-8')
    return root


def validate(root: Path, **kwargs: Any) -> dict[str, Any]:
    """显式执行参数匹配的接口夹具，无env或网络。"""
    values = dict(run_kind='scientific_training', budget=1800000,
                  output=root / OUTPUT, execute=True)
    values.update(kwargs)
    return validate_registration(root, root / CONFIG, **values)


def mock_evaluation(current: B0R2Harness, profile: str, full: bool) -> dict[str, Any]:
    """模拟案例分母与调度，不产生虚假导航结果。"""
    return dict(fixture_only=True, count=300 if full else 30, profile=profile,
                evaluation_env_transitions=0, warmup_control_transitions=0)


def make_harness(project: Any, *, group: str = 'C300',
                 callback: Any = mock_evaluation) -> B0R2Harness:
    """复用无Actor的已有调度夹具，不消耗科研seed运行。"""
    scenario = load_training_scenario_config(
        Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml')
    config = harness_config(APPROVED, group, 11)
    config = replace(config, sac=replace(config.sac, device='cpu'))
    return B0R2Harness(config, project, scenario, group=group, code_version='r2-mock-only',
                       evaluation_callback=callback, agent=FixtureAgent(config.sac),
                       env_factory=FixtureEnv)


def move_mock_counter(current: B0R2Harness, point: int) -> None:
    """合成过去事件，不冒充真实环境transition/update。"""
    current.transitions = point
    current.agent.counters['environment_steps'] = point
    current.next_validation_transition = (point // 25000 + 1) * 25000
    current.next_checkpoint_transition = (point // 25000 + 1) * 25000
    current.completed_validation_keys = current.required_validation_keys(point)
    current.validation_count = len(current.completed_validation_keys)


def test_default_r2_entry_only_preflights(capsys: Any) -> None:
    """真实配置无execute时不会训练。"""
    root = Path(__file__).parents[1]
    assert run_b0_training.main(['--r2-registration', str(root / CONFIG)]) == 0
    assert 'PREFLIGHT_ONLY' in capsys.readouterr().out


@pytest.mark.parametrize('key,value', [
    ('training_seeds', [11, 22, 44]), ('task_profile', 'cv_train_v1'),
    ('total_transition_budget', 1800001), ('transition_budget_per_run', 500000),
    ('device', 'cpu'), ('method', 'CONSTRAINED_SAC'), ('repair_round', 3),
    ('groups', {'C300': {'w_goal': 100}, 'G200': {'w_goal': 150}}),
])
def test_unapproved_r2_configuration_rejected(tmp_path: Path, key: str, value: Any) -> None:
    """不能用R2开关变相换预算、候选、方法或seed。"""
    root = material(tmp_path)
    raw = deepcopy(APPROVED)
    raw[key] = value
    (root / CONFIG).write_text(yaml.safe_dump(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='授权不一致'):
        validate(root)


@pytest.mark.parametrize('values', [
    {'run_kind': 'engineering_smoke'}, {'budget': 900000}, {'output': Path('wrong')},
])
def test_explicit_execution_identity_required(tmp_path: Path, values: dict[str, Any]) -> None:
    """登记存在不替代显式执行身份。"""
    with pytest.raises(ValueError, match='显式匹配'):
        validate(material(tmp_path), **values)


def test_missing_or_failed_actual_audit_blocks_training(tmp_path: Path) -> None:
    """关键科学错误未排除时不能直接投入六run。"""
    root = material(tmp_path, audit=False)
    with pytest.raises(ValueError, match='真实模型'):
        validate(root)
    audit = dict(registration_id=REGISTRATION_ID, confirmed_critical_L1_defects=['defect'],
                 unresolved_scientific_conflicts=[], training_permitted=True)
    (root / OUTPUT / 'PRETRAIN_MODEL_REWARD_AUDIT.json').write_text(
        json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError, match='关键L1'):
        validate(root)


def test_legal_r2_dispatches_only_mock(tmp_path: Path, monkeypatch: Any) -> None:
    """实际CLI合法放行到唯一批次，不在测试执行科研训练。"""
    from auv_risk_rl.training import repair_r2

    root = material(tmp_path)
    calls = []
    monkeypatch.setattr(run_b0_training, 'ROOT', root)
    monkeypatch.setattr(repair_r2, 'run_r2_batch',
                        lambda *a, **k: calls.append((a, k)) or {'status': 'MOCK_ONLY'})
    assert run_b0_training.main([
        '--r2-registration', str(root / CONFIG), '--execute', '--run-kind',
        'scientific_training', '--transition-budget', '1800000',
        '--output-dir', str(root / OUTPUT)]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize('option', ['--mvp-registration', '--repair-registration', '--resume'])
def test_r2_cannot_mix_old_entry(option: str) -> None:
    """旧模型或课程批次不能冒充新组从零训练。"""
    with pytest.raises(SystemExit) as caught:
        run_b0_training.main(['--r2-registration', CONFIG, option, 'wrong'])
    assert caught.value.code == 2


@pytest.mark.parametrize('seed', [11, 22, 33])
def test_pair_random_streams_identical_and_only_goal_differs(seed: int) -> None:
    """名称不进入RNG；除task.goal外完整配置逐字段一致。"""
    control = harness_config(APPROVED, 'C300', seed)
    candidate = harness_config(APPROVED, 'G200', seed)
    assert control.sac == candidate.sac
    assert replace(candidate, task=control.task) == control
    assert control.sac.learning_rate == 3e-4
    assert control.task.w_goal == 100 and candidate.task.w_goal == 200


@pytest.mark.parametrize('seed', [11, 22, 33])
def test_actual_pair_initial_models_rng_and_scenario_identical(
    project_config: Any, seed: int,
) -> None:
    """真实CPU初始化逐值比较，不采动作、不写Replay、不进行梯度或科研训练。"""
    configs = [harness_config(APPROVED, group, seed) for group in ('C300', 'G200')]
    configs = [replace(config, sac=replace(config.sac, device='cpu')) for config in configs]
    agents = [OrdinarySACAgent(config.sac, source_fingerprint='r2-paired-init-test-only')
              for config in configs]
    left, right = agents
    for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2'):
        assert states_equal(getattr(left, name).state_dict(), getattr(right, name).state_dict())
    assert states_equal(left.log_alpha, right.log_alpha)
    assert states_equal(left.generator.get_state(), right.generator.get_state())
    assert states_equal(left.replay.state_dict(), right.replay.state_dict())
    assert left.counters == right.counters == dict(
        environment_steps=0, gradient_updates=0, episodes=0, actor=0, q1=0, q2=0, alpha=0)
    scenario = load_training_scenario_config(
        Path(__file__).parents[1] / 'configs/train_scenario_v1.yaml')
    sources = [B0ScenarioSource(config, project_config, scenario) for config in configs]
    scenarios = [source.scenario(0) for source in sources]
    assert states_equal(asdict(scenarios[0]), asdict(scenarios[1]))
    assert sources[0].environment_seed(0) == sources[1].environment_seed(0)
    environments = [source.make_env(value)
                    for source, value in zip(sources, scenarios, strict=True)]
    assert environments[0].task_config == configs[0].task
    assert environments[1].task_config == configs[1].task


def test_all_monitor_points_plus_two_additional_val300(project_config: Any) -> None:
    """13次monitor和2次完整Val；100k/300k不能悄悄替代或重复评价。"""
    calls = []

    def callback(current: B0R2Harness, profile: str, full: bool) -> dict[str, Any]:
        """仅记录调度计划。"""
        calls.append((current.transitions, full))
        return mock_evaluation(current, profile, full)

    current = make_harness(project_config, callback=callback)
    for point in current.MONITOR_POINTS:
        current.transitions = point
        current.ensure_validation()
        before = current.state_dict()
        current.ensure_validation()
        assert states_equal(before, current.state_dict())
    assert calls == [(point, full) for point in current.MONITOR_POINTS
                     for full in ((False, True) if point in current.FULL_POINTS else (False,))]
    assert current.validation_count == 15


def test_mock_mid_episode_exact_resume(project_config: Any, tmp_path: Path) -> None:
    """含完整环境/RNG/诊断/组/reward的中途恢复，不执行神经更新。"""
    left = make_harness(project_config, group='G200')
    left.ensure_validation()
    for _ in range(4):
        left.step()
    path = tmp_path / 'trusted_r2_mock.pt'
    left.save_checkpoint(path)
    right = make_harness(project_config, group='G200')
    right.load_checkpoint(path, trusted_local=True)
    for _ in range(4):
        left.step()
        right.step()
    assert states_equal(left.state_dict(), right.state_dict())


@pytest.mark.parametrize('fault', ['group', 'reward', 'future', 'missing', 'duplicate', 'budget'])
def test_corrupted_or_cross_group_resume_rejected_before_mutation(
    project_config: Any, fault: str,
) -> None:
    """先验证身份及评价位置，再触及模型/Replay/RNG。"""
    current = make_harness(project_config)
    current.ensure_validation()
    current.step()
    before = current.state_dict()
    damaged = deepcopy(before)
    repair = damaged['repair']
    if fault == 'group':
        repair['group'] = 'G200'
    elif fault == 'reward':
        repair['reward_config']['w_goal'] = 200
    elif fault == 'future':
        repair['completed_validation_keys'].append('obstacle_free:25000:monitor')
    elif fault == 'missing':
        repair['completed_validation_keys'] = []
    elif fault == 'duplicate':
        repair['completed_validation_keys'] *= 2
    else:
        repair['final_budget_logged'] = True
    with pytest.raises(ValueError, match='R2恢复'):
        current.load_state_dict(damaged)
    assert states_equal(before, current.state_dict())


def test_300k_stops_without_rewriting_terminal_flags(project_config: Any) -> None:
    """最后两步是Mock计数，预算不把未完成片段伪造成timeout/success。"""
    current = make_harness(project_config)
    move_mock_counter(current, 299998)
    current.step()
    current.step()
    current.record_final_budget_stop()
    assert current.transitions == 300000 and len(current.agent.replay) == 2
    assert current.agent.counters['gradient_updates'] == 0
    assert all(not row['complete'] and row['budget_stop'] for row in current.episode_log)
    with pytest.raises(RuntimeError, match='预算'):
        current.step()


def test_resume_existing_run_without_reliable_checkpoint_rejected(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    """掉失恢复点不得隐式重启已有run；在模型构造前拒绝，不运行训练。"""
    from auv_risk_rl.training import repair_r2

    root = material(tmp_path)
    output = root / OUTPUT
    seed_dir = output / 'C300/seed_11'
    seed_dir.mkdir(parents=True)
    (seed_dir / 'segments.json').write_text('[]', encoding='utf-8')
    state = dict(registration_id=REGISTRATION_ID, registration=APPROVED,
                 experiment_code_commit='mock-reviewed', status='RUNNING', completed_runs=[],
                 runs={'C300_seed11': dict(group='C300', seed=11, transitions=25000,
                                           updates=15001, status='RUNNING',
                                           reward_config=APPROVED['groups']['C300'])})
    (output / 'batch_state.json').write_text(json.dumps(state), encoding='utf-8')
    monkeypatch.setattr(repair_r2, 'verify_code_identity', lambda *args: None)
    monkeypatch.setattr(repair_r2, 'resource_identity', lambda *args: None)
    monkeypatch.setattr(repair_r2, 'load_project_config', lambda *args: None)
    monkeypatch.setattr(repair_r2, 'load_training_scenario_config', lambda *args: None)

    class MockPool:
        """无真实环境或随机流的构造前检查夹具。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compact_manifest(self) -> dict[str, Any]:
            return dict(fixture_only=True)

    def forbidden_harness(*args: Any, **kwargs: Any) -> None:
        """若到达此处即证明已有run被错误地从零重建。"""
        raise AssertionError('reached fresh model construction without resume checkpoint')

    monkeypatch.setattr(repair_r2, 'FixedValidationPool', MockPool)
    monkeypatch.setattr(repair_r2, 'B0R2Harness', forbidden_harness)
    with pytest.raises(RuntimeError, match='可靠checkpoint缺失'):
        repair_r2.run_r2_batch(root, APPROVED, resume=True)


@pytest.mark.parametrize('fault', ['completed_missing_row', 'completed_not_done',
                                  'wrong_reward', 'invalid_count', 'incomplete_batch'])
def test_resume_inventory_rejects_false_completion_or_identity(
    tmp_path: Path, fault: str,
) -> None:
    """批次JSON不能绕过真实完整六run；占位文件仅测试清单，不是模型。"""
    seed_dir = tmp_path / 'C300/seed_11'
    seed_dir.mkdir(parents=True)
    (seed_dir / 'latest_resume.pt').write_text('fixture only', encoding='utf-8')
    row = dict(group='C300', seed=11, reward_config=deepcopy(APPROVED['groups']['C300']),
               transitions=300000, updates=290001, checkpoint_transition=300000,
               status='COMPLETED')
    state = dict(status='RUNNING', completed_runs=['C300_seed11'], runs={'C300_seed11': row})
    if fault == 'completed_missing_row':
        state['runs'] = {}
    elif fault == 'completed_not_done':
        row.update(status='RUNNING', transitions=25000, updates=15001)
    elif fault == 'wrong_reward':
        row['reward_config']['w_goal'] = 200
    elif fault == 'invalid_count':
        row['updates'] = 300000
    else:
        state['status'] = 'COMPLETED'
    with pytest.raises(ValueError, match='R2'):
        validate_resume_inventory(tmp_path, state, APPROVED)


def test_resume_inventory_allows_confirmed_prefix_and_unstarted_runs(tmp_path: Path) -> None:
    """可靠已完成前缀可继续剩余run；不需要复制或反序列化大Replay。"""
    seed_dir = tmp_path / 'C300/seed_11'
    seed_dir.mkdir(parents=True)
    (seed_dir / 'latest_resume.pt').write_text('fixture only', encoding='utf-8')
    row = dict(group='C300', seed=11, reward_config=APPROVED['groups']['C300'],
               transitions=300000, updates=290001, checkpoint_transition=300000,
               status='COMPLETED')
    state = dict(status='RUNNING', completed_runs=['C300_seed11'], runs={'C300_seed11': row})
    validate_resume_inventory(tmp_path, state, APPROVED)
