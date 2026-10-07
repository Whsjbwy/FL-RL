"""R1唯一候选与科研入口身份守卫；模拟调度不产生训练数据。"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from scripts import run_b0_training

from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig
from auv_risk_rl.training.mvp_batch import SegmentLog
from auv_risk_rl.training.repair_registration import (
    APPROVED,
    CONFIG,
    DOCUMENT,
    OUTPUT,
    REGISTRATION_ID,
    validate_registration,
)


def material(root: Path, *, decision: bool = True) -> Path:
    """只写小型显式测试夹具，不能当作真实诊断/学习结果。"""
    import json

    (root / 'configs').mkdir()
    (root / 'docs').mkdir()
    (root / OUTPUT).mkdir(parents=True)
    (root / CONFIG).write_text(yaml.safe_dump(APPROVED), encoding='utf-8')
    (root / DOCUMENT).write_text(REGISTRATION_ID, encoding='utf-8')
    if decision:
        (root / APPROVED['branch_decision_file']).write_text(json.dumps(dict(
            fixture_only=True, registration_id=REGISTRATION_ID,
            selected_branch='L2_COMMON_LEARNING_RATE', confirmed_L1_defects=[],
            unresolved_key_error_or_protocol_conflict=False)), encoding='utf-8')
        (root / APPROVED['comparability_file']).write_text(json.dumps(dict(
            fixture_only=True, control_reuse_eligible=True,
            control_experiment_commit=APPROVED['control_experiment_commit'],
            only_scientific_parameter_changed='learning_rate', seeds=[11, 22, 33])),
            encoding='utf-8')
    return root


def validate(root: Path, *, execute: bool = True) -> dict:
    """模拟显式匹配命令参数，不运行worker。"""
    return validate_registration(root, root / CONFIG, run_kind='scientific_training',
                                 budget=300000, output=root / OUTPUT, execute=execute)


def test_default_repair_preflight_has_no_training(capsys) -> None:
    """真实登记在无执行开关时仍仅打印身份。"""
    root = Path(__file__).resolve().parents[1]
    assert run_b0_training.main(['--repair-registration', str(root / CONFIG)]) == 0
    assert 'PREFLIGHT_ONLY' in capsys.readouterr().out


@pytest.mark.parametrize('key,value', [
    ('training_seeds', [11, 22, 44]), ('task_profile', 'cv_train_v1'),
    ('transition_budget_per_seed', 300000), ('total_transition_budget', 600000),
    ('control_group', 'ENGINEERING_CHECKPOINT'), ('device', 'cpu'),
    ('method', 'CONSTRAINED_SAC'), ('repair_round', 2),
])
def test_candidate_identity_drift_rejected(tmp_path, key, value) -> None:
    """不同seed/方法/阶段/预算/轮次不能借R1开关进入科研。"""
    root = material(tmp_path)
    raw = deepcopy(APPROVED)
    raw[key] = value
    (root / CONFIG).write_text(yaml.safe_dump(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='授权不符'):
        validate(root)


def test_missing_actual_branch_evidence_rejected(tmp_path) -> None:
    """登记路径非空不等于诊断条件已满足。"""
    root = material(tmp_path, decision=False)
    with pytest.raises(ValueError, match='须先完成'):
        validate(root)


@pytest.mark.parametrize('key,value', [
    ('confirmed_L1_defects', ['actual defect']),
    ('unresolved_key_error_or_protocol_conflict', True),
    ('selected_branch', 'L1_REPAIR'),
])
def test_key_error_blocks_l2_execution(tmp_path, key, value) -> None:
    """存在关键错误或L1分支时不能同时试学习率。"""
    import json

    root = material(tmp_path)
    path = root / APPROVED['branch_decision_file']
    decision = json.loads(path.read_text(encoding='utf-8'))
    decision[key] = value
    path.write_text(json.dumps(decision), encoding='utf-8')
    with pytest.raises(ValueError, match='不支持'):
        validate(root)


def test_noncomparable_control_cannot_be_silently_reused(tmp_path) -> None:
    """不匹配时须重新登记C/R，而不是虚报成对条件。"""
    import json

    root = material(tmp_path)
    path = root / APPROVED['comparability_file']
    comparison = json.loads(path.read_text(encoding='utf-8'))
    comparison['control_reuse_eligible'] = False
    path.write_text(json.dumps(comparison), encoding='utf-8')
    with pytest.raises(ValueError, match='不支持'):
        validate(root)


def test_unregistered_lr_change_and_engineering_lr_change_rejected() -> None:
    """原生产默认仍3e-4；只有精确R1配置可使用唯一候选。"""
    assert SACConfig().learning_rate == 3e-4
    for kind, budget in (('scientific_training', 100000), ('engineering_smoke', 264)):
        with pytest.raises(ValueError, match='禁止修改'):
            B0HarnessConfig(run_kind=kind, transition_budget=budget,
                            sac=SACConfig(learning_rate=1e-4))
    allowed = B0HarnessConfig(run_kind='scientific_training', transition_budget=100000,
                              validation_episodes=30, research_registration=REGISTRATION_ID,
                              sac=SACConfig(learning_rate=1e-4))
    assert allowed.sac.learning_rate == 1e-4


@pytest.mark.parametrize('value', [3e-5, 2e-4, 5e-4])
def test_no_second_learning_rate_candidate(value) -> None:
    """不能借修复身份任意修改公共步长。"""
    with pytest.raises(ValueError, match='禁止修改'):
        B0HarnessConfig(run_kind='scientific_training', transition_budget=100000,
                        validation_episodes=30, research_registration=REGISTRATION_ID,
                        sac=SACConfig(learning_rate=value))


def test_legal_registered_entry_only_dispatches_r1_mock(tmp_path, monkeypatch) -> None:
    """合法条件启用批次；夹具无env/Actor/backward。"""
    from auv_risk_rl.training import repair_r1

    root = material(tmp_path)
    calls = []
    monkeypatch.setattr(run_b0_training, 'ROOT', root)
    monkeypatch.setattr(repair_r1, 'run_repair_batch',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or
                        {'status': 'MOCK_ONLY_NOT_TRAINING'})
    assert run_b0_training.main([
        '--repair-registration', str(root / CONFIG), '--execute', '--run-kind',
        'scientific_training', '--transition-budget', '300000',
        '--output-dir', str(root / OUTPUT)]) == 0
    assert len(calls) == 1 and calls[0][0][1] == APPROVED


def test_v1_and_repair_registration_cannot_be_combined() -> None:
    """互斥入口防止误跑CV旧批次。"""
    with pytest.raises(SystemExit) as error:
        run_b0_training.main(['--repair-registration', CONFIG, '--mvp-registration',
                              'configs/stage2_b0_mvp_v1.yaml'])
    assert error.value.code == 2


def test_segment_logger_records_own_registration(tmp_path) -> None:
    """复用缓冲/确认前缀日志但不冒称V1结果。"""
    import json

    logger = SegmentLog(tmp_path, 11, parent=None, cutoff=None, registration_id=REGISTRATION_ID)
    logger('episode', {'fixture_only': True, 'log_sequence': 1})
    logger.confirm(1, 'COMPLETED')
    logger.close()
    record = json.loads((logger.directory / 'episode.jsonl').read_text(encoding='utf-8'))
    assert record['registration_id'] == REGISTRATION_ID and record['fixture_only']
