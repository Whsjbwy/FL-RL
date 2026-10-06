"""真实登记守卫与默认preflight；所有执行测试使用mock，不开展科研训练。"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from scripts.run_b0_training import main

from auv_risk_rl.training.mvp_batch import BatchLock, SegmentLog, finalize_results, harness_config
from auv_risk_rl.training.mvp_registration import APPROVED, CONFIG, DOCUMENT, OUTPUT
from auv_risk_rl.training.mvp_registration import validate_registration as validate

ROOT = Path(__file__).resolve().parents[1]


def registered_root(tmp_path: Path, raw: dict | None = None) -> Path:
    """小型临时登记材料，不复制整套工程或虚拟环境。"""
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'docs').mkdir()
    (tmp_path / CONFIG).write_text(yaml.safe_dump(APPROVED if raw is None else raw),
                                   encoding='utf-8')
    (tmp_path / DOCUMENT).write_text('STAGE2_B0_MVP_BATCH_V1', encoding='utf-8')
    return tmp_path


def test_mvp_default_preflight(capsys) -> None:
    """真实登记也不能在没有执行开关时开始学习。"""
    assert main(['--mvp-registration', str(ROOT / CONFIG)]) == 0
    assert 'PREFLIGHT_ONLY' in capsys.readouterr().out


@pytest.mark.parametrize('field,value', [('training_seeds', [11, 22, 44]),
                                        ('transition_budget_per_seed', 500000),
                                        ('run_kind', 'engineering_smoke'),
                                        ('method', 'CONSTRAINED_SAC'), ('device', 'cpu')])
def test_bad_registration_rejected(tmp_path, field, value) -> None:
    """seed/预算/方法/设备漂移不能凭非空路径放行。"""
    raw = deepcopy(APPROVED)
    raw[field] = value
    root = registered_root(tmp_path, raw)
    with pytest.raises(ValueError):
        validate(root, root / CONFIG, run_kind=None, budget=None, output=None, execute=False)


@pytest.mark.parametrize('kind,budget,directory', [(None, 900000, OUTPUT),
                                                ('scientific_training', 300000, OUTPUT),
                                                ('scientific_training', 900000, 'wrong')])
def test_explicit_execution_identity(tmp_path, kind, budget, directory) -> None:
    """执行的批次总预算及输出必须显式匹配。"""
    root = registered_root(tmp_path)
    with pytest.raises(ValueError):
        validate(root, root / CONFIG, run_kind=kind, budget=budget,
                 output=root / directory, execute=True)


def test_missing_document_rejected(tmp_path) -> None:
    """没有真实运行登记文档就不能开启科研入口。"""
    root = registered_root(tmp_path)
    (root / DOCUMENT).unlink()
    with pytest.raises(ValueError):
        validate(root, root / CONFIG, run_kind=None, budget=None, output=None, execute=False)


def test_legal_entry_dispatches_batch(monkeypatch) -> None:
    """合法明确登记启用科研调度，mock证明入口不再无条件拒绝。"""
    calls = []
    monkeypatch.setattr('auv_risk_rl.training.mvp_batch.run_batch',
                        lambda root, raw, **kwargs: calls.append((root, raw, kwargs)) or
                        {'status': 'MOCK_ONLY'})
    assert main(['--mvp-registration', str(ROOT / CONFIG), '--execute', '--run-kind',
                 'scientific_training', '--transition-budget', '900000',
                 '--output-dir', str(ROOT / OUTPUT)]) == 0
    assert len(calls) == 1 and calls[0][1] == APPROVED


def test_independent_seed_production_configs() -> None:
    """所有科学规模保留，三个seed的模型/策略/Replay流独立。"""
    configs = [harness_config(APPROVED, seed) for seed in (11, 22, 33)]
    assert all(c.sac.learning_starts == 10000 and c.sac.replay_capacity == 500000
               and c.sac.batch_size == 256 and c.num_envs == 2 for c in configs)
    assert len({c.sac.initialization_seed for c in configs}) == 3
    with pytest.raises(ValueError):
        harness_config(APPROVED, 44)


def test_live_lock_rejects_duplicate(tmp_path) -> None:
    """真实活PID锁不允许恢复或新开第二个同批次进程。"""
    lock = BatchLock(tmp_path / 'batch.lock', resume=False)
    try:
        with pytest.raises(RuntimeError):
            BatchLock(tmp_path / 'batch.lock', resume=True)
    finally:
        lock.close()


def test_log_resume_preserves_failed_tail(tmp_path) -> None:
    """旧尾部保留且有效前缀明确；新段序号续接不会覆盖文件。"""
    first = SegmentLog(tmp_path, 11, parent=None, cutoff=None)
    first('episode', {'log_sequence': 1, 'complete': False})
    first.confirm(1, 'RUNNING')
    first('episode', {'log_sequence': 2, 'complete': False})
    first.close()
    second = SegmentLog(tmp_path, 11, parent=tmp_path / 'latest.pt', cutoff=1)
    assert second.metadata[0]['valid_log_sequence'] == 1
    assert len((first.directory / 'episode.jsonl').read_text(encoding='utf-8').splitlines()) == 2
    second('episode', {'log_sequence': 2, 'complete': True})
    second.confirm(2, 'COMPLETED')
    second.close()


def test_unfinished_batch_cannot_finalize(tmp_path) -> None:
    """RUNNING不能生成最终科研结果或假装固定策略诊断已经完成。"""
    with pytest.raises(RuntimeError):
        finalize_results(tmp_path, {'status': 'RUNNING'})
