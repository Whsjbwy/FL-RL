"""本批科研身份核对；真实登记、固定预算和Git版本不能被任意非空字符串替代。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

REGISTRATION_ID = 'STAGE2_B0_MVP_BATCH_V1'
DOCUMENT = 'docs/STAGE2_B0_MVP_V1.md'
CONFIG = 'configs/stage2_b0_mvp_v1.yaml'
OUTPUT = 'results/stage2_b0_mvp_v1'
APPROVED = {
    'registration_id': REGISTRATION_ID, 'registration_document': DOCUMENT,
    'run_kind': 'scientific_training', 'method': 'B0_FULL_STATE_ORDINARY_SAC',
    'project_config': 'configs/stage0.yaml', 'scenario_config': 'configs/train_scenario_v1.yaml',
    'output_directory': OUTPUT, 'training_seeds': [11, 22, 33],
    'transition_budget_per_seed': 300000, 'total_transition_budget': 900000,
    'num_envs': 2, 'device': 'cuda',
    'curriculum': [{'profile': 'obstacle_free', 'start': 1, 'end': 100000},
                   {'profile': 'cv_train_v1', 'start': 100001, 'end': 300000}],
    'execution_order': [{'seed': seed, 'stop': stop}
                        for stop in (100000, 300000) for seed in (11, 22, 33)],
    'retain_agent_optimizer_replay_at_switch': True,
    'validation': {
        'root_seed': 20261006, 'base_indices': {'first': 0, 'last': 299},
        'monitor_indices': {'first': 0, 'last': 29},
        'obstacle_free_monitor_steps': [0, 25000, 50000, 75000],
        'obstacle_free_full_steps': [100000],
        'cv_monitor_steps': [100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000],
        'cv_full_steps': [300000], 'task_horizon': 1000, 'deterministic': True,
        'trajectory_indices': [0, 1, 2], 'earliest_failure_trajectory': True,
    },
    'checkpoint_interval': 25000, 'logging_flush_records': 100, 'primary_checkpoint': 'final',
    'sac': {'gamma': 0.999, 'tau': 0.005, 'learning_rate': 0.0003, 'batch_size': 256,
            'replay_capacity': 500000, 'learning_starts': 10000, 'utd': 1,
            'initial_alpha': 0.2, 'target_entropy': -3.0},
}


def git(root: Path, *arguments: str) -> str:
    """只调用正常Git版本管理，不计算额外文件摘要。"""
    result = subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', *arguments],
                            cwd=root, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def validate_registration(root: Path, path: Path, *, run_kind: str | None,
                          budget: int | None, output: Path | None,
                          execute: bool) -> dict[str, Any]:
    """必须是实际canonical登记且全部字段与本次授权一致，不接受任意登记文件。"""
    if path.resolve() != (root / CONFIG).resolve() or not path.is_file():
        raise ValueError('科研入口必须读取实际canonical运行登记配置。')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if raw != APPROVED:
        raise ValueError('登记的seed、预算、课程、参数、验证或输出身份与本轮授权不一致。')
    document = root / DOCUMENT
    if not document.is_file() or REGISTRATION_ID not in document.read_text(encoding='utf-8'):
        raise ValueError('真实运行前登记文档缺失或身份不符。')
    if execute and (run_kind != 'scientific_training' or budget != 900000
                    or output is None or output.resolve() != (root / OUTPUT).resolve()):
        raise ValueError('执行须显式匹配科研run_kind、批次900000预算和登记输出目录。')
    return raw


def verify_code_identity(root: Path, commit: str) -> None:
    """结果/回执提交可新增；科学代码、配置及登记不得偏离被测实验提交。"""
    protected = ['src', 'tests', 'scripts', 'configs', DOCUMENT,
                 'pyproject.toml', 'requirements-b1-tools.txt']
    git(root, 'cat-file', '-e', f'{commit}^{{commit}}')
    changed = git(root, 'diff', '--name-only', commit, '--', *protected)
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '--', *protected)
    if changed or untracked:
        raise ValueError(f'实验代码/配置/登记已偏离固定提交，拒绝执行/恢复：{changed}\n{untracked}')
