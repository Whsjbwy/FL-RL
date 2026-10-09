"""最后一轮B0共同reward与预算对照的显式科研守卫，不放宽旧登记。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from auv_risk_rl.training.mvp_registration import git

REGISTRATION_ID = 'STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET'
DOCUMENT = 'docs/STAGE2_B0_R2.md'
CONFIG = 'configs/stage2_b0_r2.yaml'
OUTPUT = 'results/stage2_b0_r2'
APPROVED = {
    'registration_id': REGISTRATION_ID, 'registration_document': DOCUMENT,
    'run_kind': 'scientific_training', 'method': 'B0_FULL_STATE_ORDINARY_SAC',
    'repair_round': 2, 'project_config': 'configs/stage0.yaml',
    'scenario_config': 'configs/train_scenario_v1.yaml', 'output_directory': OUTPUT,
    'training_seeds': [11, 22, 33], 'groups': {'C300': {'w_goal': 100.0},
                                            'G200': {'w_goal': 200.0}},
    'run_order': [f'{group}_seed{seed}' for group in ('C300', 'G200') for seed in (11, 22, 33)],
    'task_profile': 'obstacle_free', 'transition_budget_per_run': 300000,
    'total_transition_budget': 1800000, 'num_envs': 2, 'device': 'cuda',
    'reward': {'w_progress': 1.0, 'w_time': 0.01, 'w_smooth': 0.02},
    'validation': {'root_seed': 20261006, 'base_indices': {'first': 0, 'last': 299},
                   'monitor_indices': {'first': 0, 'last': 29},
                   'monitor_steps': list(range(0, 300001, 25000)),
                   'full_steps': [100000, 300000], 'full_is_additional_to_monitor': True,
                   'task_horizon': 1000, 'deterministic': True,
                   'trajectory_indices': [0, 1, 2], 'earliest_failure_trajectory': True},
    'checkpoint_interval': 25000, 'logging_flush_records': 100, 'primary_checkpoint': 'final',
    'sac': {'gamma': 0.999, 'tau': 0.005, 'learning_rate': 0.0003, 'batch_size': 256,
            'replay_capacity': 500000, 'learning_starts': 10000, 'utd': 1,
            'initial_alpha': 0.2, 'target_entropy': -3.0},
}


def validate_registration(root: Path, path: Path, *, run_kind: str | None,
                          budget: int | None, output: Path | None,
                          execute: bool) -> dict[str, Any]:
    """只有真实canonical登记、完整两组预算和无关键缺陷的审核才能科研执行。"""
    if path.resolve() != (root / CONFIG).resolve() or not path.is_file():
        raise ValueError('R2必须使用实际canonical登记配置。')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if raw != APPROVED:
        raise ValueError('R2组/seed/reward/预算/验证/方法与授权不一致。')
    document = root / DOCUMENT
    if not document.is_file() or REGISTRATION_ID not in document.read_text(encoding='utf-8'):
        raise ValueError('R2真实运行前登记缺失。')
    if not execute:
        return raw
    if (run_kind != 'scientific_training' or budget != 1800000 or output is None
            or output.resolve() != (root / OUTPUT).resolve()):
        raise ValueError('R2须显式匹配scientific_training、1800000预算及独立输出。')
    audit_path = root / OUTPUT / 'PRETRAIN_MODEL_REWARD_AUDIT.json'
    if not audit_path.is_file():
        raise ValueError('R2必须先完成真实模型/目标/reward一致性审核。')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit.get('registration_id') != REGISTRATION_ID
            or audit.get('confirmed_critical_L1_defects') != []
            or audit.get('unresolved_scientific_conflicts') != []
            or audit.get('training_permitted') is not True):
        raise ValueError('R2审核未排除关键L1/科学冲突，拒绝训练。')
    return raw


def verify_code_identity(root: Path, commit: str) -> None:
    """用正常Git检查真实被测科学代码与固定登记，报告提交不改变运行身份。"""
    protected = ['src', 'tests', 'scripts', 'configs', DOCUMENT,
                 'pyproject.toml', 'requirements-b1-tools.txt']
    git(root, 'cat-file', '-e', f'{commit}^{{commit}}')
    changed = git(root, 'diff', '--name-only', commit, '--', *protected)
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '--', *protected)
    if changed or untracked:
        raise ValueError(f'R2科学代码/配置/测试/登记相对实验提交漂移：{changed}\n{untracked}')
