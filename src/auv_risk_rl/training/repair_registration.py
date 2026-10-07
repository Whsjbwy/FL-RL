"""本轮唯一R1修复候选的科研身份守卫；不放宽原生产或MVP登记。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from auv_risk_rl.training.mvp_registration import git

REGISTRATION_ID = 'STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1'
DOCUMENT = 'docs/STAGE2_B0_REPAIR_R1.md'
CONFIG = 'configs/stage2_b0_repair_r1.yaml'
OUTPUT = 'results/stage2_b0_repair_r1'
APPROVED = {
    'registration_id': REGISTRATION_ID, 'registration_document': DOCUMENT,
    'run_kind': 'scientific_training', 'method': 'B0_FULL_STATE_ORDINARY_SAC',
    'branch': 'L2_COMMON_LEARNING_RATE', 'repair_round': 1,
    'project_config': 'configs/stage0.yaml', 'scenario_config': 'configs/train_scenario_v1.yaml',
    'output_directory': OUTPUT, 'training_seeds': [11, 22, 33],
    'task_profile': 'obstacle_free', 'transition_budget_per_seed': 100000,
    'total_transition_budget': 300000, 'hard_new_training_limit': 600000,
    'control_group': 'REUSE_V1_FIRST_100K',
    'control_experiment_commit': '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898',
    'num_envs': 2, 'device': 'cuda',
    'validation': {'root_seed': 20261006, 'base_indices': {'first': 0, 'last': 299},
                   'monitor_indices': {'first': 0, 'last': 29},
                   'monitor_steps': [0, 25000, 50000, 75000], 'full_steps': [100000],
                   'task_horizon': 1000, 'deterministic': True,
                   'trajectory_indices': [0, 1, 2], 'earliest_failure_trajectory': True},
    'checkpoint_interval': 25000, 'logging_flush_records': 100,
    'small_model_steps': [25000, 50000, 75000, 100000], 'primary_checkpoint': 'final',
    'sac': {'gamma': 0.999, 'tau': 0.005, 'learning_rate': 0.0001, 'batch_size': 256,
            'replay_capacity': 500000, 'learning_starts': 10000, 'utd': 1,
            'initial_alpha': 0.2, 'target_entropy': -3.0},
    'branch_decision_file': OUTPUT + '/branch_decision.json',
    'comparability_file': OUTPUT + '/control_comparability.json',
}


def validate_registration(root: Path, path: Path, *, run_kind: str | None,
                          budget: int | None, output: Path | None,
                          execute: bool) -> dict[str, Any]:
    """候选、登记、诊断分支和有效随机流对照必须真实匹配；默认仍仅preflight。"""
    if path.resolve() != (root / CONFIG).resolve() or not path.is_file():
        raise ValueError('R1科研入口必须读取实际canonical修复配置。')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if raw != APPROVED:
        raise ValueError('R1 seed/预算/唯一学习率/验证/方法/输出与授权不符。')
    document = root / DOCUMENT
    if not document.is_file() or REGISTRATION_ID not in document.read_text(encoding='utf-8'):
        raise ValueError('R1真实预登记缺失。')
    if not execute:
        return raw
    if (run_kind != 'scientific_training' or budget != 300000 or output is None
            or output.resolve() != (root / OUTPUT).resolve()):
        raise ValueError('R1执行须显式匹配科研run_kind、300000预算和独立输出。')
    decision_path = root / raw['branch_decision_file']
    comparable_path = root / raw['comparability_file']
    if not decision_path.is_file() or not comparable_path.is_file():
        raise ValueError('须先完成真实诊断分支判定与C组随机流/科学语义对照。')
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    comparison = json.loads(comparable_path.read_text(encoding='utf-8'))
    if (decision.get('registration_id') != REGISTRATION_ID
            or decision.get('selected_branch') != 'L2_COMMON_LEARNING_RATE'
            or decision.get('confirmed_L1_defects') != []
            or decision.get('unresolved_key_error_or_protocol_conflict') is not False
            or comparison.get('control_reuse_eligible') is not True
            or comparison.get('control_experiment_commit') != raw['control_experiment_commit']
            or comparison.get('only_scientific_parameter_changed') != 'learning_rate'
            or comparison.get('seeds') != [11, 22, 33]):
        raise ValueError('诊断/可比性证据不支持本轮L2复测，拒绝执行。')
    return raw


def verify_code_identity(root: Path, commit: str) -> None:
    """报告可继续提交；本轮科学代码/配置/测试/登记相对运行前提交必须保持一致。"""
    protected = ['src', 'tests', 'scripts', 'configs', DOCUMENT,
                 'pyproject.toml', 'requirements-b1-tools.txt']
    git(root, 'cat-file', '-e', f'{commit}^{{commit}}')
    changes = git(root, 'diff', '--name-only', commit, '--', *protected)
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '--', *protected)
    if changes or untracked:
        raise ValueError(f'R1实验源码/配置/测试/登记发生漂移：{changes}\n{untracked}')
