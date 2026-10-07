"""按实际V1身份、显式随机流和普通Git差异检查C复用；不训练或生成额外摘要。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from auv_risk_rl.config import load_project_config
from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.fixed_validation import FixedValidationPool
from auv_risk_rl.training.harness import states_equal
from auv_risk_rl.training.mvp_analysis import confirmed_records
from auv_risk_rl.training.mvp_registration import git
from auv_risk_rl.training.repair_registration import REGISTRATION_ID
from auv_risk_rl.training.scenarios import B0ScenarioSource

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """检查旧真实种子/有效日志和新初始化，只有公共学习率是科学参数差异。"""
    old_root = ROOT / 'results/stage2_b0_mvp_v1'
    state = json.loads((old_root / 'batch_state.json').read_text(encoding='utf-8'))
    commit = '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898'
    if state['experiment_code_commit'] != commit or state['status'] != 'COMPLETED':
        raise ValueError('旧C真实批次或实验代码身份不符。')
    shared_paths = ['src/auv_risk_rl/rl', 'src/auv_risk_rl/env',
                    'src/auv_risk_rl/dynamics', 'src/auv_risk_rl/frames.py',
                    'src/auv_risk_rl/seeding.py', 'src/auv_risk_rl/training/harness.py',
                    'src/auv_risk_rl/training/scenarios.py',
                    'src/auv_risk_rl/training/fixed_validation.py',
                    'configs/stage0.yaml', 'configs/train_scenario_v1.yaml']
    changed = git(ROOT, 'diff', '--name-only', commit, '--', *shared_paths)
    if changed:
        raise ValueError('共享科学路径变化，不能默认复用C：' + changed)
    project = load_project_config(ROOT / 'configs/stage0.yaml')
    scenario = load_training_scenario_config(ROOT / 'configs/train_scenario_v1.yaml')
    pool = FixedValidationPool(project, scenario, root_seed=20261006)
    old_manifest = json.loads((old_root / 'validation_manifest.json').read_text(encoding='utf-8'))
    current_manifest = pool.compact_manifest()
    # V1发布补充了split及说明，实际场景/噪声/生成版本仍逐字段精确比较。
    annotation_keys = old_manifest.keys() - current_manifest.keys()
    if annotation_keys != {'identity_note', 'split'} or old_manifest['split'] != 'validation':
        raise ValueError('旧Val清单出现未说明的附加字段，拒绝默认忽略。')
    historical_annotations = {key: old_manifest[key] for key in annotation_keys}
    manifest_matches = current_manifest == {key: old_manifest[key] for key in current_manifest}
    if not manifest_matches:
        raise ValueError('原固定Val基底/环境种子与R1不匹配。')
    rows = []
    for seed in (11, 22, 33):
        original = derived_sac_config(SACConfig(device='cuda'), training_seed=seed,
                                      run_kind='scientific_training')
        candidate = derived_sac_config(SACConfig(device='cuda', learning_rate=1e-4),
                                       training_seed=seed, run_kind='scientific_training')
        values = B0HarnessConfig(run_kind='scientific_training', training_seed=seed,
                                 transition_budget=100000, validation_episodes=30,
                                 research_registration=REGISTRATION_ID, sac=candidate)
        source = B0ScenarioSource(values, project, scenario)
        episodes = [row for kind, row in confirmed_records(old_root, seed, commit, {})
                    if kind == 'episode' and row['task_profile'] == 'obstacle_free']
        for episode in episodes:
            if episode['config']['sac'] != asdict(original):
                raise ValueError('旧C有效初始化/Actor/Replay参数不同。')
            index = episode['scenario_index']
            actual = source.scenario(index)
            if (episode['scenario_root_seed'] != source.split_seed('train')
                    or episode['scenario_id'] != actual.scenario_id
                    or episode['warmup']['root_seed'] != source.environment_seed(index)
                    or episode['config']['run_kind'] != values.run_kind
                    or episode['config']['num_envs'] != values.num_envs):
                raise ValueError('run_id意外改变成对外生场景/环境流或采样调度。')
        steps = sum(episode['steps'] for episode in episodes)
        if steps != 100000:
            raise ValueError('旧C空场景预算不等于100000。')
        control_agent = OrdinarySACAgent(original, source_fingerprint='git:C-reference')
        repaired_agent = OrdinarySACAgent(candidate, source_fingerprint='git:R1-reference')
        names = ('actor', 'q1', 'q2', 'target_q1', 'target_q2')
        same_weights = all(
            states_equal(getattr(control_agent, name).state_dict(),
                         getattr(repaired_agent, name).state_dict()) for name in names)
        rng_matches = (torch.equal(control_agent.generator.get_state(),
                                   repaired_agent.generator.get_state())
                       and states_equal(control_agent.replay.rng.bit_generator.state,
                                        repaired_agent.replay.rng.bit_generator.state))
        alpha_matches = torch.equal(control_agent.log_alpha, repaired_agent.log_alpha)
        if not (same_weights and rng_matches and alpha_matches):
            raise ValueError('成对初始网络/Actor/Replay流或alpha不一致。')
        rows.append(dict(seed=seed, initialization_seed=original.initialization_seed,
                         actor_seed=original.actor_seed, replay_seed=original.replay_seed,
                         scenario_root_seed=source.split_seed('train'),
                         environment_root_seed=source.split_seed('train', 'environment'),
                         first_environment_seed=source.environment_seed(0),
                         verified_training_episodes_including_fragments=len(episodes),
                         verified_control_transitions=steps,
                         initial_models_exact_equal=same_weights,
                         initial_actor_replay_rng_equal=rng_matches,
                         initial_alpha_equal=alpha_matches,
                         initial_adam_states_empty=all(not optimizer.state for optimizer in
                                                      repaired_agent.optimizers.values()),
                         control_lr=original.learning_rate, candidate_lr=candidate.learning_rate))
        del control_agent, repaired_agent
    record = dict(recorded_at_utc=datetime.now(UTC).isoformat(),
                  control_experiment_commit=commit, control_reuse_eligible=True,
                  only_scientific_parameter_changed='learning_rate', seeds=[11, 22, 33],
                  effective_random_streams=rows, shared_scientific_git_diff=changed,
                  checked_shared_paths=shared_paths, fixed_validation_manifest_exact_equal=True,
                  historical_manifest_annotations=historical_annotations,
                  changed_nonrandom_identity_fields=['research_registration', 'transition_budget',
                                                     'code_version', 'output_directory'],
                  budget_note='V1前100k与R1同任务；停止标签不改变已存transition。',
                  future_realized_rng_divergence_expected=True,
                  rng_note='学习率影响参数/轨迹/episode长短后，场景发放位置可不同；'
                           '相同派生规则不表示不同算法强制同终止或同时间线。',
                  actual_env_transitions=0, complete_sac_updates=0,
                  fresh_model_initialization_pairs=3, control_rerun_required=False)
    (Path(__file__).parent / 'control_comparability.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
