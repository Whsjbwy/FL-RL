"""R1固定100k终点对照与可重建结果；默认只读日志，显式授权才运行九条冻结诊断。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
V1 = ROOT/'results'/'stage2_b0_mvp_v1'
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(TASK)]
REGISTRATION_ID = 'STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1'
CONTROL_COMMIT = '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898'
SEEDS = (11, 22, 33)
POINTS = (0, 25000, 50000, 75000, 100000)
COMPONENTS = ('progress', 'goal', 'time', 'smoothness')
UPDATE_FIELDS = ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss', 'alpha',
                 'actor_gradient_norm', 'q1_gradient_norm', 'q2_gradient_norm',
                 'alpha_gradient_norm')
EVENTS = ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')
COLORS = ('#0072B2', '#D55E00', '#009E73')


def json_default(value: Any) -> Any:
    """只转换NumPy数组/标量；未知类型和非有限数值继续显式报错。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def json_write(path: Path, value: Any) -> None:
    """本轮派生汇总可重新生成，原始V1/R1日志永不修改。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=json_default)+'\n',
                    encoding='utf-8')


def csv_write(path: Path, rows: list[dict[str, Any]], *, compressed: bool = False) -> None:
    """字段并集保留未知null；必要轨迹直接gzip，不生成重复大型CSV。"""
    if not rows:
        raise ValueError(f'不能将空记录冒充完成结果: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    opener = gzip.open if compressed else Path.open
    mode = 'wt' if compressed else 'w'
    with opener(path, mode, encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False,
                                              default=json_default)
                             if isinstance(value, list | dict | tuple | np.ndarray)
                             else json_default(value) if isinstance(value, np.generic) else value
                             for key, value in row.items()})


def moment(values: list[float]) -> dict[str, Any]:
    """跨seed样本标准差n-1；episode不是新的训练重复。"""
    if not values:
        return dict(n=0, mean=None, sample_sd=None)
    values_array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values_array)):
        raise FloatingPointError('结果统计含非有限值。')
    return dict(n=len(values), mean=float(values_array.mean()),
                sample_sd=float(values_array.std(ddof=1)) if len(values) > 1 else None)


def bin_stop(transition: int) -> int:
    """整episode归结束transition区间；与逐transition数据分母分开。"""
    return ((max(1, transition)-1)//25000+1)*25000


def episode_details(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """复用可信科学分母，补R1实际测得的边界子类型/目标接近字段。"""
    from auv_risk_rl.training.mvp_analysis import summarize_episodes

    stats = summarize_episodes(episodes)
    complete = [row for row in episodes if row['complete']]
    boundary = [row for row in complete if row['failure_type'] == 'operational_boundary_failure']
    subtype = Counter(row.get('boundary_subtype') or 'NOT_RECORDED' for row in boundary)
    extra_names = ('minimum_goal_distance_m', 'minimum_goal_distance_time_s',
                   'first_within_10m_time_s', 'first_goal_entry_time_s',
                   'initial_distance_m', 'final_distance_m')
    approach_measured = sum('first_within_10m_time_s' in row for row in complete)
    entry_measured = sum('first_goal_entry_time_s' in row for row in complete)
    stats.update(boundary_subtypes=dict(subtype),
                 boundary_subtype_scope='complete episodes; C historical subtype unavailable',
                 complete_episode_steps=sum(row['steps'] for row in complete),
                 successful_episode_transitions=sum(row['steps'] for row in complete
                                                   if row['failure_type'] == 'goal_success'),
                 successful_terminal_transitions=sum(row['failure_type'] == 'goal_success'
                                                    for row in complete),
                 reward_components={name: moment([row['reward_components'][name]
                                                 for row in complete]) for name in COMPONENTS},
                 additional_moments={name: moment([float(row[name]) for row in complete
                                                  if row.get(name) is not None])
                                     for name in extra_names},
                 near_goal_10m_episode_count=(sum(row.get('first_within_10m_time_s') is not None
                                                 for row in complete)
                                             if approach_measured else None),
                 goal_approach_measured_episodes=approach_measured,
                 goal_entry_observed_episode_count=(sum(
                     row.get('first_goal_entry_time_s') is not None for row in complete)
                                                   if entry_measured else None),
                 success_missing_goal_entry_diagnostic=(sum(
                     row['failure_type'] == 'goal_success' and 'first_goal_entry_time_s' in row
                     and row['first_goal_entry_time_s'] is None for row in complete)
                                                       if entry_measured else None),
                 goal_entry_without_success=(sum(row.get('first_goal_entry_time_s') is not None
                                                 and row['failure_type'] != 'goal_success'
                                                 for row in complete)
                                             if entry_measured else None))
    return stats


def public_episode(episode: dict[str, Any], group: str,
                   validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """逐episode公开必要原始标量；配置/版本可追踪，完整轨迹另保存。"""
    keys = ('training_seed', 'task_profile', 'scenario_id', 'scenario_index', 'scenario_root_seed',
            'environment_seed', 'episode_id', 'env_slot', 'split', 'start_transition',
            'last_transition', 'steps', 'physical_time_s', 'reward', 'path_length_m',
            'path_length_definition', 'minimum_clearance_m', 'action_saturation_count',
            'failure_type', 'complete', 'terminated', 'truncated', 'success', 'collision',
            'boundary', 'task_timeout', 'external_truncation', 'phase_boundary', 'budget_stop',
            'boundary_subtype', 'boundary_constraints', 'minimum_goal_distance_m',
            'minimum_goal_distance_time_s', 'first_within_10m_time_s', 'first_goal_entry_time_s',
            'initial_distance_m', 'final_distance_m', 'diagnostic_scope', 'registration_id',
            'code_version', 'run_kind', 'method', 'segment_id', 'log_sequence')
    output = {key: episode.get(key) for key in keys}
    output.update(group=group, physical_event_denominator_eligible=episode['complete'],
                  **{f'reward_{name}': episode['reward_components'][name] for name in COMPONENTS})
    warmup = episode.get('warmup', {})
    output.update(warmup_environment_seed=warmup.get('root_seed'),
                  warmup_duration_s=warmup.get('warmup_duration_s'))
    source = episode.get('_log_source', {})
    output.update(source_path=source.get('path'), source_line=source.get('line_number'),
                  valid_log_sequence=source.get('valid_log_sequence'))
    if validation is not None:
        output.update(at_transition=validation['at_transition'],
                      evaluation_key=validation['evaluation_key'],
                      evaluation_kind='Val300' if validation['full'] else 'monitor30',
                      evaluation_episode_count=validation['count'],
                      validation_root_seed=validation['validation_root_seed'],
                      base_scenario_id=episode['base_scenario_id'],
                      actual_scenario_id=episode['actual_scenario_id'],
                      environment_seed=episode['environment_seed'],
                      code_version=validation['code_version'],
                      registration_id=validation['registration_id'],
                      log_sequence=validation['log_sequence'],
                      segment_id=validation['segment_id'],
                      method=validation['method'], run_kind=validation['run_kind'],
                      source_path=validation['_log_source']['path'],
                      source_line=validation['_log_source']['line_number'],
                      valid_log_sequence=validation['_log_source']['valid_log_sequence'])
    return output


def trajectory_rows(row: dict[str, Any], episode: dict[str, Any],
                    group: str) -> list[dict[str, Any]]:
    """仅原预登记保留轨迹；控制节点原值、不插值、不按效果筛选。"""
    identity = dict(group=group, training_seed=row['training_seed'],
                    at_transition=row['at_transition'], scenario_index=episode['index'],
                    scenario_id=episode['scenario_id'],
                    evaluation_kind='Val300' if row['full'] else 'monitor30',
                    trajectory_retention_reasons=episode.get('trajectory_retention_reasons'),
                    profile=row['profile'], failure_type=episode['failure_type'],
                    goal_north_m=episode['goal_position_ned_m'][0],
                    goal_east_m=episode['goal_position_ned_m'][1],
                    goal_down_m=episode['goal_position_ned_m'][2])
    initial = episode['initial_position_ned_m']
    output = [dict(identity, control_step=0, time_s=0., north_m=initial[0], east_m=initial[1],
                   down_m=initial[2], action_surge=None, action_yaw=None, action_pitch=None,
                   reward=None, reward_progress=None, reward_goal=None, reward_time=None,
                   reward_smoothness=None)]
    elapsed = 0.
    for node in episode['trajectory']:
        elapsed += node['elapsed_s']
        position, action = node['position_ned_m'], node['action']
        output.append(dict(
            identity, control_step=node['task_step'], time_s=elapsed,
            north_m=position[0], east_m=position[1], down_m=position[2],
            action_surge=action[0], action_yaw=action[1], action_pitch=action[2],
            reward=node['reward'], **{f'reward_{name}': node['reward_components'][name]
                                      for name in COMPONENTS}))
    return output


def collect_group(group: str, root: Path, code: str, registration: str,
                  ) -> dict[str, Any]:
    """仅确认日志前缀；C限定V1原100k，不把CV阶段并入修复曲线。"""
    from auv_risk_rl.training.mvp_analysis import Moments, confirmed_records, validation_point

    episodes, validations, update_bins = [], [], []
    public_training, public_validation, trajectories = [], [], []
    source_diagnostics, seed_counts = {}, {}
    expected_reg = registration
    for seed in SEEDS:
        diagnostics: dict[str, Any] = {}
        bins: dict[int, dict[str, Any]] = {}
        update_count = transitions = 0
        seed_episodes = []
        seed_validations = []
        observed_updates = []
        for kind, row in confirmed_records(root, seed, code, diagnostics,
                                           registration_id=expected_reg):
            if kind == 'episode':
                if row['task_profile'] != 'obstacle_free':
                    continue
                if row['last_transition'] > 100000:
                    raise ValueError('空场景结束超100k，不能混入对照。')
                seed_episodes.append(row)
                transitions += row['steps']
                public_training.append(public_episode(row, group))
            elif kind == 'update':
                transition = row['transition']
                if transition > 100000:
                    continue
                if transition != 10000+update_count:
                    raise ValueError(f'{group} seed{seed}更新并非10000起连续一次/transition。')
                update_count += 1
                observed_updates.append(transition)
                entry = bins.setdefault(bin_stop(transition),
                                        {name: Moments() for name in UPDATE_FIELDS})
                for name in UPDATE_FIELDS:
                    entry[name].add(row['metrics'][name])
                entry['last_alpha'] = row['metrics']['alpha']
            else:
                if row['profile'] != 'obstacle_free':
                    continue
                validation_point(row)
                seed_validations.append(row)
                for episode in row['episodes']:
                    public_validation.append(public_episode(episode, group, row))
                    if 'trajectory' in episode:
                        trajectories.extend(trajectory_rows(row, episode, group))
        observed = sorted((row['at_transition'], row['full']) for row in seed_validations)
        expected = [(point, point == 100000) for point in POINTS]
        if observed != expected:
            raise ValueError(f'{group} seed{seed}实际固定验证点缺失/重复: {observed}')
        if transitions != 100000 or update_count != 90001:
            raise ValueError(f'{group} seed{seed}确认训练步/update数不符: '
                             f'{transitions}/{update_count}')
        episodes.extend(seed_episodes)
        validations.extend(seed_validations)
        seed_counts[str(seed)] = dict(transitions=transitions, complete_sac_updates=update_count,
                                     optimizer_steps=4*update_count,
                                     first_update_transition=observed_updates[0],
                                     last_update_transition=observed_updates[-1])
        source_diagnostics[str(seed)] = diagnostics
        for stop, metrics in sorted(bins.items()):
            update_bins.append(dict(group=group, training_seed=seed, bin_first=stop-24999,
                                    bin_last=stop, updates=metrics['alpha'].count,
                                    last_alpha=metrics['last_alpha'],
                                    metrics={key: metrics[key].result() for key in UPDATE_FIELDS}))
    return dict(group=group, episodes=episodes, validations=validations, update_bins=update_bins,
                seed_counts=seed_counts, source_diagnostics=source_diagnostics,
                public_training=public_training, public_validation=public_validation,
                trajectory_rows=trajectories)


def validation_identity(row: dict[str, Any]) -> dict[str, Any]:
    """直接场景/初态/环境噪声种子核对，不生成文件摘要。"""
    return dict(root=row['validation_root_seed'], indices=row['indices'],
                episodes=[dict(index=ep['index'], base_scenario_id=ep['base_scenario_id'],
                               actual_scenario_id=ep['actual_scenario_id'],
                               environment_seed=ep['environment_seed'],
                               initial_position_ned_m=ep['initial_position_ned_m'],
                               goal_position_ned_m=ep['goal_position_ned_m'])
                          for ep in row['episodes']])


def compare_pairs(control: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """每seed/点相同验证基底；学习导致episode长度不同不是配对失效。"""
    c_index = {(row['training_seed'], row['at_transition']): row
               for row in control['validations']}
    comparisons = []
    for row in candidate['validations']:
        c_row = c_index[row['training_seed'], row['at_transition']]
        exact = validation_identity(row) == validation_identity(c_row)
        if not exact:
            raise ValueError('C/R1固定验证几何、初态或噪声身份不一致。')
        comparisons.append(dict(training_seed=row['training_seed'],
                                at_transition=row['at_transition'],
                                count=row['count'], direct_scenario_identity_equal=exact))
    return comparisons


def validate_completed_state() -> dict[str, Any]:
    """真实worker完成和全部seed实测计数满足后，才允许终点分析/新冻结诊断。"""
    from auv_risk_rl.training.repair_registration import APPROVED

    path = TASK/'batch_state.json'
    if not path.is_file():
        raise RuntimeError('R1实际批次状态尚不存在；不能生成完成结果。')
    state = json.loads(path.read_text(encoding='utf-8'))
    if (state.get('status') != 'COMPLETED' or state.get('registration_id') != REGISTRATION_ID
            or state.get('registration') != APPROVED
            or state.get('completed_seeds') != list(SEEDS)):
        raise RuntimeError('R1批次未真实完成或登记不符；拒绝终点科研结论。')
    for seed in SEEDS:
        row = state['seeds'][str(seed)]
        if (row.get('status') != 'COMPLETED' or row.get('transitions') != 100000
                or row.get('updates') != 90001 or row.get('checkpoint_transition') != 100000
                or row.get('profile') != 'obstacle_free'):
            raise RuntimeError('R1实际seed完成/保存计数不符。')
    return state


def run_final_cases(state: dict[str, Any], trusted: bool) -> dict[str, Any]:
    """每seed R01–R03一次，不训练；共享旧158条预算防重复执行/越界。"""
    if not trusted:
        raise ValueError('须显式确认仅加载本项目可信本机模型。')
    import torch
    from replay_diagnostics import fixed_cases, run_trajectory

    from auv_risk_rl.config import load_project_config
    from auv_risk_rl.env.b0_navigation import B0NavigationEnv
    from auv_risk_rl.rl.networks import Actor

    destination = TASK/'analysis'/'final_fixed_diagnostics.json'
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        if (previous['experiment_code_commit'] != state['experiment_code_commit']
                or previous['status'] != 'COMPLETED' or len(previous['cases']) != 9):
            raise RuntimeError('已有R1固定诊断不完整/身份不符，不自动重复。')
        return previous
    project = load_project_config(ROOT/'configs'/'stage0.yaml')
    cases = []
    for seed in SEEDS:
        model = TASK/'models'/f'seed_{seed}_100000.pt'
        payload = torch.load(model, map_location='cpu', weights_only=True)
        expected = dict(format='b0-repair-r1-network-snapshot-v1',
                        registration_id=REGISTRATION_ID, seed=seed, transition=100000,
                        code_version=state['experiment_code_commit'],
                        run_kind='scientific_training', method='B0_FULL_STATE_ORDINARY_SAC',
                        branch='L2_COMMON_LEARNING_RATE')
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError('R1终点模型实际格式/登记/seed/方法/代码不符。')
        if payload['sac_config']['learning_rate'] != 1e-4:
            raise ValueError('R1终点模型公共学习率不符。')
        if (payload['counters']['environment_steps'] != 100000
                or payload['counters']['gradient_updates'] != 90001):
            raise ValueError('R1终点模型真实训练计数不符。')
        device = payload['sac_config']['device']
        if device not in ('cuda', 'cuda:0') or not torch.cuda.is_available():
            raise RuntimeError('R1固定诊断CUDA不可用，不能静默CPU回退。')
        with torch.random.fork_rng(devices=[]):
            actor = Actor().to(device=device, dtype=torch.float32)
        actor.load_state_dict(payload['actor'], strict=True)
        actor.eval().requires_grad_(False)
        for case in fixed_cases()[:3]:
            identifier = f'r1_final_s{seed}_{case.case_id}'
            existing = TASK/'diagnostic_replay'/identifier/'summary.json'
            if existing.is_file():
                summary = json.loads(existing.read_text(encoding='utf-8'))
                if (summary.get('status') != 'COMPLETED' or summary.get('experiment_code_commit')
                        != state['experiment_code_commit']):
                    raise RuntimeError('已有该例失败/身份不符，不能自动重跑。')
            else:
                env = B0NavigationEnv(project, case.initial_state(), case.obstacles(), case.goal(),
                                      f'learned-fixed-{case.case_id}')
                metadata = dict(trajectory_id=identifier, selection='R1_FINAL_FIXED',
                                training_seed=seed, model_transition=100000,
                                task_profile='obstacle_free', environment_seed=case.seed,
                                scenario_id=case.case_id,
                                experiment_code_commit=state['experiment_code_commit'],
                                registration_id=REGISTRATION_ID,
                                model_file=model.relative_to(ROOT).as_posix())
                summary, _ = run_trajectory(metadata, env, actor, device)
            # 大型完整next_observation只在本机gzip，公开第一事件前后/输入/小步因果链。
            public = {key: value for key, value in summary.items() if key != 'final_event'}
            event = summary.get('final_event')
            if event is not None:
                public['first_event_chain'] = {key: event[key] for key in (
                    'state_before', 'state_after', 'physical_command', 'nominal_actor_action',
                    'input_audit', 'integration_segments', 'reward_components', 'failure_type')}
            cases.append(public)
    result = dict(status='COMPLETED', experiment_code_commit=state['experiment_code_commit'],
                  cases=cases,
                  diagnostic_control_transitions=sum(row['transitions'] for row in cases),
                  warmup_control_transitions=sum(row['warmup_control_transitions']
                                                for row in cases),
                  successes=sum(row['failure_type'] == 'goal_success' for row in cases),
                  scientific_training_steps=0, scientific_training_updates=0,
                  training_replay_writes=0, actor_gradient_operations=0,
                  claim='Only preregistered fixed learned-policy diagnostic, no new training.')
    json_write(destination, result)
    return result


def analyze(state: dict[str, Any], diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """一趟完整确认日志，固定终点C/R1配对；不替换best模型或拼接新旧run。"""
    c = collect_group('C_3e-4', V1, CONTROL_COMMIT, 'STAGE2_B0_MVP_BATCH_V1')
    r = collect_group('R1_1e-4', TASK, state['experiment_code_commit'], REGISTRATION_ID)
    identity_pairs = compare_pairs(c, r)
    validation_points, endpoint_rows, train_bins, update_bins = [], [], [], []
    validation_full_records = []
    for data in (c, r):
        group = data['group']
        for row in data['validations']:
            point = dict(group=group, training_seed=row['training_seed'],
                         at_transition=row['at_transition'], full=row['full'],
                         evaluation_kind='Val300' if row['full'] else 'monitor30',
                         all_registered=episode_details(row['episodes']),
                         paired_monitor30=episode_details([episode for episode in row['episodes']
                                                          if episode['index'] < 30]))
            validation_points.append(point)
            validation_full_records.append(dict(group=group, **row))
            if row['full']:
                stats = point['all_registered']
                endpoint_rows.append(dict(
                    group=group, training_seed=row['training_seed'], at_transition=100000,
                    episodes=stats['complete_physical_episodes'],
                    **stats['event_counts'], success_rate=stats['success_rate'],
                    boundary_rate=stats['boundary_rate'], timeout_rate=stats['timeout_rate'],
                    boundary_subtypes=stats['boundary_subtypes'],
                    mean_reward=stats['moments']['reward']['mean'],
                    reward_components=stats['reward_components'],
                    mean_minimum_goal_distance_m=(stats['additional_moments']
                                                  ['minimum_goal_distance_m']['mean']),
                    minimum_goal_distance_measured_n=(stats['additional_moments']
                                                      ['minimum_goal_distance_m']['n']),
                    near_goal_10m_episode_count=stats['near_goal_10m_episode_count'],
                    success_missing_goal_entry_diagnostic=stats[
                        'success_missing_goal_entry_diagnostic']))
        for seed in SEEDS:
            for stop in (25000, 50000, 75000, 100000):
                rows = [row for row in data['episodes'] if row['training_seed'] == seed
                        and bin_stop(row['last_transition']) == stop]
                train_bins.append(dict(group=group, training_seed=seed, bin_last=stop,
                                       episode_bin_scope='whole episode by final transition',
                                       **episode_details(rows)))
        update_bins.extend(data['update_bins'])
    aggregated = {}
    for group in ('C_3e-4', 'R1_1e-4'):
        rows = [row for row in endpoint_rows if row['group'] == group]
        aggregated[group] = {field: moment([row[field] for row in rows])
                             for field in ('success_rate', 'boundary_rate', 'timeout_rate',
                                           'mean_reward')}
    paired_changes = []
    for seed in SEEDS:
        before = next(row for row in endpoint_rows if row['group'] == 'C_3e-4'
                      and row['training_seed'] == seed)
        after = next(row for row in endpoint_rows if row['group'] == 'R1_1e-4'
                     and row['training_seed'] == seed)
        paired_changes.append(dict(
            training_seed=seed,
            success_rate_change=after['success_rate']-before['success_rate'],
            boundary_rate_change=after['boundary_rate']-before['boundary_rate'],
            timeout_rate_change=after['timeout_rate']-before['timeout_rate'],
            reward_change=after['mean_reward']-before['mean_reward']))
    public = TASK/'public_evidence'
    public.mkdir(exist_ok=True)
    csv_write(public/'training_episodes.csv', c['public_training']+r['public_training'])
    csv_write(public/'validation_episodes.csv', c['public_validation']+r['public_validation'])
    csv_write(public/'selected_validation_trajectories.csv.gz',
              c['trajectory_rows']+r['trajectory_rows'], compressed=True)
    output = TASK/'analysis'
    output.mkdir(exist_ok=True)
    csv_write(output/'endpoint_comparison.csv', endpoint_rows)
    flat_points = []
    for row in validation_points:
        for scope in ('all_registered', 'paired_monitor30'):
            stats = row[scope]
            flat_points.append(dict(
                group=row['group'], training_seed=row['training_seed'],
                at_transition=row['at_transition'], evaluation_kind=row['evaluation_kind'],
                scope=scope, episodes=stats['complete_physical_episodes'],
                **stats['event_counts'], success_rate=stats['success_rate'],
                collision_rate=stats['collision_rate'], boundary_rate=stats['boundary_rate'],
                timeout_rate=stats['timeout_rate'],
                mean_reward=stats['moments']['reward']['mean'],
                mean_minimum_goal_distance_m=(stats['additional_moments']
                                              ['minimum_goal_distance_m']['mean']),
                boundary_subtypes=stats['boundary_subtypes']))
    csv_write(output/'validation_points.csv', flat_points)
    csv_write(output/'training_episode_bins.csv', train_bins)
    csv_write(output/'update_bins.csv', update_bins)
    csv_write(output/'paired_endpoint_changes.csv', paired_changes)
    model_inventory = []
    for seed in SEEDS:
        for point in (25000, 50000, 75000, 100000):
            path = TASK/'models'/f'seed_{seed}_{point}.pt'
            model_inventory.append(dict(seed=seed, transition=point,
                                        path=path.relative_to(ROOT).as_posix(),
                                        exists=path.is_file(),
                                        purpose='Frozen diagnostic snapshot; no best selection'))
    return dict(
        status='R1_BATCH_DATA_COMPLETE', generated_at=datetime.now(UTC).isoformat(),
        registration_id=REGISTRATION_ID, branch='L2_COMMON_LEARNING_RATE', repair_round=1,
        control_reused=True, control_experiment_commit=CONTROL_COMMIT,
        experiment_code_commit=state['experiment_code_commit'],
        changed_scientific_parameter_only='common Adam learning_rate: 3e-4 -> 1e-4',
        seed_counts=r['seed_counts'], control_seed_counts=c['seed_counts'],
        actual_new_scientific_training_transitions=sum(row['transitions']
                                                      for row in r['seed_counts'].values()),
        actual_new_complete_sac_updates=sum(row['complete_sac_updates']
                                           for row in r['seed_counts'].values()),
        actual_new_optimizer_steps=sum(row['optimizer_steps']
                                       for row in r['seed_counts'].values()),
        reused_control_training_transitions=300000, reused_control_complete_sac_updates=270003,
        endpoint_comparison=endpoint_rows, across_training_seed_mean_sample_sd=aggregated,
        paired_endpoint_changes=paired_changes,
        validation_points=validation_points, training_episode_bins=train_bins,
        update_bins=update_bins,
        raw_log_confirmation=r['source_diagnostics'], control_confirmation=c['source_diagnostics'],
        fixed_validation_direct_identity_comparisons=identity_pairs,
        fixed_learned_diagnostics=diagnostics, small_model_inventory=model_inventory,
        actual_runtime_state=state,
        public_files=[str(path.relative_to(ROOT)) for path in public.iterdir() if path.is_file()],
        interpretation_boundaries=[
            'C and R1 are separate runs; no CV continuation or concatenated learning curve.',
            'Primary endpoint remains registered 100k; intermediate snapshots are diagnostic only.',
            'Val300/monitor30 are development validation, not independent Test-ID or OOD.',
            'Independent training N=3; episode counts do not create extra training replicates.',
            'C complete boundary subtype/minimum target distance were not historically logged.',
            'No parameter selection, new candidate, training extension or reward change.',
            'Only obstacle-free repair is tested; complete Stage2/CV success cannot be claimed.'],
        remaining_l2_rounds_subject_to_new_authorization=1,
        scientific_stage2_decision='REQUIRES_REVIEW_OF_ACTUAL_R1_OUTCOMES')


def plots(result: dict[str, Any], output: Path) -> list[str]:
    """只由真实结果绘图，无平滑、不截取最好点、不合并不同run。"""
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    files = []
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    metrics = [('success_rate', 'Success fraction'), ('boundary_rate', 'Boundary fraction'),
               ('timeout_rate', 'Timeout fraction'), ('reward', 'Undiscounted task reward'),
               ('minimum_goal_distance_m', 'Closest goal distance (m)'),
               ('near_goal_10m_episode_count', 'Count entering 10 m / 30')]
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for seed, color in zip(SEEDS, COLORS, strict=True):
            for group, linestyle in (('C_3e-4', '--'), ('R1_1e-4', '-')):
                points = sorted([row for row in result['validation_points']
                                 if row['training_seed'] == seed and row['group'] == group],
                                key=lambda row: row['at_transition'])
                ys = []
                for point in points:
                    stats = point['paired_monitor30']
                    if metric == 'reward':
                        value = stats['moments']['reward']['mean']
                    elif metric == 'minimum_goal_distance_m':
                        value = stats['additional_moments'][metric]['mean']
                    elif metric == 'near_goal_10m_episode_count':
                        measured = stats['goal_approach_measured_episodes']
                        value = stats[metric] if measured else None
                    else:
                        value = stats[metric]
                    ys.append(math.nan if value is None else value)
                axis.plot([point['at_transition']/1000 for point in points], ys,
                          linestyle=linestyle, marker='o', color=color,
                          label=f'{group} seed {seed}')
        axis.set_xlabel('Training transitions (thousands)')
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        if metric.endswith('_rate'):
            axis.set_ylim(-.02, 1.02)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('B0 obstacle-free R1 single-factor comparison | fixed monitor indices 0–29\n'
                   'N=30 at every plotted point; 100k paired subset of separate Val300. '
                   'Dashed C is historical; solid R1 is a new run.')
    name = 'paired_monitor30_learning_curves.svg'
    figure.savefig(output/name)
    figure.savefig(output/name.replace('.svg', '.png'), dpi=150)
    plt.close(figure)
    files.append(name)
    figure, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    for axis, (field, label) in zip(axes, [('success_rate', 'Success fraction'),
                                         ('boundary_rate', 'Boundary fraction'),
                                         ('timeout_rate', 'Timeout fraction'),
                                         ('mean_reward', 'Mean task reward')], strict=True):
        for group, marker in (('C_3e-4', 'o'), ('R1_1e-4', 's')):
            rows = sorted([row for row in result['endpoint_comparison'] if row['group'] == group],
                          key=lambda row: row['training_seed'])
            axis.plot(SEEDS, [row[field] for row in rows], marker=marker, label=group)
        axis.set_xticks(SEEDS)
        axis.set_xlabel('Independent training seed')
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        if field.endswith('_rate'):
            axis.set_ylim(-.02, 1.02)
    axes[0].legend()
    figure.suptitle('Fixed 100k obstacle-free endpoint | each raw seed uses Val300 N=300\n'
                   'No best-checkpoint selection; not independent Test-ID.')
    name = 'fixed_100k_val300_raw_seeds.svg'
    figure.savefig(output/name)
    figure.savefig(output/name.replace('.svg', '.png'), dpi=150)
    plt.close(figure)
    files.append(name)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, field in zip(axes.ravel(), ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss',
                                        'alpha', 'actor_gradient_norm'), strict=True):
        for seed, color in zip(SEEDS, COLORS, strict=True):
            for group, style in (('C_3e-4', '--'), ('R1_1e-4', '-')):
                rows = sorted([row for row in result['update_bins']
                               if row['group'] == group and row['training_seed'] == seed],
                              key=lambda row: row['bin_last'])
                axis.plot([row['bin_last']/1000 for row in rows],
                          [row['metrics'][field]['mean'] for row in rows],
                          linestyle=style, marker='o', color=color, label=f'{group} seed {seed}')
        axis.set_xlabel('End of update bin (thousands of training transitions)')
        axis.set_ylabel(f'{field}: actual bin mean')
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('C / R1 actual confirmed update diagnostics, 25k bins\n'
                   'Finite losses and gradients are numerical evidence, not learned-task success.')
    name = 'update_diagnostics_comparison.svg'
    figure.savefig(output/name)
    figure.savefig(output/name.replace('.svg', '.png'), dpi=150)
    plt.close(figure)
    files.append(name)
    return files


def main() -> int:
    """默认完整只读汇总；plot-only可使用已存在Matplotlib运行时且不导入Torch。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-final-diagnostics', action='store_true')
    parser.add_argument('--trusted-local', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--analysis-json', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.plot_only:
        if args.execute_final_diagnostics or args.analysis_json is None or args.output_dir is None:
            raise ValueError('plot-only必须明确已有结果JSON/输出，不执行环境诊断。')
        result = json.loads(args.analysis_json.read_text(encoding='utf-8'))
        files = plots(result, args.output_dir)
        print(json.dumps(dict(status='PLOTS_WRITTEN_FROM_ACTUAL_LOGS', figures=files)))
        return 0
    state = validate_completed_state()
    diagnostics = None
    path = TASK/'analysis'/'final_fixed_diagnostics.json'
    if args.execute_final_diagnostics:
        diagnostics = run_final_cases(state, args.trusted_local)
    elif path.is_file():
        diagnostics = json.loads(path.read_text(encoding='utf-8'))
    result = analyze(state, diagnostics)
    output = TASK/'analysis'/'r1_analysis.json'
    json_write(output, result)
    print(json.dumps(dict(status=result['status'],
                          actual_new_training_transitions=result[
                              'actual_new_scientific_training_transitions'],
                          actual_new_complete_sac_updates=result['actual_new_complete_sac_updates'],
                          endpoint_comparison=result['endpoint_comparison'],
                          learned_fixed_diagnostics=(None if diagnostics is None else
                                                     dict(count=len(diagnostics['cases']),
                                                          successes=diagnostics['successes'],
                                                          transitions=diagnostics[
                                                              'diagnostic_control_transitions']))),
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
