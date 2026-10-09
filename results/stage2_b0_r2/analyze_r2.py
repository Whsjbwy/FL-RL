"""R2只读有效日志汇总；部分结果不冒充完成，不运行策略、重放或学习。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
REGISTRATION_ID = 'STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET'
METHOD = 'B0_FULL_STATE_ORDINARY_SAC'
GROUPS = {'C300': 100.0, 'G200': 200.0}
SEEDS = (11, 22, 33)
POINTS = tuple(range(0, 300001, 25000))
FULL_POINTS = (100000, 300000)
COMPONENTS = ('progress', 'goal', 'time', 'smoothness')
UPDATE_FIELDS = ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss', 'alpha',
                 'actor_gradient_norm', 'q1_gradient_norm', 'q2_gradient_norm',
                 'alpha_gradient_norm')
EVENTS = ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')
DISTANCE_FIELDS = ('initial_distance_m', 'final_distance_m', 'minimum_goal_distance_m',
                   'minimum_goal_distance_time_s', 'first_within_10m_time_s',
                   'first_within_5m_time_s', 'first_within_3m_time_s', 'first_goal_entry_time_s')
COLORS = {'C300': '#0072B2', 'G200': '#D55E00'}
TRAINING_COMMAND_NAME = 'r2_scientific_batch'
TRAINING_COMMAND = [
    '.venv-b1/Scripts/python.exe', 'scripts/run_b0_training.py',
    '--r2-registration', 'configs/stage2_b0_r2.yaml', '--execute',
    '--run-kind', 'scientific_training', '--transition-budget', '1800000',
    '--output-dir', 'results/stage2_b0_r2',
]


def read_json(path: Path) -> Any:
    """只读原始JSON；损坏记录不被默认值掩盖。"""
    return json.loads(path.read_text(encoding='utf-8'))


def json_write(path: Path, value: Any) -> None:
    """只覆盖本工具派生输出，拒绝非有限JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def require(condition: bool, message: str) -> None:
    """身份/分母不合法时直接停止，不能用部分记录制造完整终点。"""
    if not condition:
        raise ValueError(message)


def moment(values: Iterable[float]) -> dict[str, Any]:
    """沿用在线n-1样本SD；episode和独立训练seed的统计层级分开。"""
    from auv_risk_rl.training.mvp_analysis import Moments

    result = Moments()
    for value in values:
        result.add(value)
    return result.result()


def csv_value(value: Any) -> Any:
    """CSV中的结构字段保留为真实JSON，null写空单元格。"""
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


class RowWriter:
    """流式保存紧凑原始标量和保留轨迹，不把全部更新/观察堆入内存。"""

    def __init__(self, path: Path, fields: Iterable[str], *, compressed: bool = False) -> None:
        """即使部分结果暂无行也写明表头，不伪造零样本结果。"""
        self.path = path
        self.fields = list(fields)
        self.count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = (gzip.open(path, 'wt', newline='', encoding='utf-8') if compressed
                       else path.open('w', newline='', encoding='utf-8'))
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
        self.writer.writeheader()

    def add(self, row: dict[str, Any]) -> None:
        """拒绝意外丢字段；每条导出行都有明确来源与任务身份。"""
        require(not set(row).difference(self.fields), '导出字段没有登记到表头。')
        self.writer.writerow({key: csv_value(value) for key, value in row.items()})
        self.count += 1

    def close(self) -> None:
        """派生文件写完即关闭，原始训练日志始终只读。"""
        self.stream.close()


EPISODE_FIELDS = (
    'group', 'w_goal', 'scope', 'training_seed', 'task_profile', 'scenario_id',
    'scenario_index', 'scenario_root_seed', 'environment_seed', 'base_scenario_id',
    'actual_scenario_id', 'episode_id', 'env_slot', 'split', 'start_transition',
    'last_transition', 'at_transition', 'evaluation_kind', 'evaluation_key', 'steps',
    'warmup_root_seed', 'warmup_duration_s', 'warmup_control_transitions',
    'physical_time_s', 'reward', 'common100utility', 'reward_progress', 'reward_goal',
    'reward_time', 'reward_smoothness', 'path_length_m', 'path_length_definition',
    'minimum_clearance_m', 'action_saturation_count', 'failure_type', 'complete',
    'success', 'collision', 'boundary', 'task_timeout', 'external_truncation',
    'phase_boundary', 'budget_stop', 'boundary_subtype', 'boundary_constraints',
    *DISTANCE_FIELDS, 'diagnostic_scope', 'registration_id', 'code_version', 'run_kind',
    'method', 'segment_id', 'log_sequence', 'source_path', 'source_line', 'valid_log_sequence',
)
TRAJECTORY_FIELDS = (
    'group', 'training_seed', 'at_transition', 'evaluation_kind', 'evaluation_key',
    'scenario_index', 'scenario_id', 'failure_type', 'trajectory_retention_reasons',
    'task_step', 'physical_time_s', 'state_basis', 'north_m', 'east_m', 'down_m',
    'yaw_rad', 'pitch_rad', 'surge_speed_mps', 'yaw_rate_rad_s', 'pitch_rate_rad_s',
    'goal_north_m', 'goal_east_m', 'goal_down_m', 'goal_distance_m',
    'action_surge', 'action_yaw', 'action_pitch', 'physical_command',
    'reward', 'reward_progress', 'reward_goal', 'reward_time', 'reward_smoothness',
)


def public_episode(episode: dict[str, Any], group: str,
                   validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """原始奖励与离线共同goal100效用并列，不改写任何实际学习回报。"""
    row = {key: episode.get(key) for key in EPISODE_FIELDS}
    source = episode.get('_log_source', {}) if validation is None else validation['_log_source']
    weight = GROUPS[group]
    row.update(group=group, w_goal=weight, scope='train' if validation is None else 'validation',
               common100utility=episode['reward']-(weight-100)*int(episode['success']),
               **{f'reward_{name}': episode['reward_components'][name] for name in COMPONENTS},
               source_path=f'{group}/{source.get("path")}',
               source_line=source.get('line_number'),
               valid_log_sequence=source.get('valid_log_sequence'))
    warmup = episode.get('warmup', {})
    duration = warmup.get('warmup_duration_s')
    row.update(warmup_root_seed=warmup.get('root_seed'), warmup_duration_s=duration,
               warmup_control_transitions=None if duration is None else round(duration/0.2))
    if validation is None and row['environment_seed'] is None:
        row['environment_seed'] = warmup.get('root_seed')
    if validation is not None:
        for key in ('at_transition', 'evaluation_key', 'registration_id', 'code_version',
                    'run_kind', 'method', 'segment_id', 'log_sequence'):
            row[key] = validation[key]
        row['evaluation_kind'] = 'Val300' if validation['full'] else 'monitor30'
        row['scenario_index'] = episode['index']
    return row


def episode_stats(episodes: list[dict[str, Any]], weight: float) -> dict[str, Any]:
    """完整物理episode为事件分母；预算片段不混入SR/timeout或奖励均值。"""
    from auv_risk_rl.training.mvp_analysis import summarize_episodes

    result = summarize_episodes(episodes)
    complete = [row for row in episodes if row['complete']]
    boundary = [row for row in complete if row['failure_type'] == 'operational_boundary_failure']
    result.update(
        boundary_subtypes=dict(Counter(row.get('boundary_subtype') or 'NOT_RECORDED'
                                       for row in boundary)),
        reward_components={name: moment(row['reward_components'][name] for row in complete)
                           for name in COMPONENTS},
        common100utility=moment(row['reward']-(weight-100)*int(row['success'])
                                for row in complete),
        goal_diagnostics={name: moment(float(row[name]) for row in complete
                                       if row.get(name) is not None) for name in DISTANCE_FIELDS},
        complete_episode_transitions=sum(row['steps'] for row in complete),
        successful_episode_transitions=sum(row['steps'] for row in complete if row['success']),
        successful_terminal_transitions=sum(row['success'] for row in complete),
    )
    for field, label in (('first_within_10m_time_s', '10m'), ('first_within_5m_time_s', '5m'),
                         ('first_within_3m_time_s', '3m'), ('first_goal_entry_time_s', '2m')):
        measured = sum(field in row for row in complete)
        result[f'capture_{label}_measured_episodes'] = measured
        result[f'capture_{label}_episodes'] = (sum(row.get(field) is not None for row in complete)
                                              if measured else None)
    measured = result['capture_2m_measured_episodes']
    result.update(
        success_missing_goal_entry=(sum(row['success'] and row.get('first_goal_entry_time_s')
                                        is None for row in complete) if measured else None),
        goal_entry_without_success=(sum(not row['success'] and row.get('first_goal_entry_time_s')
                                        is not None for row in complete) if measured else None),
        reward_component_sum_max_abs_error=max((abs(row['reward']-sum(
            row['reward_components'][name] for name in COMPONENTS)) for row in complete),
                                               default=None),
        goal_component_max_abs_error=max((abs(row['reward_components']['goal']
                                              - weight*int(row['success'])) for row in complete),
                                           default=None),
    )
    return result


def flat_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """统一点/分箱/终点字段，未知量继续null，率保持0..1。"""
    row = {key: stats[key] for key in ('complete_physical_episodes', 'fragments',
           'logged_training_steps', 'success_rate', 'collision_rate', 'boundary_rate',
           'timeout_rate', 'action_saturation_transition_rate', 'boundary_subtypes',
           'capture_10m_episodes', 'capture_5m_episodes', 'capture_3m_episodes',
           'capture_2m_episodes', 'success_missing_goal_entry', 'goal_entry_without_success')}
    for event in EVENTS:
        row[f'count_{event}'] = stats['event_counts'][event]
    for key in ('reward', 'travel_time_s', 'penalized_time_s', 'path_length_m',
                'successful_path_length_m'):
        row[f'{key}_mean'] = stats['moments'][key]['mean']
    row['common100utility_mean'] = stats['common100utility']['mean']
    for name in COMPONENTS:
        row[f'reward_{name}_mean'] = stats['reward_components'][name]['mean']
    for name in DISTANCE_FIELDS:
        row[f'{name}_mean'] = stats['goal_diagnostics'][name]['mean']
        row[f'{name}_observed_n'] = stats['goal_diagnostics'][name]['n']
    return row


def identity(episode: dict[str, Any]) -> dict[str, Any]:
    """按实际ID、噪声种子和几何直接比较，绝不生成摘要门禁。"""
    keys = ('index', 'base_scenario_id', 'actual_scenario_id', 'environment_seed',
            'initial_position_ned_m', 'goal_position_ned_m')
    return {key: episode[key] for key in keys}


def retained_trajectory(row: dict[str, Any], episode: dict[str, Any], group: str,
                        writer: RowWriter) -> dict[str, Any] | None:
    """只导出预先保留的实际控制节点；最小距离插值状态不冒充实际积分节点。"""
    nodes = episode.get('r2_state_trajectory', episode.get('trajectory'))
    if nodes is None:
        return None
    reasons = episode.get('trajectory_retention_reasons', [])
    require(bool(reasons), '保留轨迹缺少预登记选样依据。')
    allowed = {'preregistered_index', 'earliest_failure_by_index', 'earliest_success_by_index'}
    require(set(reasons).issubset(allowed), '轨迹包含未登记选样理由。')
    require(len(nodes) == episode['steps'], '保留轨迹节点数不等于实际控制transition数。')
    initial = episode.get('formal_initial_state')
    before = writer.count
    common = dict(group=group, training_seed=row['training_seed'],
                  at_transition=row['at_transition'],
                  evaluation_kind='Val300' if row['full'] else 'monitor30',
                  evaluation_key=row['evaluation_key'], scenario_index=episode['index'],
                  scenario_id=episode['scenario_id'], failure_type=episode['failure_type'],
                  trajectory_retention_reasons=reasons,
                  **dict(zip(('goal_north_m', 'goal_east_m', 'goal_down_m'),
                             episode['goal_position_ned_m'], strict=True)))
    elapsed = 0.0
    for number, node in enumerate(([initial] if initial is not None else [])+nodes):
        if node is None:
            continue
        if 'physical_time_s' in node:
            elapsed = node['physical_time_s']
        else:
            elapsed += node.get('elapsed_s', 0.0)
        position = node['position_ned_m']
        output = dict(common, task_step=node.get('task_step', number), physical_time_s=elapsed,
                      state_basis=node.get('state_basis', 'ACTUAL_CONTROL_NODE_POSITION_ONLY'),
                      north_m=position[0], east_m=position[1], down_m=position[2])
        for name in ('yaw_rad', 'pitch_rad', 'surge_speed_mps', 'yaw_rate_rad_s',
                     'pitch_rate_rad_s', 'goal_distance_m', 'physical_command', 'reward'):
            output[name] = node.get(name)
        action = node.get('action')
        for index, name in enumerate(('action_surge', 'action_yaw', 'action_pitch')):
            output[name] = None if action is None else action[index]
        parts = node.get('reward_components', {})
        output.update({f'reward_{name}': parts.get(name) for name in COMPONENTS})
        writer.add(output)
    return dict(common, rows=writer.count-before,
                minimum_goal_state=episode.get('minimum_goal_state'))


def expected_points() -> set[tuple[int, bool]]:
    """13个monitor及两个额外full，不能用同点的一个评价覆盖另一个。"""
    return {(point, False) for point in POINTS} | {(point, True) for point in FULL_POINTS}


def completion_gate(root: Path, state: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    """最终分析须真实worker及启动器退出0；partial不做任何完成推断。"""
    if partial:
        return dict(status='PARTIAL_ONLY', batch_state=state.get('status'),
                    note='Only checkpoint-confirmed log prefixes; worker may still be running.')
    order = [f'{group}_seed{seed}' for group in GROUPS for seed in SEEDS]
    require(state.get('status') == 'COMPLETED' and state.get('completed_runs') == order,
            '批次未真实完成六run；使用--partial查看已确认前缀。')
    actual_path = root/'actual_training_worker_exit.json'
    launcher_path = root/f'{TRAINING_COMMAND_NAME}.worker.json'
    actual, launcher = read_json(actual_path), read_json(launcher_path)
    require(actual.get('status') == 'EXITED'
            and actual.get('actual_computation_pid') == state.get('pid')
            and actual.get('actual_worker_exit_code') == 0, '真实训练worker尚未退出0。')
    require(launcher.get('status') == 'EXITED' and launcher.get('direct_child_exit_code') == 0
            and launcher.get('code_commit') == state['experiment_code_commit']
            and launcher.get('command') == TRAINING_COMMAND,
            '实际启动器退出/代码/执行命令与R2批次不一致。')
    commands = [json.loads(line) for line in (root/'commands.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    matches = [record for record in commands if record.get('name') == TRAINING_COMMAND_NAME]
    require(len(matches) == 1, '训练命令完成记录缺失或重复；拒绝混合多个attempt。')
    command = matches[0]
    for key in ('direct_child_pid', 'code_commit', 'command', 'start_utc'):
        require(command.get(key) == launcher.get(key), f'训练命令回执{key}不一致。')
    require(command.get('exit_code') == 0, '实际训练命令退出码不是0。')
    stamps = [datetime.fromisoformat(value) for value in
              (launcher['start_utc'], launcher['end_utc'], command['end_utc'])]
    require(all(value.tzinfo is not None for value in stamps)
            and stamps[0] <= stamps[1] <= stamps[2], '实际记录器时间写入顺序错误。')
    return dict(status='ACTUAL_COMPLETION_CONFIRMED', actual_worker=actual, launcher=launcher,
                command=command, timestamp_rule='start <= launcher end <= commands end')


def collect_run(root: Path, group: str, seed: int, code: str, *, partial: bool,
                writers: dict[str, RowWriter], canonical: dict[int, dict[str, Any]],
                ) -> dict[str, Any]:
    """流式读取确认前缀，实时state可超前但不能补造尚未确认的episode。"""
    from auv_risk_rl.training.mvp_analysis import Moments, confirmed_records, validation_point

    diagnostics: dict[str, Any] = {}
    episodes, points, trajectories, strata = [], [], [], []
    update_bins: dict[int, dict[str, Any]] = {}
    episode_bins: dict[int, list[dict[str, Any]]] = {}
    updates = max_confirmed = validation_transitions = warmup = 0
    observed: set[tuple[int, bool]] = set()
    config_expected = dict(w_progress=1.0, w_goal=GROUPS[group], w_time=0.01,
                           w_smooth=0.02, d_C=0.05)
    for kind, row in confirmed_records(root/group, seed, code, diagnostics,
                                       registration_id=REGISTRATION_ID):
        require(row.get('group') == group, f'{group}/seed{seed}存在非登记组日志。')
        if kind != 'update':
            require(row.get('task_profile') == 'obstacle_free',
                    f'{group}/seed{seed}存在非登记profile日志。')
        task = row.get('task_config', row.get('reward_config'))
        require(task is not None and all(task.get(key) == value
                                         for key, value in config_expected.items()),
                f'{group}/seed{seed}实际reward/task配置不匹配。')
        if kind == 'update':
            transition = row['transition']
            require(transition == 10000+updates and transition <= 300000,
                    f'{group}/seed{seed}完整update并非10000起连续一次/transition。')
            updates += 1
            max_confirmed = max(max_confirmed, transition)
            stop = ((transition-1)//25000+1)*25000
            bucket = update_bins.setdefault(stop, {name: Moments() for name in UPDATE_FIELDS})
            for name in UPDATE_FIELDS:
                bucket[name].add(row['metrics'][name])
            bucket['last_alpha'] = row['metrics']['alpha']
        elif kind == 'episode':
            require(0 <= row['last_transition'] <= 300000, '训练episode超出授权预算。')
            episodes.append(row)
            max_confirmed = max(max_confirmed, row['at_transition'])
            stop = ((max(1, row['at_transition'])-1)//25000+1)*25000
            episode_bins.setdefault(stop, []).append(row)
            writers['episodes'].add(public_episode(row, group))
        else:
            validation_point(row)
            key = row['at_transition'], row['full']
            require(key in expected_points() and key not in observed, '固定验证点重复或未登记。')
            require(row.get('r2_diagnostic_state_unchanged') is True,
                    'R2验证缺少直接训练状态隔离证据。')
            observed.add(key)
            max_confirmed = max(max_confirmed, row['at_transition'])
            validation_transitions += row['evaluation_env_transitions']
            warmup += row['warmup_control_transitions']
            first_failure = next((ep['index'] for ep in row['episodes']
                                  if not ep['success']), None)
            first_success = next((ep['index'] for ep in row['episodes'] if ep['success']), None)
            selected_indices = {0, 1, 2} | {index for index in (first_failure, first_success)
                                             if index is not None}
            actual_indices = {ep['index'] for ep in row['episodes']
                              if 'trajectory' in ep or 'r2_state_trajectory' in ep}
            require(selected_indices == actual_indices, '实际保留轨迹不符合固定ID/最早事件选样。')
            for episode in row['episodes']:
                require(episode['steps'] <= 1000 and not episode.get('external_truncation'),
                        '正式固定验证混入工程截断或超出任务时域。')
                index, current = episode['index'], identity(episode)
                if index in canonical:
                    require(current == canonical[index], '跨组/seed/检查点固定验证基底不一致。')
                else:
                    canonical[index] = current
                writers['episodes'].add(public_episode(episode, group, row))
                trajectory = retained_trajectory(row, episode, group, writers['trajectories'])
                if trajectory is not None:
                    trajectories.append(trajectory)
            stats = episode_stats(row['episodes'], GROUPS[group])
            point = dict(group=group, training_seed=seed, w_goal=GROUPS[group],
                         profile='obstacle_free', at_transition=row['at_transition'],
                         evaluation_kind='Val300' if row['full'] else 'monitor30',
                         evaluation_key=row['evaluation_key'], count=row['count'],
                         full=row['full'], stats=stats,
                         validation_root_seed=row['validation_root_seed'],
                         source_path=f'{group}/{row["_log_source"]["path"]}',
                         source_line=row['_log_source']['line_number'])
            points.append(point)
            for event in EVENTS:
                selected = [ep for ep in row['episodes'] if ep['failure_type'] == event]
                strata.append(dict(group=group, training_seed=seed, scope='validation',
                                   at_transition=row['at_transition'],
                                   evaluation_kind=point['evaluation_kind'], failure_type=event,
                                   **flat_metrics(episode_stats(selected, GROUPS[group]))))
    logged_steps = sum(row['steps'] for row in episodes)
    if not partial:
        require(observed == expected_points(), f'{group}/seed{seed}没有完整15点固定验证。')
        require(logged_steps == 300000 and updates == 290001,
                f'{group}/seed{seed}实际episode步数/update数不符。')
    training_bins = []
    for stop, rows in sorted(episode_bins.items()):
        training_bins.append(dict(group=group, training_seed=seed, bin_first=stop-24999,
                                  bin_last=stop,
                                  **flat_metrics(episode_stats(rows, GROUPS[group]))))
        for event in EVENTS:
            selected = [ep for ep in rows if ep['complete'] and ep['failure_type'] == event]
            strata.append(dict(group=group, training_seed=seed, scope='train',
                               at_transition=stop, evaluation_kind=None, failure_type=event,
                               **flat_metrics(episode_stats(selected, GROUPS[group]))))
    update_rows = [dict(group=group, training_seed=seed, bin_first=stop-24999, bin_last=stop,
                        complete_sac_updates=values['alpha'].count,
                        optimizer_steps=4*values['alpha'].count, last_alpha=values['last_alpha'],
                        metrics={name: values[name].result() for name in UPDATE_FIELDS})
                   for stop, values in sorted(update_bins.items())]
    return dict(group=group, training_seed=seed, max_confirmed_transition=max_confirmed,
                logged_episode_transitions=logged_steps, confirmed_complete_sac_updates=updates,
                confirmed_logged_episode_warmup_control_transitions=sum(round(
                    row['warmup']['warmup_duration_s']/0.2) for row in episodes
                    if row.get('warmup', {}).get('warmup_duration_s') is not None),
                confirmed_optimizer_steps=4*updates,
                evaluation_env_transitions=validation_transitions,
                evaluation_warmup_control_transitions=warmup,
                observed_validation_points=len(points),
                missing_validation_points=[dict(at_transition=step, full=full)
                                           for step, full in sorted(expected_points()-observed)],
                training_summary=episode_stats(episodes, GROUPS[group]),
                training_bins=training_bins, update_bins=update_rows, validation_points=points,
                termination_strata=strata, retained_trajectories=trajectories,
                segment_diagnostics=diagnostics)


def point_row(point: dict[str, Any]) -> dict[str, Any]:
    """monitor30及Val300均原样保留，额外full不能替换monitor曲线点。"""
    keys = ('group', 'training_seed', 'w_goal', 'profile', 'at_transition',
            'evaluation_kind', 'evaluation_key', 'count', 'full', 'validation_root_seed')
    return {**{key: point[key] for key in keys}, **flat_metrics(point['stats'])}


def aggregate_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只在同组/同点/同规模内跨seed，明确实际n和预期三个独立训练重复。"""
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for point in points:
        groups.setdefault((point['group'], point['at_transition'],
                           point['evaluation_kind']), []).append(point_row(point))
    result = []
    for (group, transition, kind), rows in sorted(groups.items()):
        seeds = [row['training_seed'] for row in rows]
        require(len(seeds) == len(set(seeds)), '跨seed统计出现重复训练seed。')
        metrics = flat_metrics(points[0]['stats']) if points else {}
        for metric in metrics:
            if isinstance(rows[0][metric], dict):
                continue
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            result.append(dict(group=group, at_transition=transition, evaluation_kind=kind,
                               metric=metric, training_seeds=sorted(seeds),
                               expected_training_seed_count=3, **moment(values)))
    return result


def endpoint_pairs(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """100k与300k成对比较；原reward分别报告，共同效用只离线减去到达权重差。"""
    index = {(row['group'], row['training_seed'], row['at_transition']): point_row(row)
             for row in points if row['full']}
    result = []
    fields = ('success_rate', 'boundary_rate', 'timeout_rate', 'reward_mean',
              'common100utility_mean', 'minimum_goal_distance_m_mean', 'travel_time_s_mean',
              'action_saturation_transition_rate')
    for seed in SEEDS:
        for transition in FULL_POINTS:
            c, g = index.get(('C300', seed, transition)), index.get(('G200', seed, transition))
            if c is None or g is None:
                continue
            row = dict(training_seed=seed, at_transition=transition,
                       c_complete=c['complete_physical_episodes'],
                       g_complete=g['complete_physical_episodes'],
                       direct_validation_identity_equal=True)
            for name in fields:
                row[f'C300_{name}'], row[f'G200_{name}'] = c[name], g[name]
                row[f'G200_minus_C300_{name}'] = (g[name]-c[name]
                                                if c[name] is not None and g[name] is not None
                                                else None)
            result.append(row)
    return result


def budget_comparison(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同run100k至300k固定Val比较，不把它误写成独立训练重复。"""
    index = {(row['group'], row['training_seed'], row['at_transition']): point_row(row)
             for row in points if row['full']}
    metrics = ('success_rate', 'boundary_rate', 'timeout_rate', 'common100utility_mean',
               'reward_mean', 'minimum_goal_distance_m_mean', 'travel_time_s_mean')
    result = []
    for group in GROUPS:
        for seed in SEEDS:
            start, end = index.get((group, seed, 100000)), index.get((group, seed, 300000))
            if start is None or end is None:
                continue
            row = dict(group=group, training_seed=seed, start=100000, end=300000,
                       evaluation_kind='Val300', same_run_continued_without_cv=True)
            for name in metrics:
                row[f'at100k_{name}'], row[f'at300k_{name}'] = start[name], end[name]
                row[f'change_{name}'] = (end[name]-start[name] if start[name] is not None
                                        and end[name] is not None else None)
            result.append(row)
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """动态摘要表为空时写明无确认结果，不建立虚构数据行。"""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        path.write_text('NO_CONFIRMED_ROWS\n', encoding='utf-8')
        return
    writer = RowWriter(path, fields)
    try:
        for row in rows:
            writer.add(row)
    finally:
        writer.close()


def analyze(root: Path, *, partial: bool) -> dict[str, Any]:
    """真实运行后才生成数值；部分与最终输出目录及结果身份严格分开。"""
    from auv_risk_rl.training.r2_registration import APPROVED

    state = read_json(root/'batch_state.json')
    require(state.get('registration_id') == REGISTRATION_ID, 'batch_state不是本R2登记。')
    code = state.get('experiment_code_commit')
    require(isinstance(code, str) and len(code) == 40, '实际实验Git提交缺失。')
    require(state.get('registration') == APPROVED, '批次完整登记不匹配冻结R2配置。')
    gate = completion_gate(root, state, partial=partial)
    output = root/('partial_analysis' if partial else 'analysis')
    output.mkdir(parents=True, exist_ok=True)
    writers = dict(episodes=RowWriter(output/'all_episodes.csv.gz', EPISODE_FIELDS,
                                      compressed=True),
                   trajectories=RowWriter(output/'retained_trajectories.csv.gz',
                                          TRAJECTORY_FIELDS, compressed=True))
    canonical: dict[int, dict[str, Any]] = {}
    runs = []
    try:
        for group in GROUPS:
            for seed in SEEDS:
                result = collect_run(root, group, seed, code, partial=partial,
                                     writers=writers, canonical=canonical)
                result['runtime_record'] = state.get('runs', {}).get(f'{group}_seed{seed}')
                if not partial:
                    runtime = result['runtime_record'] or {}
                    require(runtime.get('status') == 'COMPLETED'
                            and runtime.get('transitions') == 300000
                            and runtime.get('updates') == 290001
                            and runtime.get('optimizer_steps') == 1160004,
                            '实际run运行计数与确认日志/冻结语义不符。')
                    require(runtime['evaluation_env_transitions']
                            == result['evaluation_env_transitions']
                            and runtime['evaluation_warmup_control_transitions']
                            == result['evaluation_warmup_control_transitions'],
                            '验证transition/warm-up计数与实际日志不符。')
                    require(runtime['training_warmup_transitions']
                            == result['confirmed_logged_episode_warmup_control_transitions'],
                            '实际训练合法warm-up计数与所有episode/片段记录不符。')
                runs.append(result)
    finally:
        for writer in writers.values():
            writer.close()
    points = [point for run in runs for point in run['validation_points']]
    aggregates = aggregate_points(points)
    flat_points = [point_row(point) for point in points]
    endpoints = [row for row in flat_points if row['full']]
    endpoint_aggregate = [row for row in aggregates if row['evaluation_kind'] == 'Val300']
    pairs = endpoint_pairs(points)
    budget_changes = budget_comparison(points)
    update_rows = []
    for run in runs:
        for row in run['update_bins']:
            flat = {key: value for key, value in row.items() if key != 'metrics'}
            for name, stats in row['metrics'].items():
                flat.update({f'{name}_{key}': value for key, value in stats.items()})
            update_rows.append(flat)
    write_rows(output/'points.csv', flat_points)
    write_rows(output/'points_seed_mean_sd.csv', aggregates)
    write_rows(output/'endpoints_by_seed.csv', endpoints)
    write_rows(output/'endpoints_seed_mean_sd.csv', endpoint_aggregate)
    write_rows(output/'endpoint_paired_comparison.csv', pairs)
    write_rows(output/'within_run_budget_comparison.csv', budget_changes)
    write_rows(output/'training_25k_bins.csv',
               [row for run in runs for row in run['training_bins']])
    write_rows(output/'update_25k_bins.csv', update_rows)
    write_rows(output/'termination_strata.csv', [row for run in runs
                                                for row in run['termination_strata']])
    trajectories = [row for run in runs for row in run['retained_trajectories']]
    json_write(output/'trajectory_inventory.json', trajectories)
    for run in runs:
        run.pop('retained_trajectories')
        run.pop('termination_strata')
        group, seed = run['group'], run['training_seed']
        run['small_model_inventory'] = [dict(
            transition=point, path=f'{group}/models/seed_{seed}_{point}.pt',
            exists=(root/group/'models'/f'seed_{seed}_{point}.pt').is_file(),
            bytes=(root/group/'models'/f'seed_{seed}_{point}.pt').stat().st_size
            if (root/group/'models'/f'seed_{seed}_{point}.pt').is_file() else None)
            for point in POINTS if point]
        if not partial:
            require(all(row['exists'] and row['bytes'] > 0
                        for row in run['small_model_inventory']),
                    f'{group}/seed{seed}缺失实际25k小模型。')
    result = dict(
        registration_id=REGISTRATION_ID, experiment_code_commit=code,
        created_at=datetime.now(UTC).isoformat(), batch_root=str(root.resolve()),
        status='PARTIAL_CONFIRMED_PREFIX_ONLY' if partial else 'BATCH_DATA_COMPLETE',
        batch_execution_status=state.get('status'), completion_gate=gate,
        stage2_scientific_decision='NOT_ASSIGNED_BY_ANALYSIS_TOOL_REQUIRES_REVIEW',
        task_profile='obstacle_free', groups=GROUPS, independent_training_seeds=list(SEEDS),
        runs=runs, validation_points=flat_points, seed_mean_sample_sd=aggregates,
        endpoint_pairs=pairs, within_run_budget_comparison=budget_changes, counts=dict(
            confirmed_complete_sac_updates=sum(run['confirmed_complete_sac_updates']
                                               for run in runs),
            confirmed_optimizer_steps=sum(run['confirmed_optimizer_steps'] for run in runs),
            confirmed_logged_episode_transitions=sum(run['logged_episode_transitions']
                                                     for run in runs),
            actual_runtime_training_transitions=state.get('actual_training_transitions'),
            actual_runtime_sac_updates=state.get('actual_sac_updates'),
            confirmed_validation_transitions=sum(run['evaluation_env_transitions'] for run in runs),
            confirmed_validation_warmup_control_transitions=sum(
                run['evaluation_warmup_control_transitions'] for run in runs),
            confirmed_logged_episode_warmup_control_transitions=sum(
                run['confirmed_logged_episode_warmup_control_transitions'] for run in runs),
            compact_episode_rows=writers['episodes'].count,
            retained_control_node_rows=writers['trajectories'].count,
            new_analysis_environment_transitions=0, new_analysis_updates=0),
        direct_fixed_validation_identity=dict(compared_indices=sorted(canonical),
                                              no_identity_disagreement_observed=True),
        definitions=dict(
            denominators='Only complete physical episodes; fragments separately recorded.',
            validation='13 monitor30 plus additional Val300 at100k/300k; no Test/OOD.',
            common100utility='raw return-(actual w_goal-100)*int(success), offline task utility.',
            training_bins=('Whole episode assigned by ending global transition; '
                           'not transition windows.'),
            update_bins='Exact confirmed updates assigned by global transition into25k windows.',
            seed_sd='n-1 sample SD across actual training seeds, never across episodes as seeds.',
            path_length='CONTROL_NODE_POLYLINE, original harness definition.',
            near_goal='Recorded executed0.05s segment diagnostics, not extra simulation.',
            unfinished='Partial confirmed prefixes may omit active episodes and unconfirmed tails.',
            checkpoint_selection='Fixed300k primary and100k development endpoint; never best.',
            observation='B0 current fullstate; no future/risk/execution filter.',
            unmeasured='null is unavailable/not applicable; zero is only an actual recorded zero.',
        ),
    )
    if not partial:
        counts = result['counts']
        require(counts['confirmed_logged_episode_transitions']
                == counts['actual_runtime_training_transitions'] == 1800000,
                '六run实际训练总步数不匹配确认日志。')
        require(counts['confirmed_complete_sac_updates']
                == counts['actual_runtime_sac_updates'] == 1740006,
                '六run实际完整SAC更新总数不匹配确认日志。')
    filename = 'R2_PARTIAL_RESULT.json' if partial else 'R2_RESULT.json'
    json_write(output/filename, result)
    write_report(output, result, partial=partial)
    return result


def write_report(output: Path, result: dict[str, Any], *, partial: bool) -> None:
    """报告只呈现已确认数值，科学Gate留给完整证据审阅。"""
    lines = ['# R2 '+('PARTIAL confirmed log summary' if partial else 'final data summary'), '',
             f"Experiment code: `{result['experiment_code_commit']}`.",
             f"Batch state: `{result['batch_execution_status']}`; analysis: `{result['status']}`.",
             'Only confirmed checkpoint log prefixes; no new environment steps or updates.', '',
             '|Group|Seed|Confirmed max transition|Logged episode steps|SAC updates|Evaluations|',
             '|---|---:|---:|---:|---:|---:|']
    for run in result['runs']:
        lines.append(f"|{run['group']}|{run['training_seed']}|{run['max_confirmed_transition']}|"
                     f"{run['logged_episode_transitions']}|"
                     f"{run['confirmed_complete_sac_updates']}|{run['observed_validation_points']}|")
    lines += ['', '## Fixed100k/300k Val300 endpoints', '',
              '|Group|Seed|Step|N complete|SR|Boundary|Timeout|Raw return|Common goal100 utility|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['validation_points']:
        if row['full']:
            lines.append(f"|{row['group']}|{row['training_seed']}|{row['at_transition']}|"
                         f"{row['complete_physical_episodes']}|{row['success_rate']:.6f}|"
                         f"{row['boundary_rate']:.6f}|{row['timeout_rate']:.6f}|"
                         f"{row['reward_mean']:.6f}|{row['common100utility_mean']:.6f}|")
    lines += ['', 'Cross-seed mean and n−1 sample SD: `endpoints_seed_mean_sd.csv`.',
              'All raw compact episodes: `all_episodes.csv.gz`; missing fields stay empty/null.',
              'Boundary/goal capture/reward strata: `termination_strata.csv`.',
              'Exact retained control nodes: `retained_trajectories.csv.gz`.', '',
              'Monitor30 and Val300 are separate repeated development validation evidence.',
              'Raw rewards100/200 differ; common100utility is only an offline comparison.',
              'Scientific Stage2 decision is not automatically assigned from completion or SR.',
              'No CV, Test/OOD, new training, or hypothesis confirmation is produced by this tool.']
    if partial:
        lines += ['', '**PARTIAL: unstarted runs, active episodes and unconfirmed tails are not**',
                  '**completed data. This report is neither PASS nor a scientific Gate.**']
    (output/('R2_PARTIAL_REPORT.md' if partial else 'R2_RESULT_REPORT.md')).write_text(
        '\n'.join(lines)+'\n', encoding='utf-8')


def plots(output: Path, *, partial: bool, trajectory_plots: bool = False) -> None:
    """独立绘图模式只读派生结果；SVG无平滑，NED正Down在显示轴向上明确反转。"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    result = read_json(output/('R2_PARTIAL_RESULT.json' if partial else 'R2_RESULT.json'))
    destination = output/'figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none'})
    metrics = [('success_rate', 'Success rate (%)', 100),
               ('boundary_rate', 'Boundary failures (%)', 100),
               ('timeout_rate', 'Task timeout (%)', 100),
               ('common100utility_mean', 'Common goal100 utility', 1),
               ('minimum_goal_distance_m_mean', 'Mean minimum goal distance (m)', 1),
               ('reward_mean', 'Actual reward (goal100 / goal200)', 1)]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    for axis, (metric, label, scale) in zip(axes.flat, metrics, strict=True):
        for group in GROUPS:
            for seed in SEEDS:
                rows = sorted((row for row in result['validation_points']
                               if row['group'] == group and row['training_seed'] == seed
                               and not row['full']), key=lambda row: row['at_transition'])
                axis.plot([row['at_transition']/1000 for row in rows],
                          [row[metric]*scale if row[metric] is not None else math.nan
                           for row in rows],
                          color=COLORS[group], marker='o', markersize=3, alpha=0.65,
                          linestyle={11: '-', 22: '--', 33: ':'}[seed], label=f'{group},seed{seed}')
        axis.set(xlabel='Training transitions (thousands)', ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(('PARTIAL — ' if partial else '')+'Fixed monitor30; additional Val300 separate')
    fig.savefig(destination/'monitor30_curves.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    endpoint_metrics = [('success_rate', 'Val300 success rate (%)', 100),
                        ('boundary_rate', 'Val300 boundary failures (%)', 100),
                        ('timeout_rate', 'Val300 task timeout (%)', 100),
                        ('common100utility_mean', 'Val300 common goal100 utility', 1)]
    selections = [(group, point) for group in GROUPS for point in FULL_POINTS]
    for axis, (metric, label, scale) in zip(axes.flat, endpoint_metrics, strict=True):
        for position, (group, point) in enumerate(selections):
            rows = [row for row in result['validation_points'] if row['group'] == group
                    and row['at_transition'] == point and row['full']]
            values = [row[metric]*scale for row in rows if row[metric] is not None]
            if not values:
                continue
            stats = moment(values)
            axis.scatter([position+(number-1)*0.06 for number in range(len(values))],
                         values, color=COLORS[group], s=20, alpha=0.7)
            axis.errorbar(position, stats['mean'], yerr=stats['sample_sd'],
                          color='black', marker='_', markersize=14, capsize=4)
            axis.annotate(f"n={len(values)} seeds", (position, stats['mean']),
                          xytext=(0, 8), textcoords='offset points', fontsize=6, ha='center')
        axis.set(xticks=list(range(len(selections))),
                 xticklabels=[f'{group}\n{point//1000}k' for group, point in selections],
                 ylabel=label)
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle(('PARTIAL — ' if partial else '')
                 +'Fixed Val300: individual seeds and mean ± sample SD; N=300 episodes/seed')
    fig.savefig(destination/'fixed_endpoint_val300.svg')
    plt.close(fig)
    plot_reward_components(result, destination, plt)
    plot_endpoint_distributions(output, destination, plt, partial=partial)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, ('alpha', 'actor_loss', 'q1_loss', 'q2_loss'), strict=True):
        for run in result['runs']:
            bins = run['update_bins']
            axis.plot([row['bin_last']/1000 for row in bins],
                      [row['metrics'][metric]['mean'] for row in bins], color=COLORS[run['group']],
                      linestyle={11: '-', 22: '--', 33: ':'}[run['training_seed']],
                      label=f"{run['group']},seed{run['training_seed']}")
        axis.set(xlabel='Training transitions (thousands)', ylabel=f'{metric}:25k update mean')
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(destination/'update_diagnostics.svg')
    plt.close(fig)
    if trajectory_plots:
        plot_trajectories(output, destination, plt)


def plot_reward_components(result: dict[str, Any], destination: Path, plt: Any) -> None:
    """四项真实reward与两个离线共同效用字段分开标注，不用return替代任务完成。"""
    panels = [
        ('reward_progress_mean', 'Actual progress component'),
        ('reward_goal_mean', 'Actual terminal component: C300×100 / G200×200'),
        ('reward_time_mean', 'Actual time component'),
        ('reward_smoothness_mean', 'Actual smoothness component'),
        ('common_goal', 'Offline common goal100 terminal component'),
        ('common100utility_mean', 'Offline common goal100 total utility'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels, strict=True):
        for group in GROUPS:
            for seed in SEEDS:
                rows = sorted((row for row in result['validation_points']
                               if row['group'] == group and row['training_seed'] == seed
                               and not row['full']), key=lambda row: row['at_transition'])
                if not rows:
                    continue
                values = [row['reward_goal_mean']-(GROUPS[group]-100)*row['success_rate']
                          if metric == 'common_goal' else row[metric] for row in rows]
                axis.plot([row['at_transition']/1000 for row in rows], values,
                          color=COLORS[group], linestyle={11: '-', 22: '--', 33: ':'}[seed],
                          marker='o', markersize=3, alpha=0.7, label=f'{group},seed{seed}')
        axis.set(xlabel='Training transitions (thousands)', ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(('PARTIAL — ' if result['status'].startswith('PARTIAL') else '')
                 +'Fixed monitor30: complete N=30/point; raw components vs offline utility')
    fig.savefig(destination/'reward_components_and_common100utility.svg')
    plt.close(fig)


def compact_endpoint_episodes(output: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    """只读一次紧凑episode文件提取真正300k Val300，不混入monitor30或训练记录。"""
    selected: dict[tuple[str, int], list[dict[str, str]]] = {}
    with gzip.open(output/'all_episodes.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if (row['scope'] == 'validation' and row['at_transition'] == '300000'
                    and row['evaluation_kind'] == 'Val300'):
                require(row['complete'] == 'True', '300k Val300含非完整episode。')
                selected.setdefault((row['group'], int(row['training_seed'])), []).append(row)
    for (group, seed), rows in selected.items():
        require(len(rows) == 300
                and {int(row['scenario_index']) for row in rows} == set(range(300)),
                f'{group}/seed{seed}紧凑终点记录分母/固定索引不完整。')
    return selected


def plot_endpoint_distributions(output: Path, destination: Path, plt: Any,
                                 *, partial: bool) -> None:
    """终点ECDF与终止子型均使用真实完整300分母，缺失距离不被默认为零。"""
    selected = compact_endpoint_episodes(output)
    if not selected:
        return
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    observed_rows = []
    for (group, seed), rows in sorted(selected.items()):
        values = sorted(float(row['minimum_goal_distance_m']) for row in rows
                        if row['minimum_goal_distance_m'] != '')
        require(all(math.isfinite(value) for value in values), '最小距离ECDF含非有限数据。')
        count = len(rows)
        if values:
            axis.step([values[0], *values], [0.0, *[(index+1)/count
                                                  for index in range(len(values))]],
                      where='post', color=COLORS[group],
                      linestyle={11: '-', 22: '--', 33: ':'}[seed],
                      label=f'{group},seed{seed}: measured{len(values)}/complete{count}')
        observed_rows.append(dict(group=group, training_seed=seed, at_transition=300000,
                                  evaluation_kind='Val300', complete_episodes=count,
                                  measured_minimum_goal_distance_episodes=len(values),
                                  missing_minimum_goal_distance_episodes=count-len(values)))
    axis.set(xlabel='Recorded minimum goal distance over executed segments (m)',
             ylabel='Empirical cumulative count / complete Val300 N', ylim=(0.0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    axis.set_title(('PARTIAL available endpoints — ' if partial else '')
                   +'Fixed300k Val300; missing measurements never assigned zero')
    fig.savefig(destination/'final_val300_minimum_goal_distance_ecdf.svg')
    plt.close(fig)
    write_rows(destination/'final_val300_distribution_denominators.csv', observed_rows)
    keys = sorted(selected)
    subtype_names = sorted({row['boundary_subtype'] or 'NOT_RECORDED'
                            for rows in selected.values() for row in rows
                            if row['failure_type'] == 'operational_boundary_failure'})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    counts = [Counter(row['failure_type'] for row in selected[key]) for key in keys]
    bottoms = [0.0]*len(keys)
    for event in EVENTS:
        heights = [100*counter[event]/len(selected[key])
                   for key, counter in zip(keys, counts, strict=True)]
        axes[0].bar(range(len(keys)), heights, bottom=bottoms, label=event)
        bottoms = [start+height for start, height in zip(bottoms, heights, strict=True)]
    subtype_counts = [Counter(row['boundary_subtype'] or 'NOT_RECORDED'
                              for row in selected[key]
                              if row['failure_type'] == 'operational_boundary_failure')
                      for key in keys]
    bottoms = [0.0]*len(keys)
    subtype_rows = []
    for subtype in subtype_names:
        heights = [100*counter[subtype]/len(selected[key])
                   for key, counter in zip(keys, subtype_counts, strict=True)]
        axes[1].bar(range(len(keys)), heights, bottom=bottoms, label=subtype)
        bottoms = [start+height for start, height in zip(bottoms, heights, strict=True)]
        for key, counter in zip(keys, subtype_counts, strict=True):
            subtype_rows.append(dict(group=key[0], training_seed=key[1], at_transition=300000,
                                     boundary_subtype=subtype, actual_count=counter[subtype],
                                     complete_episode_denominator=len(selected[key]),
                                     fraction_of_complete=counter[subtype]/len(selected[key])))
    for axis in axes:
        axis.set(xticks=list(range(len(keys))),
                 xticklabels=[f'{group}\nseed{seed}\nN={len(selected[group, seed])}'
                              for group, seed in keys], ylabel='% of complete Val300 episodes')
        axis.grid(axis='y', alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    axes[0].set_title('Physical termination counts; complete N=300 per available endpoint')
    axes[1].set_title('Recorded boundary subtypes; denominator stays all complete episodes')
    fig.suptitle(('PARTIAL — ' if partial else '')+'Fixed300k Val300 termination decomposition')
    fig.savefig(destination/'final_val300_termination_subtypes.svg')
    plt.close(fig)
    write_rows(destination/'final_val300_boundary_subtype_counts.csv', subtype_rows)


def plot_trajectories(output: Path, destination: Path, plt: Any) -> None:
    """仅300k额外Val300预先选中的所有轨迹，成功/失败都画，不挑最好一条。"""
    cases: dict[tuple[str, int, str], list[dict[str, str]]] = {}
    with gzip.open(output/'retained_trajectories.csv.gz', 'rt', newline='',
                   encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['at_transition'] == '300000' and row['evaluation_kind'] == 'Val300':
                cases.setdefault((row['group'], int(row['training_seed']),
                                  row['scenario_id']), []).append(row)
    for group in GROUPS:
        for seed in SEEDS:
            selected = [(key, rows) for key, rows in cases.items()
                        if key[:2] == (group, seed)]
            if not selected:
                continue
            fig = plt.figure(figsize=(12, 4), layout='constrained')
            three = fig.add_subplot(131, projection='3d')
            ne, nd = fig.add_subplot(132), fig.add_subplot(133)
            for _, rows in selected:
                north = [float(row['north_m']) for row in rows]
                east = [float(row['east_m']) for row in rows]
                down = [float(row['down_m']) for row in rows]
                goal = [float(rows[0][name])
                        for name in ('goal_north_m', 'goal_east_m', 'goal_down_m')]
                label = f"index{rows[0]['scenario_index']}:{rows[0]['failure_type']}"
                line, = three.plot(north, east, down, label=label)
                color = line.get_color()
                three.scatter(*goal, marker='x', color=color)
                ne.plot(east, north, color=color, label=label)
                ne.scatter(goal[1], goal[0], marker='x', color=color)
                nd.plot(north, down, color=color, label=label)
                nd.scatter(goal[0], goal[2], marker='x', color=color)
            three.set(xlabel='North (m)', ylabel='East (m)', zlabel='Down (m, positive down)')
            three.invert_zaxis()
            ne.set(xlabel='East (m)', ylabel='North (m)', aspect='equal')
            nd.set(xlabel='North (m)', ylabel='Down (m, positive down)', aspect='equal')
            nd.invert_yaxis()
            ne.grid(alpha=0.25)
            nd.grid(alpha=0.25)
            fig.suptitle(f'{group},seed{seed},300k Val300 — retained by preregistered IDs/events')
            ne.legend(fontsize=5, loc='best')
            fig.savefig(destination/f'{group}_seed{seed}_300k_retained_trajectories.svg')
            plt.close(fig)


def main() -> int:
    """默认最终分析要求真实完成；训练中须显式--partial；绘图不读取模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=TASK)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--trajectory-plots', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == TASK.resolve(), 'R2分析只读取实际任务目录，不混入其他项目。')
    output = root/('partial_analysis' if args.partial else 'analysis')
    if not args.plot_only:
        result = analyze(root, partial=args.partial)
        print(json.dumps(dict(status=result['status'], counts=result['counts']),
                         ensure_ascii=False))
    if args.plot_only or args.plots or args.trajectory_plots:
        plots(output, partial=args.partial, trajectory_plots=args.trajectory_plots)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
