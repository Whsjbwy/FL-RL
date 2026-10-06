"""只读导出已完成B0批次的逐episode原值与登记轨迹，不导出模型/Replay/update。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[2] / 'src/auv_risk_rl/training/mvp_analysis.py'
SPEC = importlib.util.spec_from_file_location('existing_mvp_analysis', SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError('现有MVP分析模块不能读取。')
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)

REGISTRATION = 'STAGE2_B0_MVP_BATCH_V1'
METHOD = 'B0_FULL_STATE_ORDINARY_SAC'
SEEDS = (11, 22, 33)
EVENTS = ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')
POINTS = {(profile, step, full) for profile, points in ANALYSIS.EXPECTED_VALIDATIONS.items()
          for step, full in points}
EPISODE_FIELDS = (
    'training_seed', 'task_profile', 'at_transition', 'scenario_id', 'base_scenario_id',
    'actual_scenario_id', 'scenario_root_seed', 'scenario_index', 'index', 'environment_seed',
    'episode_id', 'env_slot', 'split', 'start_transition', 'last_transition', 'steps',
    'physical_time_s', 'reward', 'path_length_m', 'path_length_definition',
    'minimum_clearance_m', 'action_saturation_count', 'failure_type', 'complete',
    'terminated', 'truncated', 'success', 'collision', 'boundary', 'task_timeout',
    'external_truncation', 'phase_boundary', 'budget_stop', 'boundary_reason',
    'initial_distance_m', 'final_distance_m', 'progress_m',
)
REWARD_FIELDS = ('progress', 'goal', 'time', 'smoothness')
SOURCE_FIELDS = ('segment_id', 'log_sequence', 'valid_log_sequence', 'source_path',
                 'source_line', 'registration_id', 'method', 'run_kind', 'code_version')
EXTRA_FIELDS = ('evaluation_kind', 'evaluation_key', 'validation_root_seed',
                'evaluation_episode_count', 'event_denominator_eligible',
                'warmup_duration_s', 'warmup_environment_seed', 'warmup_json',
                'trajectory_retention_reasons_json', 'initial_position_ned_m_json',
                'goal_position_ned_m_json')
TRAJECTORY_FIELDS = (
    'training_seed', 'task_profile', 'at_transition', 'index', 'node_index', 'task_step',
    'time_s', 'elapsed_s', 'north_m', 'east_m', 'down_m', 'action_speed_normalized',
    'action_yaw_rate_normalized', 'action_pitch_rate_normalized', 'reward',
    'reward_progress', 'reward_goal', 'reward_time', 'reward_smoothness',
    'minimum_clearance_m', 'failure_type',
)


def read_json(path: Path) -> Any:
    """拒绝JSON非有限常量，避免把损坏日志写成有效公开证据。"""
    def invalid(value: str) -> None:
        raise ValueError(f'JSON含非有限数值：{value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def cell(value: Any) -> Any:
    """CSV以字面null保留未测量/不适用，显式bool保持true/false。"""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('导出数值必须有限。')
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return value


def validate_complete(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """只接受真实COMPLETED且既有全量分析已经核验计数/完整固定验证的批次。"""
    state = read_json(root / 'batch_state.json')
    analysis = read_json(root / 'analysis/batch_analysis.json')
    if state.get('status') != 'COMPLETED' or state.get('registration_id') != REGISTRATION:
        raise ValueError('批次尚未COMPLETED或登记身份不符；不导出最终证据。')
    code = state['experiment_code_commit']
    if (analysis.get('status') != 'BATCH_DATA_COMPLETE'
            or analysis.get('registration_id') != REGISTRATION
            or analysis.get('experiment_code_commit') != code):
        raise ValueError('全量分析未完成或代码/登记身份不一致。')
    jobs = {(job['seed'], job['stop']) for job in state['completed_jobs']}
    if jobs != {(seed, stop) for seed in SEEDS for stop in (100000, 300000)}:
        raise ValueError('六个登记任务没有完整完成。')
    for seed in SEEDS:
        row = state['seeds'][str(seed)]
        summary = next(item for item in analysis['seed_summaries']
                       if item['training_seed'] == seed)
        if (row['transitions'] != 300000 or row['updates'] != 290001
                or summary['status'] != 'COMPLETE'
                or summary['authoritative_logged_steps'] != row['transitions']
                or summary['authoritative_logged_updates'] != row['updates']):
            raise ValueError('本批真实计数与已核验原始日志不一致。')
    return state, analysis


def records(root: Path, seed: int, code: str,
            expected_segments: list[dict[str, Any]], excluded: Counter[str],
            ) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """复用现有confirmed reader，丢弃update输出；不另建日志身份/恢复解释规则。"""
    segments = read_json(root / f'seed_{seed}/segments.json')
    if segments != expected_segments:
        raise ValueError('segment登记已偏离完成后的全量分析，需重新核验。')
    diagnostics: dict[str, Any] = {}
    for kind, row in ANALYSIS.confirmed_records(root, seed, code, diagnostics):
        if kind == 'update':
            continue
        origin = row['_log_source']
        source = {key: row[key] for key in SOURCE_FIELDS if key in row}
        source.update(training_seed=row['training_seed'],
                      valid_log_sequence=origin['valid_log_sequence'],
                      source_path=origin['path'], source_line=origin['line_number'])
        yield kind, row, source
    excluded.update(diagnostics['excluded_unconfirmed_rows'])


def episode_row(episode: dict[str, Any], source: dict[str, Any],
                validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """保留全部必要原始标量/ID/四项reward；不会把未完成片段加入物理事件分母。"""
    result = {key: episode.get(key) for key in EPISODE_FIELDS}
    result.update(source)
    result.update({f'reward_{key}': episode['reward_components'][key] for key in REWARD_FIELDS})
    warmup = episode.get('warmup', {})
    result.update(warmup_duration_s=warmup.get('warmup_duration_s'),
                  warmup_environment_seed=warmup.get('root_seed'), warmup_json=warmup,
                  trajectory_retention_reasons_json=episode.get('trajectory_retention_reasons'),
                  initial_position_ned_m_json=episode.get('initial_position_ned_m'),
                  goal_position_ned_m_json=episode.get('goal_position_ned_m'),
                  event_denominator_eligible=(episode['complete'] is True
                                              and episode['failure_type'] in EVENTS))
    if validation is not None:
        result.update(training_seed=source['training_seed'], task_profile=validation['profile'],
                      at_transition=validation['at_transition'],
                      evaluation_kind='Val300' if validation['full'] else 'monitor30',
                      evaluation_key=validation['evaluation_key'],
                      validation_root_seed=validation['validation_root_seed'],
                      evaluation_episode_count=validation['count'])
    return result


def trajectory_rows(episode: dict[str, Any], source: dict[str, Any],
                    point: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """导出实际NED节点与指令，初态动作/reward为null；不插值、不补未来轨迹。"""
    elapsed = 0.0
    nodes = [dict(task_step=0, elapsed_s=0.0,
                  position_ned_m=episode['initial_position_ned_m'])] + episode['trajectory']
    for number, node in enumerate(nodes):
        elapsed += node['elapsed_s']
        position = node['position_ned_m']
        action = node.get('action', [None, None, None])
        if len(position) != 3 or len(action) != 3:
            raise ValueError('轨迹位置/指令必须三维。')
        row = dict(training_seed=source['training_seed'], task_profile=point['profile'],
                   at_transition=point['at_transition'], index=episode['index'], node_index=number,
                   task_step=node['task_step'], time_s=elapsed, elapsed_s=node['elapsed_s'],
                   north_m=position[0], east_m=position[1], down_m=position[2],
                   action_speed_normalized=action[0], action_yaw_rate_normalized=action[1],
                   action_pitch_rate_normalized=action[2], reward=node.get('reward'),
                   minimum_clearance_m=node.get('minimum_clearance_m'),
                   failure_type=node.get('failure_type'))
        row.update({f'reward_{key}': node.get('reward_components', {}).get(key)
                    for key in REWARD_FIELDS})
        yield row
    if not math.isclose(elapsed, episode['physical_time_s'], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError('实际轨迹累计时间与原始episode时间不一致。')


def write_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    """全部列显式写出null，避免空格/空值被下游自动当作零。"""
    writer.writerow({key: cell(row.get(key)) for key in writer.fieldnames})


def export(root: Path) -> dict[str, Any]:
    """生成一份公开CSV证据集；不改原始日志、批次状态或科学结果。"""
    root = root.resolve()
    state, analysis = validate_complete(root)
    output = root / 'public_evidence'
    output.mkdir(exist_ok=True)
    fields = list(EPISODE_FIELDS + SOURCE_FIELDS + EXTRA_FIELDS)
    fields.extend('reward_' + key for key in REWARD_FIELDS)
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    paths = [output / name for name in ('training_episodes.csv', 'validation_episodes.csv',
                                       'selected_validation_trajectories.csv')]
    partials = [path.with_suffix('.csv.partial') for path in paths]
    with (partials[0].open('w', newline='', encoding='utf-8') as train_stream,
          partials[1].open('w', newline='', encoding='utf-8') as val_stream,
          partials[2].open('w', newline='', encoding='utf-8') as trajectory_stream):
        training, validation = (csv.DictWriter(stream, fieldnames=fields)
                                for stream in (train_stream, val_stream))
        trajectories = csv.DictWriter(trajectory_stream, fieldnames=TRAJECTORY_FIELDS)
        for writer in (training, validation, trajectories):
            writer.writeheader()
        for seed in SEEDS:
            summary = next(row for row in analysis['seed_summaries']
                           if row['training_seed'] == seed)
            observed, steps = set(), Counter()
            for kind, row, source in records(root, seed, state['experiment_code_commit'],
                                            summary['segment_diagnostics']['segments'], excluded):
                if kind == 'episode':
                    write_row(training, episode_row(row, source))
                    counts['training_episodes'] += 1
                    steps[row['task_profile']] += row['steps']
                    continue
                key = (row['profile'], row['at_transition'], row['full'])
                ANALYSIS.validation_point(row)
                if key not in POINTS or key in observed:
                    raise ValueError('固定验证点重复或未登记。')
                observed.add(key)
                episodes = row['episodes']
                first_failure = next((item['index'] for item in episodes
                                      if item['failure_type'] != 'goal_success'), None)
                expected_trajectories = {0, 1, 2}
                if first_failure is not None:
                    expected_trajectories.add(first_failure)
                retained = {item['index'] for item in episodes if 'trajectory' in item}
                if retained != expected_trajectories:
                    raise ValueError('保留轨迹不是登记索引0/1/2+最早失败。')
                for episode in episodes:
                    if episode['complete'] is not True or episode['failure_type'] not in EVENTS:
                        raise ValueError('科研验证必须完整物理终止，不接受工程截断。')
                    write_row(validation, episode_row(episode, source, row))
                    counts['validation_episodes'] += 1
                    if 'trajectory' in episode:
                        counts['selected_trajectories'] += 1
                        for node in trajectory_rows(episode, source, row):
                            write_row(trajectories, node)
                            counts['trajectory_nodes'] += 1
                counts['validation_points'] += 1
            if observed != POINTS or steps != {'obstacle_free': 100000, 'cv_train_v1': 200000}:
                raise ValueError('逐episode训练步数或完整固定验证安排与完成记录不一致。')
    # 原始证据始终保留；只有全部身份/计数验证成功后才替换导出文件。
    for partial, destination in zip(partials, paths, strict=True):
        partial.replace(destination)
    report = dict(created_at=datetime.now(UTC).isoformat(), registration_id=REGISTRATION,
                  experiment_code_commit=state['experiment_code_commit'], counts=dict(counts),
                  excluded_unconfirmed_rows=dict(excluded), outputs=[path.name for path in paths],
                  output_bytes=sum(path.stat().st_size for path in paths), null_token='null',
                  bool_tokens=['true', 'false'], raw_sources='batch-local confirmed JSONL',
                  production_configuration=state['registration'],
                  event_denominator='complete physical episodes; phase/budget/external fragments '
                  'excluded, each row has explicit event_denominator_eligible',
                  path_length_definition='CONTROL_NODE_POLYLINE',
                  trajectory_selection='indices0/1/2 plus earliest failed base index at every '
                  'registered fixed validation point; actual NED coordinates and elapsed time',
                  no_obstacle_clearance='null; not measured/applicable, never replaced by zero',
                  repeated_validation='monitor30 and Val300 retain their exact checkpoint/sample '
                  'identity; episodes are not independent training seeds')
    (output / 'EXPORT_RECORD.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main() -> int:
    """默认本批结果路径，未完成批次fail-fast，不运行环境或梯度操作。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(export(args.batch_root), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
