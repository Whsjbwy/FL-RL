"""从公开CSV复原已登记验证轨迹，复用冻结绘图函数；不读本机raw日志或模型。"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

METHOD = 'B0_FULL_STATE_ORDINARY_SAC'
REGISTRATION = 'STAGE2_B0_MVP_BATCH_V1'
SEEDS = (11, 22, 33)
ROOT = Path(__file__).resolve().parent
INTEGER_FIELDS = {
    'training_seed', 'at_transition', 'scenario_root_seed', 'scenario_index', 'index',
    'environment_seed', 'episode_id', 'env_slot', 'start_transition', 'last_transition',
    'steps', 'action_saturation_count', 'log_sequence', 'valid_log_sequence', 'source_line',
    'validation_root_seed', 'evaluation_episode_count', 'warmup_environment_seed',
    'node_index', 'task_step',
}
FLOAT_FIELDS = {
    'physical_time_s', 'reward', 'path_length_m', 'minimum_clearance_m', 'initial_distance_m',
    'final_distance_m', 'progress_m', 'warmup_duration_s', 'time_s', 'elapsed_s', 'north_m',
    'east_m', 'down_m', 'action_speed_normalized', 'action_yaw_rate_normalized',
    'action_pitch_rate_normalized', 'reward_progress', 'reward_goal', 'reward_time',
    'reward_smoothness',
}


def read_json(path: Path) -> Any:
    """读取公开结构化证据；JSON非有限值不是可接受的图形输入。"""
    def invalid(value: str) -> None:
        raise ValueError(f'JSON含非有限值：{value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def read_csv(path: Path) -> list[dict[str, Any]]:
    """遵循导出器的显式null/bool/JSON和原始浮点编码，不把不适用值改成零。"""
    result = []
    opener = (gzip.open(path, 'rt', newline='', encoding='utf-8')
              if path.suffix == '.gz' else path.open(newline='', encoding='utf-8'))
    with opener as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for name, value in raw.items():
                if value is None or name is None:
                    raise ValueError('CSV列数与表头不一致。')
                if value == 'null':
                    parsed = None
                elif value in ('true', 'false'):
                    parsed = value == 'true'
                elif name.endswith('_json'):
                    parsed = json.loads(value)
                elif name in INTEGER_FIELDS:
                    parsed = int(value)
                elif name in FLOAT_FIELDS:
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError('公开CSV含NaN/Inf。')
                else:
                    parsed = value
                row[name.removesuffix('_json')] = parsed
            result.append(row)
    return result


def point_key(row: dict[str, Any]) -> tuple[int, str, int]:
    """同一seed/profile/训练checkpoint识别验证点，不能把两个100k任务版本混合。"""
    profile = row.get('profile', row.get('task_profile'))
    return row['training_seed'], profile, row['at_transition']


def retained_rows(analysis: dict[str, Any], evidence: Path, *,
                  prefer_compressed: bool = False) -> list[dict[str, Any]]:
    """核对完整数据身份，复原原绘图API需要的登记episode和真实控制节点。"""
    exported = read_json(evidence / 'EXPORT_RECORD.json')
    code = analysis['experiment_code_commit']
    if (analysis.get('status') != 'BATCH_DATA_COMPLETE'
            or analysis.get('registration_id') != REGISTRATION
            or exported.get('registration_id') != REGISTRATION
            or exported.get('experiment_code_commit') != code
            or exported.get('null_token') != 'null'):
        raise ValueError('只接受已完整核验且公开CSV与分析身份一致的实际批次。')
    summaries = analysis['seed_summaries']
    if ({item['training_seed'] for item in summaries} != set(SEEDS)
            or any(item['status'] != 'COMPLETE' for item in summaries)):
        raise ValueError('三个登记seed尚未全部完成。')
    points = {point_key(point): point for point in analysis['validation_points']}
    if len(points) != len(analysis['validation_points']):
        raise ValueError('分析包含重复验证点。')
    all_episodes = read_csv(evidence / 'validation_episodes.csv')
    raw = evidence / 'selected_validation_trajectories.csv'
    compressed = evidence / 'selected_validation_trajectories.csv.gz'
    # 显式选压缩版时不得静默退回本机raw；公开副本只有gzip时也可直接复原。
    trajectory_source = compressed if prefer_compressed or not raw.is_file() else raw
    nodes = read_csv(trajectory_source)
    if (len(all_episodes) != exported['counts']['validation_episodes']
            or len(nodes) != exported['counts']['trajectory_nodes']
            or len(points) != exported['counts']['validation_points']):
        raise ValueError('公开原始表行数与导出记录不一致。')
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in all_episodes:
        key = point_key(episode)
        point = points.get(key)
        expected = dict(registration_id=REGISTRATION, method=METHOD,
                        run_kind='scientific_training', code_version=code)
        if point is None or any(episode.get(name) != value for name, value in expected.items()):
            raise ValueError('逐场景公开记录身份/检查点不属于本批分析。')
        origin = point['trajectory_source']
        source_pairs = {'source_path': 'path', 'source_line': 'line_number',
                        'log_sequence': 'log_sequence', 'valid_log_sequence': 'valid_log_sequence'}
        if any(episode[name] != origin[original] for name, original in source_pairs.items()):
            raise ValueError('公开episode与confirmed原始行引用不一致。')
        if (episode['validation_root_seed'] != point['validation_root_seed']
                or episode['evaluation_key'] != point['evaluation_key']
                or episode['evaluation_kind'] != point['evaluation_kind']
                or episode['complete'] is not True
                or episode['event_denominator_eligible'] is not True):
            raise ValueError('固定验证身份或完整物理事件分母不一致。')
        grouped[key].append(episode)
    node_groups: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        node_groups[(*point_key(node), node['index'])].append(node)
    result, used = [], set()
    for key, point in points.items():
        episodes = grouped[key]
        expected_indices = point['base_indices']
        if (sorted(item['index'] for item in episodes) != expected_indices
                or any(item['evaluation_episode_count'] != len(expected_indices)
                       for item in episodes)):
            raise ValueError('公开场景缺失/重复或固定30/300样本数变化。')
        events = Counter(item['failure_type'] for item in episodes)
        if any(events[event] != count for event, count in
               point['all_registered_episodes']['event_counts'].items()):
            raise ValueError('公开逐场景事件分类与分析分母不一致。')
        selected = []
        for episode in episodes:
            node_key = (*key, episode['index'])
            if episode['index'] not in point['retained_trajectory_indices']:
                if node_key in node_groups:
                    raise ValueError('轨迹表包含未登记的场景选择。')
                continue
            sequence = node_groups[node_key]
            if [node['node_index'] for node in sequence] != list(range(episode['steps'] + 1)):
                raise ValueError('实际轨迹节点缺失/重复/乱序。')
            initial = sequence[0]
            if ([initial[name] for name in ('north_m', 'east_m', 'down_m')]
                    != episode['initial_position_ned_m']
                    or initial['task_step'] != 0 or initial['time_s'] != 0.0
                    or initial['elapsed_s'] != 0.0):
                raise ValueError('CSV初态不是该episode真实初态。')
            rebuilt, elapsed = [], 0.0
            for node in sequence[1:]:
                elapsed += node['elapsed_s']
                if node['elapsed_s'] <= 0.0 or node['time_s'] != elapsed:
                    raise ValueError('轨迹时间不是实际执行前缀累计时间。')
                rebuilt.append(dict(
                    task_step=node['task_step'], elapsed_s=node['elapsed_s'],
                    position_ned_m=[node[name] for name in ('north_m', 'east_m', 'down_m')],
                    action=[node[name] for name in ('action_speed_normalized',
                            'action_yaw_rate_normalized', 'action_pitch_rate_normalized')],
                    reward=node['reward'], minimum_clearance_m=node['minimum_clearance_m'],
                    failure_type=node['failure_type'],
                    reward_components={name: node['reward_' + name]
                                       for name in ('progress', 'goal', 'time', 'smoothness')}))
            if not math.isclose(elapsed, episode['physical_time_s'], rel_tol=0.0, abs_tol=1e-8):
                raise ValueError('轨迹累计时间与导出器核验的episode时长不一致。')
            selected.append(dict(episode, trajectory=rebuilt))
            used.add(node_key)
        if sorted(item['index'] for item in selected) != point['retained_trajectory_indices']:
            raise ValueError('登记的成功/失败轨迹不完整。')
        result.append(dict(point=point, episodes=selected))
    if used != set(node_groups) or len(used) != exported['counts']['selected_trajectories']:
        raise ValueError('存在未匹配的公开轨迹。')
    return result


def render_public(analysis: dict[str, Any], evidence: Path, output: Path, *,
                  prefer_compressed: bool = False) -> list[str]:
    """只在本实例替换本地raw读取器，其余绘图函数直接复用冻结脚本。"""
    retained = retained_rows(analysis, evidence, prefer_compressed=prefer_compressed)
    spec = importlib.util.spec_from_file_location(
        'existing_b0_plotter', ROOT.parents[1] / 'scripts/analyze_b0_mvp.py')
    if spec is None or spec.loader is None:
        raise ImportError('冻结的现有绘图脚本不能读取。')
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    plotter._retained_validation_rows = lambda _: retained
    return plotter.render_plots(analysis, output)


def main() -> int:
    """显式输出路径限定在本任务results；无需本机raw JSONL、Torch或checkpoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-json', type=Path, required=True)
    parser.add_argument('--public-evidence', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prefer-compressed-trajectories', action='store_true',
                        help='显式读取公开.csv.gz；不存在时fail-fast，不退回本机raw CSV。')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to(ROOT.resolve())
    analysis = read_json(args.analysis_json)
    files = render_public(analysis, args.public_evidence, output,
                          prefer_compressed=args.prefer_compressed_trajectories)
    print(json.dumps(dict(status='PUBLIC_FIGURES_REPRODUCED',
                          experiment_code_commit=analysis['experiment_code_commit'],
                          figure_files=files), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
