"""登记B0 MVP原始日志汇总与静态科研图；默认只读分析，不执行学习策略诊断。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33)
COLORS = ('#0072B2', '#D55E00', '#009E73')
METRICS = (
    ('success_rate', 'Success fraction'), ('collision_rate', 'Collision fraction'),
    ('boundary_rate', 'Boundary failure fraction'), ('timeout_rate', 'Task timeout fraction'),
    ('reward', 'Episode task reward'),
)


def parser() -> argparse.ArgumentParser:
    """独立plot-only模式不导入Torch，可使用现有文档绘图运行时。"""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--batch-root', type=Path,
                        default=ROOT / 'results' / 'stage2_b0_mvp_v1')
    result.add_argument('--output-dir', type=Path)
    result.add_argument('--plot-only', action='store_true')
    result.add_argument('--analysis-json', type=Path)
    result.add_argument('--learned-diagnostics', action='store_true')
    result.add_argument('--trusted-local-models', action='store_true')
    result.add_argument('--project-config', type=Path, default=ROOT / 'configs' / 'stage0.yaml')
    return result


def _value(stats: dict[str, Any], metric: str) -> float:
    """未测量值以曲线间断表示，不能当作零成功率或零reward。"""
    value = stats['moments']['reward']['mean'] if metric == 'reward' else stats[metric]
    return math.nan if value is None else float(value)


def trajectory_series(episode: dict[str, Any]) -> dict[str, list[float]]:
    """使用原始NED控制节点和实际执行前缀时间；不插值、不平滑、不重定坐标。"""
    positions = [episode['initial_position_ned_m']]
    times = [0.0]
    for node in episode['trajectory']:
        elapsed = float(node['elapsed_s'])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('轨迹必须记录实际正执行时间。')
        times.append(times[-1]+elapsed)
        positions.append(node['position_ned_m'])
    if any(len(position) != 3 or not all(math.isfinite(float(v)) for v in position)
           for position in positions):
        raise ValueError('实际轨迹NED坐标必须是有限三维米制量。')
    return dict(time_s=times, north_m=[float(p[0]) for p in positions],
                east_m=[float(p[1]) for p in positions], down_m=[float(p[2]) for p in positions])


def _retained_validation_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """根据已核验segment行引用流式读取轨迹，避免把全轨迹再复制进汇总JSON。"""
    root = Path(analysis['batch_root']).resolve()
    sources: dict[str, dict[int, dict[str, Any]]] = {}
    for point in analysis['validation_points']:
        reference = point.get('trajectory_source')
        if reference and point.get('retained_trajectory_indices'):
            sources.setdefault(reference['path'], {})[reference['log_sequence']] = point
    result = []
    for relative, expected in sources.items():
        path = (root / relative).resolve()
        path.relative_to(root)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                point = expected.get(row['log_sequence'])
                if point is None:
                    continue
                if (row.get('registration_id') != analysis['registration_id']
                        or row.get('code_version') != analysis['experiment_code_commit']
                        or row.get('training_seed') != point['training_seed']
                        or row.get('profile') != point['profile']
                        or row.get('at_transition') != point['at_transition']):
                    raise ValueError('曲线分析后的轨迹原始日志身份已变化。')
                episodes = [episode for episode in row['episodes'] if 'trajectory' in episode]
                if sorted(episode['index'] for episode in episodes) != sorted(
                        point['retained_trajectory_indices']):
                    raise ValueError('实际保留轨迹集合与核验记录不匹配。')
                result.append(dict(point=point, episodes=episodes))
                del expected[row['log_sequence']]
        if expected:
            raise ValueError('缺少已登记的实际轨迹日志行。')
    return result


def render_trajectories(analysis: dict[str, Any], output: Path) -> list[str]:
    """同一原始失败/成功轨迹的3D、N/E俯视、实际时间深度图，Down轴向下。"""
    import matplotlib.pyplot as plt

    files = []
    for record in _retained_validation_rows(analysis):
        point = record['point']
        fig = plt.figure(figsize=(15, 5.1), layout='constrained')
        spatial = fig.add_subplot(131, projection='3d')
        plan, depth = fig.add_subplot(132), fig.add_subplot(133)
        for episode in record['episodes']:
            series = trajectory_series(episode)
            label = f"Index {episode['index']}: {episode['failure_type']}"
            line, = spatial.plot(series['north_m'], series['east_m'], series['down_m'],
                                 label=label, linewidth=1.4)
            color = line.get_color()
            plan.plot(series['north_m'], series['east_m'], color=color, linewidth=1.4)
            depth.plot(series['time_s'], series['down_m'], color=color, linewidth=1.4)
            goal = episode['goal_position_ned_m']
            spatial.scatter(goal[0], goal[1], goal[2], color=color, marker='*', s=45)
            plan.scatter(goal[0], goal[1], color=color, marker='*', s=45)
            spatial.scatter(series['north_m'][-1], series['east_m'][-1], series['down_m'][-1],
                            color=color, marker='x', s=22)
            plan.scatter(series['north_m'][-1], series['east_m'][-1], color=color,
                         marker='x', s=22)
        spatial.set(xlabel='North (m)', ylabel='East (m)', zlabel='Down (m)')
        spatial.invert_zaxis()
        spatial.legend(fontsize=7, loc='upper left')
        plan.set(xlabel='North (m)', ylabel='East (m)', title='N/E plan; star=goal, x=last state')
        plan.set_aspect('equal', adjustable='datalim')
        depth.set(xlabel='Actual elapsed task time (s)', ylabel='Down (m)',
                  title='NED depth; no extrapolation past actual event')
        depth.invert_yaxis()
        for axis in (plan, depth):
            axis.grid(alpha=.25)
        fig.suptitle(f"Seed {point['training_seed']} | {point['profile']} | "
                     f"{point['at_transition']} transitions | {analysis['status']}\n"
                     'Registered indices 0/1/2 + earliest failure; raw control nodes, no smoothing')
        filename = (f"trajectory_seed_{point['training_seed']}_{point['profile']}_"
                    f"{point['at_transition']}.svg")
        fig.savefig(output / filename)
        plt.close(fig)
        files.append(filename)
    return files


def render_plots(analysis: dict[str, Any], output: Path) -> list[str]:
    """matplotlib独立SVG；保留三seed原始线，样本SD只基于独立训练seed。"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                                'svg.fonttype': 'none'})
    files = []
    label = analysis['status']
    for profile in ('obstacle_free', 'cv_train_v1'):
        fig, axes = plt.subplots(5, 1, figsize=(8.4, 13), sharex=True, layout='constrained')
        points = [point for point in analysis['validation_points'] if point['profile'] == profile]
        for axis, (metric, ylabel) in zip(axes, METRICS, strict=True):
            for seed, color in zip(SEEDS, COLORS, strict=True):
                selected = sorted((point for point in points if point['training_seed'] == seed),
                                  key=lambda point: point['at_transition'])
                axis.plot([point['at_transition'] / 1000 for point in selected],
                          [_value(point['paired_monitor30'], metric) for point in selected],
                          marker='o', linewidth=1.3, color=color, label=f'Seed {seed}')
            pooled = sorted((point for point in analysis['validation_seed_mean_sd']
                             if point['profile'] == profile),
                            key=lambda point: point['at_transition'])
            # 仅N=3时绘制跨seed均值/SD；部分批次仍显示已有raw seed点，不冒充完整三seed。
            full = [point for point in pooled if point['metrics'][metric]['n'] == 3]
            x = [point['at_transition'] / 1000 for point in full]
            means = [point['metrics'][metric]['mean'] for point in full]
            deviations = [point['metrics'][metric]['sample_sd'] for point in full]
            axis.plot(x, means, color='#222222', linestyle='--', label='Mean (3 seeds)')
            axis.fill_between(x, [m-s for m, s in zip(means, deviations, strict=True)],
                              [m+s for m, s in zip(means, deviations, strict=True)],
                              alpha=0.13, color='#333333', label='Sample SD (n-1)')
            axis.axvline(100, color='#666666', linestyle=':', linewidth=1)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if metric != 'reward':
                axis.set_ylim(-0.04, 1.04)
            axis.set_xlim((0, 105) if profile == 'obstacle_free' else (95, 305))
        axes[0].legend(loc='best', ncol=3, fontsize=8)
        axes[-1].set_xlabel('Global training environment transitions (thousands)')
        fig.suptitle(f'B0 {profile} | fixed indices 0-29 | {label}\n'
                     'Monitor30; endpoints use the same 30-index subset of Val300')
        filename = f'validation_{profile}.svg'
        fig.savefig(output / filename)
        plt.close(fig)
        files.append(filename)

        fig, axes = plt.subplots(5, 1, figsize=(8.4, 13), sharex=True, layout='constrained')
        bins = [row for row in analysis['training_bins'] if row['profile'] == profile]
        for axis, (metric, ylabel) in zip(axes, METRICS, strict=True):
            for seed, color in zip(SEEDS, COLORS, strict=True):
                selected = sorted((row for row in bins if row['training_seed'] == seed),
                                  key=lambda row: row['bin_end_transition'])
                axis.plot([row['bin_end_transition'] / 1000 for row in selected],
                          [_value(row, metric) for row in selected], marker='o',
                          color=color, label=f'Seed {seed}')
            axis.axvline(100, color='#666666', linestyle=':', linewidth=1)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if metric != 'reward':
                axis.set_ylim(-0.04, 1.04)
            axis.set_xlim((0, 105) if profile == 'obstacle_free' else (95, 305))
        axes[0].legend(loc='best')
        axes[-1].set_xlabel('Episode end transition, 25k bins (thousands)')
        fig.suptitle(f'B0 {profile} | complete physical training episodes | {label}\n'
                     'Raw seed bins; phase/budget fragments excluded from event fractions')
        filename = f'training_{profile}.svg'
        fig.savefig(output / filename)
        plt.close(fig)
        files.append(filename)

    fig, axes = plt.subplots(5, 1, figsize=(8.4, 13), layout='constrained')
    for axis, (metric, ylabel) in zip(axes, METRICS, strict=True):
        for profile, offset, marker in (('obstacle_free', -0.12, 'o'),
                                         ('cv_train_v1', 0.12, 's')):
            for index, (seed, color) in enumerate(zip(SEEDS, COLORS, strict=True)):
                matches = [point for point in analysis['validation_points']
                           if point['profile'] == profile and point['training_seed'] == seed
                           and point['full']]
                if matches:
                    value = _value(matches[0]['all_registered_episodes'], metric)
                    axis.scatter(index+offset, value,
                                 color=color, marker=marker, label=f'{profile}, seed {seed}')
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 1, 2], [str(seed) for seed in SEEDS])
        axis.grid(alpha=0.25)
        if metric != 'reward':
            axis.set_ylim(-0.04, 1.04)
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[-1].set_xlabel('Independent training seed')
    fig.suptitle(f'B0 full Val300 raw results | {label}\n'
                 'Circle: empty 100k; square: CV final 300k. Each profile is distinct.')
    fig.savefig(output / 'val300_raw_seeds.svg')
    plt.close(fig)
    files.append('val300_raw_seeds.svg')
    if 'batch_root' in analysis:
        files.extend(render_trajectories(analysis, output))
    return files


def write_report(analysis: dict[str, Any], output: Path) -> None:
    """单份证据说明；训练日程完成与Stage2科学判断明确分开。"""
    lines = [f"# B0 MVP batch analysis — {analysis['status']}", '',
             f"Registration: `{analysis['registration_id']}`.",
             f"Actual training code: `{analysis['experiment_code_commit']}`.", '',
             'This report uses checkpoint-confirmed segment records only. Unconfirmed tails ',
             'remain raw historical evidence and do not enter scientific denominators.', '',
             '| Seed | Data status | Confirmed transitions | Confirmed full updates |',
             '| --- | --- | ---: | ---: |']
    for seed in analysis['seed_summaries']:
        lines.append(f"| {seed['training_seed']} | {seed['status']} | "
                     f"{seed['authoritative_logged_steps']} | "
                     f"{seed['authoritative_logged_updates']} |")
    lines.extend(['', '## Full registered validation results', '',
                  '| Seed | Profile | Transition | Episodes | SR | Collision | Boundary | '
                  'Timeout | Mean task reward |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for point in analysis['validation_points']:
        if not point['full']:
            continue
        stats = point['all_registered_episodes']
        values = [_value(stats, metric) for metric, _ in METRICS]
        formatted = ['N/A' if not math.isfinite(value) else f'{value:.6g}' for value in values]
        lines.append(f"| {point['training_seed']} | {point['profile']} | "
                     f"{point['at_transition']} | {stats['complete_physical_episodes']} | "
                     + ' | '.join(formatted) + ' |')
    lines.extend(['', '## Interpretation boundaries', '',
                  '- Event fractions use complete physical episodes, including true task timeout.',
                  '- External truncations, curriculum fragments and budget-stop fragments are ',
                  '  reported separately. Zero complete episodes means N/A, not a zero rate.',
                  '- Curves use the same base indices 0–29. Val300 remains a separate full result.',
                  '- Independent training N=3. Across-seed spread uses the n−1 sample SD.',
                  '- Travel time is actual elapsed time. Penalized time assigns 200 s to failures ',
                  '  and timeout; this diagnostic does not replace raw duration.',
                  '- Empty-profile clearance is N/A. Paths are control-node polylines.',
                  '- Models are final registered checkpoints; no best-curve selection.',
                  '- Fixed learned-policy cases do not establish all random-scene reachability.',
                  '', 'LOCAL Stage2 decision: **REQUIRES REVIEW OF RAW SEED EVIDENCE**.',
                  'Schedule completion alone is not scientific GO. Stage3 is not authorized here.',
                  '', 'Detailed counts, fragments, runtime fields, missing validation points and ',
                  'segment cutoffs are in `batch_analysis.json`; raw episode/update/validation ',
                  'JSONL and segment metadata remain in the batch directory.', ''])
    (output / 'ANALYSIS_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    """只读analysis为默认；固定策略环境诊断须显式选项与完整已确认批次。"""
    args = parser().parse_args(argv)
    if args.plot_only:
        if args.analysis_json is None or args.output_dir is None or args.learned_diagnostics:
            raise ValueError('plot-only必须指定analysis JSON和输出目录，且不能运行环境诊断。')
        analysis = json.loads(args.analysis_json.read_text(encoding='utf-8'))
        print(json.dumps(dict(status=analysis['status'],
                              figures=render_plots(analysis, args.output_dir))))
        return 0
    if args.analysis_json is not None:
        raise ValueError('--analysis-json只用于独立plot-only模式。')
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    from auv_risk_rl.training.mvp_analysis import analyze_batch, write_analysis

    output = args.output_dir or args.batch_root / 'analysis'
    analysis = analyze_batch(args.batch_root)
    write_analysis(output, analysis)
    write_report(analysis, output)
    if args.learned_diagnostics:
        if not args.trusted_local_models or analysis['status'] != 'BATCH_DATA_COMPLETE':
            raise ValueError('固定策略诊断须可信本机模型确认且本批数据已完整核验。')
        from auv_risk_rl.config import load_project_config
        from auv_risk_rl.training.mvp_analysis import run_learned_diagnostics

        result = run_learned_diagnostics(args.batch_root, load_project_config(args.project_config),
                                         analysis['experiment_code_commit'])
        analysis['learned_fixed_diagnostics'] = result
        write_analysis(output, analysis)
    print(json.dumps(dict(status=analysis['status'],
                          stage2_decision=analysis['stage2_scientific_decision'],
                          output_directory=str(output.resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
