"""只读汇总V1确认日志和可信本机最终Replay；不重放、不训练、不计算文件摘要。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from auv_risk_rl.rl.replay import FAILURE_TYPES
from auv_risk_rl.training.mvp_analysis import (
    BIN_WIDTH,
    Moments,
    confirmed_records,
    validation_point,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 22, 33)
CODE = '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898'
EVENTS = ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')
COMPONENTS = ('progress', 'goal', 'time', 'smoothness')
METRICS = ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss', 'alpha',
           'actor_gradient_norm', 'q1_gradient_norm', 'q2_gradient_norm',
           'alpha_gradient_norm')


def end_bin(transition: int) -> int:
    """训练transition1..25k归首区间，不把0步验证混入训练。"""
    return ((max(1, int(transition)) - 1) // BIN_WIDTH + 1) * BIN_WIDTH


class ScalarStats:
    """保留已观测均值、样本SD、范围和实际首尾；空值不是零。"""

    def __init__(self) -> None:
        self.moment = Moments()
        self.first = self.last = self.minimum = self.maximum = None
        self.total = 0.0

    def add(self, value: float) -> None:
        """只累计有限实测标量。"""
        value = float(value)
        self.moment.add(value)
        if self.first is None:
            self.first = value
        self.last = value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.total += value

    def result(self) -> dict[str, Any]:
        """写出不以空样本充当PASS或零值的结构化记录。"""
        return {**self.moment.result(), 'first': self.first, 'last': self.last,
                'minimum': self.minimum, 'maximum': self.maximum,
                'sum': self.total if self.moment.count else None}


class Episodes:
    """完整物理episode和片段分开，reward四项逐项保留。"""

    def __init__(self) -> None:
        self.events: Counter[str] = Counter()
        self.count = self.steps = self.success_episode_steps = 0
        self.saturation = self.phase = self.budget = self.external = 0
        names = ('steps', 'physical_time_s', 'reward', 'path_length_m',
                 'minimum_clearance_m', 'initial_distance_m', 'final_distance_m')
        self.scalars = {name: ScalarStats() for name in (*names, *COMPONENTS)}

    def add(self, episode: dict[str, Any]) -> None:
        """按原日志实际值累计；未记录训练距离与边界子类型不推造。"""
        self.count += 1
        self.steps += episode['steps']
        self.saturation += episode['action_saturation_count']
        self.phase += int(episode.get('phase_boundary', False))
        self.budget += int(episode.get('budget_stop', False))
        self.external += int(episode.get('external_truncation', False))
        self.events[episode['failure_type']] += 1
        if episode['complete'] and episode['failure_type'] == 'goal_success':
            self.success_episode_steps += episode['steps']
        for name, scalar in self.scalars.items():
            value = episode['reward_components'].get(name) if name in COMPONENTS \
                else episode.get(name)
            if value is not None:
                scalar.add(value)

    def result(self) -> dict[str, Any]:
        """episode结束区间统计不冒充该区间全部Replay transition分布。"""
        return dict(episodes=self.count, logged_steps=self.steps,
                    event_counts=dict(self.events), phase_boundary_fragments=self.phase,
                    budget_stop_fragments=self.budget, external_truncations=self.external,
                    action_saturation_count=self.saturation,
                    successful_episode_transitions=self.success_episode_steps,
                    terminal_success_transitions=self.events['goal_success'],
                    scalars={key: value.result() for key, value in self.scalars.items()})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """必要的小体积结果CSV；所有字段来自本轮只读汇总。"""
    if not rows:
        raise ValueError('不能把空结果写为完成的CSV。')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, dict | list) else value
                             for key, value in row.items()})


def flatten_episodes(identity: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """CSV明确事件分母与每个reward分项统计，不隐藏缺失字段。"""
    result = {**identity, **{key: value for key, value in stats.items()
                            if key not in ('scalars', 'event_counts')}}
    for event in (*EVENTS, 'none', 'external_truncation'):
        result[event] = stats['event_counts'].get(event, 0)
    for name, scalar in stats['scalars'].items():
        for metric in ('n', 'mean', 'sample_sd', 'minimum', 'maximum', 'sum'):
            result[f'{name}_{metric}'] = scalar[metric]
    return result


def check_retained_trajectory(seed: int, row: dict[str, Any],
                              episode: dict[str, Any]) -> dict[str, Any]:
    """控制节点距离与逐步reward独立累计；不是0.05s扫掠到达判定。"""
    nodes = episode['trajectory']
    if len(nodes) != episode['steps']:
        raise ValueError('选定轨迹缺失真实控制节点。')
    goal = np.asarray(episode['goal_position_ned_m'], dtype=np.float64)
    initial = np.asarray(episode['initial_position_ned_m'], dtype=np.float64)
    distances = [float(np.linalg.norm(initial - goal))]
    times = [0.0]
    sums = {name: 0.0 for name in COMPONENTS}
    discounted = {name: 0.0 for name in COMPONENTS}
    for offset, node in enumerate(nodes):
        distances.append(float(np.linalg.norm(np.asarray(node['position_ned_m']) - goal)))
        times.append(times[-1] + node['elapsed_s'])
        for name in COMPONENTS:
            sums[name] += node['reward_components'][name]
            discounted[name] += 0.999**offset * node['reward_components'][name]
    min_index = int(np.argmin(distances))
    entered = [index for index, distance in enumerate(distances) if distance <= 2.0]
    result = dict(
        training_seed=seed, profile=row['profile'], at_transition=row['at_transition'],
        validation_kind=row['validation_kind'], scenario_index=episode['index'],
        scenario_id=episode['scenario_id'], failure_type=episode['failure_type'],
        control_transitions=episode['steps'], goal_radius_m=2.0,
        initial_distance_m=distances[0], final_distance_m=distances[-1],
        minimum_control_node_distance_m=distances[min_index],
        minimum_control_node_distance_time_s=times[min_index],
        first_control_node_inside_goal_time_s=times[entered[0]] if entered else None,
        observed_control_node_goal_entry=bool(entered),
        geometry_scope='CONTROL_NODES_ONLY; not a 0.05s swept arrival verdict',
        progress_telescope_residual_m=sums['progress'] - (distances[0] - distances[-1]),
        reward_sum_residual=sum(sums.values()) - episode['reward'],
        scalar_reward_scope='undiscounted episode return and gamma=.999 component sum separate',
    )
    for name in COMPONENTS:
        result[f'{name}_undiscounted'] = sums[name]
        result[f'{name}_gamma_weighted'] = discounted[name]
        result[f'{name}_logged_total_residual'] = sums[name] - episode['reward_components'][name]
    return result


def expected_replay(episodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """固定双环境轮转和原端点恢复物理槽来源，并拒绝缺口/重叠。"""
    size = 300000
    result = dict(episode_id=np.full(size, -1, dtype=np.int64),
                  task_step=np.zeros(size, dtype=np.int64),
                  failure_type=np.zeros(size, dtype=np.uint8),
                  terminated=np.zeros(size, dtype=np.bool_),
                  successful_episode=np.zeros(size, dtype=np.bool_))
    for episode in episodes:
        steps = episode['steps']
        if steps == 0:
            continue
        indices = episode['start_transition'] - 1 + 2 * np.arange(steps)
        if indices[-1] + 1 != episode['last_transition'] or np.any(indices >= size):
            raise ValueError('episode端点不能由固定双环境轮转重建。')
        if np.any(result['episode_id'][indices] != -1):
            raise ValueError('已确认episode轨迹的global transition重复。')
        result['episode_id'][indices] = episode['episode_id']
        result['task_step'][indices] = np.arange(1, steps + 1)
        if episode['complete']:
            event = episode['failure_type']
            result['failure_type'][indices[-1]] = FAILURE_TYPES.index(event)
            result['terminated'][indices[-1]] = True
            if event == 'goal_success':
                result['successful_episode'][indices] = True
    if np.any(result['episode_id'] == -1):
        raise ValueError('确认日志未覆盖全部300000真实transition。')
    return result


def replay_audit(path: Path, seed: int, expected: dict[str, np.ndarray]) -> dict[str, Any]:
    """仅加载本项目已确认本机文件；每次释放大型状态，不复制模型/Replay。"""
    import torch

    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['code_version'] != CODE or state['config']['training_seed'] != seed
            or state['scheduler']['transitions'] != 300000):
        raise ValueError('最终可信checkpoint实际身份与V1确认记录不符。')
    agent, replay = state['agent'], state['agent']['replay']
    if replay['size'] != 300000 or replay['cursor'] != 300000 or replay['capacity'] != 500000:
        raise ValueError('Replay存在覆盖/长度差异，不能按global transition直接解释槽。')
    fields = ('episode_id', 'task_step', 'failure_type', 'terminated', 'truncated',
              'reward', 'nominal_action', 'executed_action')
    arrays = {name: np.concatenate([replay['chunks'][block][name]
                                   for block in sorted(replay['chunks'])])[:replay['size']]
              for name in fields}
    agreement = {name: bool(np.array_equal(arrays[name], expected[name]))
                 for name in ('episode_id', 'task_step', 'failure_type', 'terminated')}
    if not all(agreement.values()) or bool(arrays['truncated'].any()):
        raise ValueError('最终Replay和确认日志的episode/时步/物理终止关系不一致。')
    agreement['nominal_equals_executed'] = bool(np.array_equal(
        arrays['nominal_action'], arrays['executed_action']))
    bins = []
    for stop in range(25000, 300001, 25000):
        selection = slice(stop - 25000, stop)
        counts = Counter(FAILURE_TYPES[int(value)] for value in arrays['failure_type'][selection])
        bins.append(dict(
            training_seed=seed, profile='obstacle_free' if stop <= 100000 else 'cv_train_v1',
            bin_first_transition=stop - 24999, bin_last_transition=stop,
            transitions=25000,
            successful_episode_transitions=int(expected['successful_episode'][selection].sum()),
            terminal_success_transitions=counts['goal_success'],
            terminal_event_counts={event: counts[event] for event in EVENTS},
            replay_reward_mean=float(arrays['reward'][selection].astype(np.float64).mean()),
            replay_reward_sum=float(arrays['reward'][selection].astype(np.float64).sum()),
            current_replay_scope='retained historical transitions; not historical RNG or batches'))
    result = dict(
        training_seed=seed, path=path.relative_to(ROOT).as_posix(),
        checkpoint_format=state['format'], torch_version=state['torch_version'],
        checkpoint_transition=state['scheduler']['transitions'], counters=agent['counters'],
        replay_capacity=replay['capacity'], replay_size=replay['size'], cursor=replay['cursor'],
        chunk_size=replay['chunk_size'], allocated_chunks=len(replay['chunks']),
        allocated_raw_array_bytes=sum(array.nbytes for chunk in replay['chunks'].values()
                                      for array in chunk.values()),
        identity_and_log_agreement=agreement,
        alpha=float(agent['log_alpha'].exp()),
        effective_sac_config=agent['config'], final_full_critic_available=True,
        historical_100k_critic_available=False, historical_100k_replay_rng_available=False,
        historical_sampled_minibatches_available=False, transition_bins=bins)
    del state, agent, replay, arrays
    gc.collect()
    return result


def write_readout(output: Path, result: dict[str, Any]) -> None:
    """将离线事实与不能从日志回答的问题分开，不预判学习率因果或科学GO。"""
    lines = [
        '# V1确认日志离线核对',
        '',
        f"生成UTC：{result['generated_at']}；实验代码：`{CODE}`。",
        '仅只读既有日志及可信最终checkpoint；本分析新增环境采样/梯度更新均为0。',
        '原V1记录不改写。每条采用日志经segment有效序号及身份检查；未确认尾部不混入。',
        '',
        '## 固定无障碍monitor30（分母始终30）',
        '',
        '| seed | 0 | 25k | 50k | 75k | 100k |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for seed in SEEDS:
        rows = sorted((row for row in result['validation_points']
                       if row['training_seed'] == seed and row['profile'] == 'obstacle_free'
                       and row['scope'] == 'paired_monitor30'),
                      key=lambda row: row['at_transition'])
        lines.append(f"| {seed} | " + ' | '.join(str(row['goal_success']) for row in rows) + ' |')
    lines += [
        '',
        '三seed均存在100k之前相对于50k或75k的退化；100k之后的CV切换不能解释该事实。',
        '100k的monitor子集与完整Val300分开保存，不把30场景说成300。',
        '',
        '## 成功相关Replay经验：整个成功episode与成功终点区分',
        '',
        '| seed | 首100k成功episode全部transition | 仅成功终点 | 首100k占比 |',
        '|---|---:|---:|---:|',
    ]
    for replay in result['final_replay_audits']:
        prefix = [row for row in replay['transition_bins'] if row['profile'] == 'obstacle_free']
        total = sum(row['successful_episode_transitions'] for row in prefix)
        terminal = sum(row['terminal_success_transitions'] for row in prefix)
        lines.append(f"| {replay['training_seed']} | {total} | {terminal} | {total / 1000:.3f}% |")
    lines += [
        '',
        '当前500k容量/300k有效记录、cursor300k，尚未发生覆盖；首100k原transition仍保留。',
        '固定双环境轮转和episode端点逐槽重建后，episode_id/task_step/failure_type/',
        'terminated与实际Replay全部完全一致；无external truncation，nominal=executed。',
        '此检查不恢复100k Replay RNG、已抽过的minibatch或历史critic。',
        '仅数17/12/15成功终点便声称“几乎无成功相关经验”不受此数据支持。',
        '',
        '## 更新与温度事实（不能代替Q校准）',
        '',
        '| seed | alpha@25k | alpha@50k | alpha@75k | alpha@100k | Q1均方损失首/末25k |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for seed in SEEDS:
        rows = [row for row in result['update_bins']
                if row['training_seed'] == seed and row['profile'] == 'obstacle_free']
        alpha = ' | '.join(f"{row['alpha_last']:.7f}" for row in rows)
        lines.append(f"| {seed} | {alpha} | "
                     f"{rows[0]['q1_loss_mean']:.4f}/{rows[-1]['q1_loss_mean']:.4f} |")
    lines += [
        '',
        '全量确认更新是从transition10000起每新transition一次，实际290001次/seed。',
        'loss/alpha/梯度范数均为有限记录；loss有限不证明critic正确或策略偏好正确。',
        '这些时序支持优化动态是待检验机制，但不能独立证明公共学习率过大。',
        '',
        '## 独立目标距离与reward累计',
        '',
    ]
    checks = result['selected_control_node_trajectory_checks']
    progress = max(abs(row['progress_telescope_residual_m']) for row in checks)
    reward = max(abs(row['reward_sum_residual']) for row in checks)
    inside_without_success = sum(row['observed_control_node_goal_entry']
                                 and row['failure_type'] != 'goal_success' for row in checks)
    lines += [
        f'{len(checks)}条已有选定控制节点轨迹，最大进展首尾抵消残差{progress:.4g}m，',
        f'独立四项reward求和与日志总return最大差{reward:.4g}。',
        f'非成功但控制节点实际进入2m球的轨迹数：{inside_without_success}。',
        '未折扣进展具有首尾抵消；gamma=.999加权进展另计，不能混为同一目标。',
        '已有轨迹不是0.05s分段状态，不能排除节点之间目标球进入；需要本轮冻结重放。',
        '旧训练日志没有目标最小距离/最终距离/边界子类型；不以0伪造这些缺失值。',
        '',
        '## 实际随机种子（用于C/R1可比性核对）',
        '',
        '| seed | 初始化 | Actor流 | Replay流 | 场景root | 首环境seed |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for identity in result['effective_seed_identities']:
        sac = identity['config']['sac']
        values = [identity['training_seed'], sac['initialization_seed'], sac['actor_seed'],
                  sac['replay_seed'], identity['scenario_root_seed'],
                  identity['first_environment_seed']]
        lines.append('| ' + ' | '.join(str(value) for value in values) + ' |')
    lines += [
        '',
        '50k/75k权重没有保存。100k/300k小模型仅Actor；只有最终300k恢复点有critics。',
        '不能拿100kActor替代中间模型，不能用300kcritic冒称100kcritic。',
        '所有episode-end区间统计归入episode末transition所在25k区间；其完整reward/',
        '步数可能跨区间。真正transition级来源/终止频率单列Replay CSV，不能混淆分母。',
        '',
        '必要结果：v1_offline_analysis.json及五份CSV。局部数据只能回答相应范围；',
        '是否存在L1错误、是否进入L2分支及Stage2科学判断由独立协议/重放证据共同决定。',
    ]
    (output / 'V1_OFFLINE_READOUT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze(batch: Path, output: Path, trusted: bool) -> dict[str, Any]:
    """一趟逐行处理870003更新；仅确认日志进入结果。"""
    training, validation, updates, retained, replay_results, identities = [], [], [], [], [], []
    source_checks = {}
    inventory = []
    for seed in SEEDS:
        diagnostics: dict[str, Any] = {}
        episodes = []
        train_groups: dict[tuple[str, int, bool], Episodes] = defaultdict(Episodes)
        update_groups: dict[int, dict[str, ScalarStats]] = {}
        update_count = 0
        for kind, row in confirmed_records(batch, seed, CODE, diagnostics):
            if kind == 'episode':
                episodes.append(row)
                key = (row['task_profile'], end_bin(row['last_transition']), row['complete'])
                train_groups[key].add(row)
                if row['episode_id'] == 0:
                    identities.append(dict(training_seed=seed, config=row['config'],
                                           scenario_root_seed=row['scenario_root_seed'],
                                           first_environment_seed=row['warmup']['root_seed']))
            elif kind == 'update':
                transition = row['transition']
                if transition != 10000 + update_count:
                    raise ValueError('确认更新日志不是10000起每transition一次连续更新。')
                update_count += 1
                values = update_groups.setdefault(end_bin(transition),
                                                   {key: ScalarStats() for key in METRICS})
                for name, scalar in values.items():
                    scalar.add(row['metrics'][name])
            else:
                validation_point(row)
                for scope in ('all_registered', 'paired_monitor30'):
                    selected = row['episodes'] if scope == 'all_registered' else \
                        [episode for episode in row['episodes'] if episode['index'] < 30]
                    acc = Episodes()
                    for episode in selected:
                        acc.add(episode)
                    validation.append(flatten_episodes(
                        dict(training_seed=seed, profile=row['profile'],
                             at_transition=row['at_transition'],
                             validation_kind=row['validation_kind'], scope=scope), acc.result()))
                retained.extend(check_retained_trajectory(seed, row, episode)
                                for episode in row['episodes'] if 'trajectory' in episode)
        if update_count != 290001:
            raise ValueError('确认V1每seed完整update数不符。')
        for (profile, stop, complete), accumulator in sorted(train_groups.items()):
            training.append(flatten_episodes(
                dict(training_seed=seed, profile=profile, bin_last_transition=stop,
                     complete=complete,
                     bin_scope='episode final transition; whole episode assigned'),
                accumulator.result()))
        for stop, values in sorted(update_groups.items()):
            item = dict(training_seed=seed, profile='obstacle_free' if stop <= 100000
                        else 'cv_train_v1', bin_first_transition=stop - 24999,
                        bin_last_transition=stop, updates=values['alpha'].moment.count)
            for name, scalar in values.items():
                for field, value in scalar.result().items():
                    item[f'{name}_{field}'] = value
            updates.append(item)
        source_checks[str(seed)] = diagnostics
        expected = expected_replay(episodes)
        if trusted:
            replay_results.append(replay_audit(batch / f'seed_{seed}/latest_resume.pt',
                                               seed, expected))
        for transition in (50000, 75000, 100000, 300000):
            path = batch / f'models/seed_{seed}_{transition}.pt'
            inventory.append(dict(training_seed=seed, transition=transition,
                                  path=path.relative_to(ROOT).as_posix(), exists=path.exists(),
                                  content='Actor only' if path.exists() else 'NOT SAVED',
                                  critic_available=False))
    result = dict(
        task='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
        generated_at=datetime.now(UTC).isoformat(), analysis_kind='READ_ONLY_CONFIRMED_V1_LOGS',
        experiment_code_commit=CODE, batch_root=batch.as_posix(),
        scientific_training_transitions_performed=0, sac_updates_performed=0,
        rollout_transitions_performed=0, source_checks=source_checks,
        effective_seed_identities=identities, model_inventory=inventory,
        training_episode_bins=training, validation_points=validation, update_bins=updates,
        selected_control_node_trajectory_checks=retained, final_replay_audits=replay_results,
        limitations=[
            'Training episode records lack final/minimum goal distance and boundary subtype.',
            'Validation trajectories lack eight-state pitch/rate data; boundary subtype unknown.',
            'Control-node goal minimum does not establish 0.05s swept goal entry or missed event.',
            'Retained 100k Replay prefix is original data, not historical RNG/critic/minibatches.',
            '50k/75k weights and 100k critics were not saved; cannot recreate them by relabeling.',
            'Success-related transitions include all successful episode steps, not just terminal.',
            'Discounted progress and undiscounted episode return are separate scalar statistics.',
        ])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'v1_training_episode_bins.csv', training)
    write_csv(output / 'v1_validation_points.csv', validation)
    write_csv(output / 'v1_update_bins.csv', updates)
    write_csv(output / 'v1_retained_trajectory_checks.csv', retained)
    if replay_results:
        bins = [row for audit in replay_results for row in audit['transition_bins']]
        write_csv(output / 'v1_replay_transition_bins.csv', bins)
    (output / 'v1_offline_analysis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_readout(output, result)
    return result


def main() -> None:
    """显式本机可信checkpoint确认，默认不解除反序列化边界。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=ROOT / 'results/stage2_b0_mvp_v1')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'offline')
    parser.add_argument('--trusted-local-checkpoints', action='store_true')
    args = parser.parse_args()
    result = analyze(args.batch_root.resolve(), args.output_dir.resolve(),
                     args.trusted_local_checkpoints)
    print(json.dumps(dict(status='ANALYSIS_WRITTEN',
                          training_episode_bins=len(result['training_episode_bins']),
                          validation_scopes=len(result['validation_points']),
                          update_bins=len(result['update_bins']),
                          retained_trajectories=len(result['selected_control_node_trajectory_checks']),
                          replay_audits=len(result['final_replay_audits']),
                          scientific_training_performed=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
