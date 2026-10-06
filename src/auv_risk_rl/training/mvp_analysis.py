"""预登记B0 MVP的流式科学记录汇总；不执行训练、不挑最佳checkpoint。"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRATION_ID = 'STAGE2_B0_MVP_BATCH_V1'
METHOD = 'B0_FULL_STATE_ORDINARY_SAC'
SEEDS = (11, 22, 33)
PROFILES = ('obstacle_free', 'cv_train_v1')
PHYSICAL_EVENTS = ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')
BIN_WIDTH = 25000
TASK_HORIZON_SECONDS = 200.0
EXPECTED_VALIDATIONS = {
    'obstacle_free': ((0, False), (25000, False), (50000, False), (75000, False),
                      (100000, True)),
    'cv_train_v1': tuple((step, False) for step in range(100000, 300000, 25000))
    + ((300000, True),),
}


class Moments:
    """在线均值/样本标准差；缺少样本时保留null，不伪造零值。"""

    def __init__(self) -> None:
        self.count = 0
        self.mean = self.m2 = 0.0

    def add(self, value: float) -> None:
        """Welford累计已观测的有限标量。"""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('不能用非有限结果构造科学汇总。')
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def result(self) -> dict[str, Any]:
        """标准差的分母为n-1；只有一个样本时样本标准差不适用。"""
        return dict(n=self.count, mean=self.mean if self.count else None,
                    sample_sd=math.sqrt(max(0.0, self.m2) / (self.count - 1))
                    if self.count > 1 else None)


class EpisodeAccumulator:
    """真实完整物理episode作为事件分母，截断和预算/课程片段另计。"""

    def __init__(self) -> None:
        self.events: Counter[str] = Counter()
        self.complete_count = self.fragment_count = self.total_steps = 0
        self.external_count = self.budget_stop_count = self.phase_boundary_count = 0
        self.saturation_count = self.complete_steps = 0
        names = ('reward', 'progress_reward', 'travel_time_s', 'penalized_time_s',
                 'progress_m', 'path_length_m', 'successful_path_length_m',
                 'minimum_clearance_m')
        self.moments = {name: Moments() for name in names}
        self.clearance_episode_count = self.near_miss_count = 0

    def add(self, episode: dict[str, Any]) -> None:
        """不从truncated单一布尔推断timeout；按规范的failure_type识别真实终止。"""
        steps = episode['steps']
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('episode steps必须是实际非负控制transition数。')
        complete = episode['complete']
        if not isinstance(complete, bool):
            raise ValueError('complete必须显式为bool。')
        event = episode['failure_type']
        self.total_steps += steps
        if not complete:
            if event not in ('none', 'external_truncation'):
                raise ValueError('未完整episode的物理事件记录不一致。')
            self.fragment_count += 1
            self.external_count += int(event == 'external_truncation')
            self.budget_stop_count += int(episode.get('budget_stop', False))
            self.phase_boundary_count += int(episode.get('phase_boundary', False))
            return
        if event not in PHYSICAL_EVENTS:
            raise ValueError('完整episode必须有真实物理任务事件。')
        self.complete_count += 1
        self.complete_steps += steps
        self.events[event] += 1
        saturation = episode['action_saturation_count']
        if isinstance(saturation, bool) or not isinstance(saturation, int) \
                or not 0 <= saturation <= steps:
            raise ValueError('动作饱和计数不是按transition记录的合法计数。')
        self.saturation_count += saturation
        reward, duration, path = (float(episode[key]) for key in
                                 ('reward', 'physical_time_s', 'path_length_m'))
        if duration < 0 or path < 0:
            raise ValueError('真实时间和路径长度不能为负。')
        self.moments['reward'].add(reward)
        self.moments['progress_reward'].add(episode['reward_components']['progress'])
        if 'progress_m' in episode:
            self.moments['progress_m'].add(episode['progress_m'])
        self.moments['travel_time_s'].add(duration)
        self.moments['penalized_time_s'].add(
            duration if event == 'goal_success' else TASK_HORIZON_SECONDS)
        self.moments['path_length_m'].add(path)
        if event == 'goal_success':
            self.moments['successful_path_length_m'].add(path)
        clearance = episode['minimum_clearance_m']
        if clearance is not None:
            self.moments['minimum_clearance_m'].add(clearance)
            self.clearance_episode_count += 1
            self.near_miss_count += int(event != 'collision' and float(clearance) < 0.5)

    def result(self) -> dict[str, Any]:
        """所有事件率共用完整物理episode分母；无障碍间距不适用。"""
        count = self.complete_count
        rates = {event: self.events[event] / count if count else None
                 for event in PHYSICAL_EVENTS}
        return dict(
            complete_physical_episodes=count,
            event_counts={event: self.events[event] for event in PHYSICAL_EVENTS},
            success_rate=rates['goal_success'], collision_rate=rates['collision'],
            boundary_rate=rates['operational_boundary_failure'], timeout_rate=rates['task_horizon'],
            denominator='complete physical episodes; external/phase/budget fragments excluded',
            fragments=self.fragment_count, external_truncations=self.external_count,
            budget_stop_fragments=self.budget_stop_count,
            phase_boundary_fragments=self.phase_boundary_count,
            logged_training_steps=self.total_steps,
            action_saturation_transition_rate=(self.saturation_count / self.complete_steps
                                               if self.complete_steps else None),
            clearance_applicable_episodes=self.clearance_episode_count,
            near_miss_count=self.near_miss_count,
            near_miss_rate_on_clearance_applicable=(
                self.near_miss_count / self.clearance_episode_count
                if self.clearance_episode_count else None),
            path_length_definition='CONTROL_NODE_POLYLINE',
            moments={name: moment.result() for name, moment in self.moments.items()},
        )


def summarize_episodes(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """方便独立验证/小型测试，主训练日志仍逐行处理。"""
    accumulator = EpisodeAccumulator()
    for episode in episodes:
        accumulator.add(episode)
    return accumulator.result()


def _contained_path(root: Path, relative: str) -> Path:
    """segment必须留在本批目录，拒绝意外读入其他任务或个人材料。"""
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def confirmed_records(root: Path, seed: int, code_version: str,
                      diagnostics: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """只采用segment登记的安全checkpoint日志cutoff；重算尾部不进入权威结果。"""
    inventory = root / f'seed_{seed}' / 'segments.json'
    if not inventory.exists():
        diagnostics['missing_segment_inventory'] = True
        return
    segments = json.loads(inventory.read_text(encoding='utf-8'))
    if not isinstance(segments, list):
        raise ValueError('segments.json必须是明确登记的segment列表。')
    seen: set[int] = set()
    diagnostics.update(confirmed_rows={}, excluded_unconfirmed_rows={}, segments=segments)
    for segment in segments:
        segment_id = segment['segment_id']
        cutoff = segment['valid_log_sequence']
        if cutoff is not None and (isinstance(cutoff, bool) or not isinstance(cutoff, int)
                                   or cutoff < 0):
            raise ValueError('有效日志cutoff必须是实际非负整数或null。')
        if segment['status'] not in ('RUNNING', 'PAUSED', 'COMPLETED', 'FAILED'):
            raise ValueError('未知segment状态。')
        directory = _contained_path(root, segment['path'])
        for kind in ('episode', 'update', 'validation'):
            path = directory / (kind + '.jsonl')
            if not path.exists():
                continue
            with path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f'原始日志损坏: {path}:{line_number}') from error
                    sequence = row['log_sequence']
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                        raise ValueError('log_sequence必须是实际正整数。')
                    if cutoff is None or sequence > cutoff:
                        target = diagnostics['excluded_unconfirmed_rows']
                        target[kind] = target.get(kind, 0) + 1
                        continue
                    expected = dict(registration_id=REGISTRATION_ID, run_kind='scientific_training',
                                    method=METHOD, training_seed=seed, segment_id=segment_id,
                                    code_version=code_version)
                    if any(row.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'日志身份不匹配，不能混入本批结果: {path}:{line_number}')
                    if sequence in seen:
                        raise ValueError('有效segment的log_sequence重复，拒绝重复计数。')
                    seen.add(sequence)
                    target = diagnostics['confirmed_rows']
                    target[kind] = target.get(kind, 0) + 1
                    row['_log_source'] = dict(path=path.relative_to(root).as_posix(),
                                               line_number=line_number,
                                               log_sequence=sequence,
                                               valid_log_sequence=cutoff)
                    yield kind, row


def validation_point(row: dict[str, Any]) -> dict[str, Any]:
    """Val300全部保留，另用相同base索引0..29作可比较的monitor曲线。"""
    profile = row['profile']
    if profile not in PROFILES or row.get('training_state_unchanged') is not True:
        raise ValueError('验证profile或训练隔离证据不合法。')
    if row.get('validation_root_seed') != 20261006 or not isinstance(row['full'], bool):
        raise ValueError('固定验证root seed/full标识不匹配预登记。')
    episodes = row['episodes']
    expected = 300 if row['full'] else 30
    if row['count'] != expected or len(episodes) != expected:
        raise ValueError('验证原始episode数不等于预登记30/300。')
    indices = [episode['index'] for episode in episodes]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError('固定验证索引必须为整数。')
    if sorted(indices) != list(range(expected)):
        raise ValueError('验证必须完整覆盖固定base索引，不可选取有利场景。')
    if any(episode.get('complete') is not True for episode in episodes):
        raise ValueError('正式验证不得含外部工程截断或未完成episode。')
    selected = [episode for episode, index in zip(episodes, indices, strict=True) if index < 30]
    return dict(training_seed=row['training_seed'], profile=profile,
                at_transition=row['at_transition'], evaluation_key=row['evaluation_key'],
                evaluation_kind='Val300' if row['full'] else 'monitor30',
                full=row['full'], validation_root_seed=row['validation_root_seed'],
                evaluation_env_transitions=row.get('evaluation_env_transitions'),
                warmup_control_transitions=row.get('warmup_control_transitions'),
                wall_clock_seconds=row.get('wall_clock_seconds'),
                trajectory_source=row.get('_log_source'),
                retained_trajectory_indices=[episode['index'] for episode in episodes
                                             if 'trajectory' in episode],
                all_registered_episodes=summarize_episodes(episodes),
                paired_monitor30=summarize_episodes(selected),
                base_indices=list(range(expected)))


def _aggregate_points(points: list[dict[str, Any]], scope: str = 'paired_monitor30',
                      ) -> list[dict[str, Any]]:
    """跨seed均值/样本SD只使用原始seed点，不把episode充当独立训练重复。"""
    groups: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for point in points:
        key = (point['profile'], point['at_transition'])
        group = groups.setdefault(key, {})
        if point['training_seed'] in group:
            raise ValueError('同seed/profile/transition存在重复验证结果。')
        group[point['training_seed']] = point
    fields = ('success_rate', 'collision_rate', 'boundary_rate', 'timeout_rate')
    result = []
    for (profile, transition), group in sorted(groups.items()):
        metrics = {}
        for field in fields:
            moment = Moments()
            for point in group.values():
                value = point[scope][field]
                if value is not None:
                    moment.add(value)
            metrics[field] = moment.result()
        moment = Moments()
        for point in group.values():
            value = point[scope]['moments']['reward']['mean']
            if value is not None:
                moment.add(value)
        metrics['reward'] = moment.result()
        result.append(dict(profile=profile, at_transition=transition,
                           training_seeds=sorted(group), independent_training_seed_count=len(group),
                           expected_training_seed_count=3, scope=scope, metrics=metrics))
    return result


def analyze_batch(root: str | Path) -> dict[str, Any]:
    """流式处理全量原始JSONL；未完成日程保持NOT_COMPLETE，不生成科学GO。"""
    root = Path(root)
    state = json.loads((root / 'batch_state.json').read_text(encoding='utf-8'))
    if state['registration_id'] != REGISTRATION_ID:
        raise ValueError('批次登记身份不符。')
    code = state['experiment_code_commit']
    if not isinstance(code, str) or not code:
        raise ValueError('必须有实际被执行的代码提交标识。')
    summaries, points, training_bins, updates = [], [], [], []
    completed_jobs = {(job['seed'], job['stop']) for job in state['completed_jobs']}
    for seed in SEEDS:
        diagnostics: dict[str, Any] = {}
        accumulators = {profile: EpisodeAccumulator() for profile in PROFILES}
        bins: dict[tuple[str, int], EpisodeAccumulator] = {}
        update_bins: dict[int, dict[str, Moments]] = {}
        update_count = last_update_transition = 0
        for kind, row in confirmed_records(root, seed, code, diagnostics):
            if kind == 'episode':
                profile = row['task_profile']
                if profile not in PROFILES:
                    raise ValueError('训练episode使用了未登记profile。')
                transition = row['at_transition']
                accumulators[profile].add(row)
                end = ((max(1, transition) - 1) // BIN_WIDTH + 1) * BIN_WIDTH
                bins.setdefault((profile, end), EpisodeAccumulator()).add(row)
            elif kind == 'validation':
                points.append(validation_point(row))
            else:
                transition = row['transition']
                if (isinstance(transition, bool) or not isinstance(transition, int)
                        or not 10000 <= transition <= 300000
                        or transition <= last_update_transition):
                    raise ValueError('权威完整update次序/起步边界不合法。')
                update_count += 1
                last_update_transition = transition
                end = ((transition - 1) // BIN_WIDTH + 1) * BIN_WIDTH
                metrics = update_bins.setdefault(end, {})
                for name, value in row['metrics'].items():
                    metrics.setdefault(name, Moments()).add(value)
        for (profile, end), accumulator in sorted(bins.items()):
            training_bins.append(dict(training_seed=seed, profile=profile, bin_end_transition=end,
                                      **accumulator.result()))
        for end, metrics in sorted(update_bins.items()):
            updates.append(dict(training_seed=seed, bin_end_transition=end,
                                metrics={name: value.result() for name, value in metrics.items()}))
        seed_state = state['seeds'].get(str(seed), {})
        transitions = seed_state.get('transitions')
        recorded_updates = seed_state.get('updates')
        logged_steps = sum(value.total_steps for value in accumulators.values())
        seed_points = [point for point in points if point['training_seed'] == seed]
        observed = {(point['profile'], point['at_transition'], point['full'])
                    for point in seed_points}
        required = {(profile, step, full) for profile, schedule in EXPECTED_VALIDATIONS.items()
                    for step, full in schedule}
        unexpected = observed - required
        if unexpected:
            raise ValueError(f'验证发生在未登记的位置: {sorted(unexpected)}')
        missing_validations = sorted(required - observed)
        final_job = all((seed, stop) in completed_jobs for stop in (100000, 300000))
        profiles_complete = (accumulators['obstacle_free'].total_steps == 100000
                             and accumulators['cv_train_v1'].total_steps == 200000)
        complete = (final_job and transitions == 300000 and recorded_updates == 290001
                    and update_count == 290001 and logged_steps == 300000
                    and profiles_complete and not missing_validations)
        summaries.append(dict(
            training_seed=seed, status='COMPLETE' if complete else 'NOT_COMPLETE',
            checkpoint_recorded_transitions=transitions,
            checkpoint_recorded_updates=recorded_updates,
            authoritative_logged_steps=logged_steps, authoritative_logged_updates=update_count,
            unrepresented_in_episode_log_steps=(transitions - logged_steps
                                                if isinstance(transitions, int) else None),
            profiles={profile: accumulator.result()
                      for profile, accumulator in accumulators.items()},
            segment_diagnostics=diagnostics, runtime_record=seed_state,
            missing_registered_validations=[dict(profile=profile, transition=step, full=full)
                                            for profile, step, full in missing_validations],
        ))
    complete = (all(row['status'] == 'COMPLETE' for row in summaries)
                and state['status'] in ('COMPLETED', 'BATCH_COMPLETED'))
    return dict(
        created_at=datetime.now(UTC).isoformat(), registration_id=REGISTRATION_ID,
        batch_root=str(root.resolve()),
        experiment_code_commit=code, batch_state_status=state['status'],
        status='BATCH_DATA_COMPLETE' if complete else 'NOT_COMPLETE',
        stage2_scientific_decision='REQUIRES_REVIEW_OF_RAW_SEED_EVIDENCE',
        independent_training_seeds=list(SEEDS), seed_summaries=summaries,
        training_bins=training_bins, update_bins=updates,
        validation_points=sorted(points, key=lambda p: (p['training_seed'], p['profile'],
                                                        p['at_transition'])),
        validation_seed_mean_sd=_aggregate_points(points),
        val300_seed_mean_sd=_aggregate_points(
            [point for point in points if point['full']], 'all_registered_episodes'),
        definitions=dict(
            event_denominator='complete physical episode; not external/phase/budget fragments',
            progress='frozen task reward progress component; not success rate',
            penalized_time='success actual duration; any physical failure/timeout 200 seconds',
            curve='same fixed base indices 0..29; Val300 separately retained',
            seed_statistics='raw seed points, mean, n-1 sample SD; N=3 independent training seeds',
            primary_checkpoint='final 300k; no best-checkpoint selection',
            unreachable_claim='fixed diagnostic cases do not prove all train-v1 samples reachable',
        ),
    )


def write_analysis(output_dir: str | Path, analysis: dict[str, Any]) -> None:
    """保留可核对JSON与精简CSV，不复制模型、Replay或逐周期训练数据。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'batch_analysis.json').write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8')
    rows = []
    for point in analysis['validation_points']:
        for scope in ('all_registered_episodes', 'paired_monitor30'):
            stats = point[scope]
            rows.append(dict(
                seed=point['training_seed'], profile=point['profile'],
                transition=point['at_transition'], evaluation_kind=point['evaluation_kind'],
                scope=scope, n_complete=stats['complete_physical_episodes'],
                success_rate=stats['success_rate'], collision_rate=stats['collision_rate'],
                boundary_rate=stats['boundary_rate'], timeout_rate=stats['timeout_rate'],
                reward_mean=stats['moments']['reward']['mean'],
                penalized_time_mean_s=stats['moments']['penalized_time_s']['mean'],
                successful_path_mean_m=stats['moments']['successful_path_length_m']['mean'],
            ))
    fields = ['seed', 'profile', 'transition', 'evaluation_kind', 'scope', 'n_complete',
              'success_rate', 'collision_rate', 'boundary_rate', 'timeout_rate', 'reward_mean',
              'penalized_time_mean_s', 'successful_path_mean_m']
    with (output / 'validation_points.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_actor_model(payload: dict[str, Any], seed: int, transition: int,
                         code_version: str) -> str:
    """只接受本登记的可信本机普通B0模型和冻结生产参数，不接收工程smoke模型。"""
    expected = dict(format='b0-mvp-actor-v1', registration_id=REGISTRATION_ID,
                    method=METHOD, run_kind='scientific_training', seed=seed,
                    transition=transition, code_version=code_version)
    if seed not in SEEDS or transition not in (100000, 300000):
        raise ValueError('策略诊断只使用登记seed及100k/300k模型。')
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('轻量模型身份与本批代码/seed/阶段不匹配。')
    parameters = dict(gamma=0.999, tau=0.005, learning_rate=3e-4, batch_size=256,
                      replay_capacity=500000, learning_starts=10000, utd=1,
                      initial_alpha=0.2, target_entropy=-3.0)
    config = payload['sac_config']
    if any(config.get(key) != value for key, value in parameters.items()):
        raise ValueError('策略模型的SAC生产配置不匹配预登记。')
    device = config.get('device')
    if device not in ('cuda', 'cuda:0'):
        raise ValueError('登记目标设备为CUDA，拒绝静默设备回退。')
    if not isinstance(payload.get('actor'), dict) or not payload['actor']:
        raise ValueError('轻量模型缺少实际Actor参数。')
    return device


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """必要证据一次写入，不把非有限值序列化成假合法结果。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _learned_case(case: Any, project: Any, actor: Any, device: str,
                  destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """每固定例只执行一次确定性策略；完整轨迹独立保存，不调用LOS或学习模块。"""
    import numpy as np
    import torch
    from scripts.run_b0_reachability import (
        MAX_TRANSITIONS_PER_CASE,
        _node,
        _obstacle_records,
        _state_record,
        state_is_legal,
    )

    from auv_risk_rl.env.b0_navigation import B0NavigationEnv

    output = destination / case.case_id
    output.mkdir(parents=True, exist_ok=True)
    summary_path, marker = output / 'case.json', output / 'attempt.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise ValueError('已有策略诊断证据身份不匹配，拒绝覆盖。')
        if previous['status'] != 'COMPLETE':
            raise RuntimeError('该固定案例已有失败证据，不自动重复尝试。')
        return previous
    if marker.exists():
        raise RuntimeError('该固定案例已有未完成尝试，需人工审查，不自动重跑。')
    # 先登记尝试，再执行环境；进程中断不能在下次运行悄悄变成第二次尝试。
    with marker.open('x', encoding='utf-8') as stream:
        json.dump(dict(metadata, case_id=case.case_id, attempt_count=1,
                       maximum_attempts=1, created_at=datetime.now(UTC).isoformat()), stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
    result: dict[str, Any] = dict(
        metadata, case_id=case.case_id, environmental_seed=case.seed,
        run_kind='learned_policy_fixed_case_diagnostic', split='learned_fixed_diagnostic',
        initial_state=_state_record(case.initial_state()),
        initial_obstacles=_obstacle_records(case.obstacles()),
        goal_position_ned_m=case.goal().tolist(), attempt_count=1,
        maximum_attempts=1, maximum_transitions=MAX_TRANSITIONS_PER_CASE,
        diagnostic_env_transitions=0, diagnostic_warmup_control_transitions=0,
        diagnostic_sac_updates=0, training_replay_writes=0,
        complete=False, status='NOT_COMPLETE',
        trajectories_file='trajectory.csv', obstacle_trajectories_file='obstacles.csv',
    )
    env = B0NavigationEnv(project, case.initial_state(), case.obstacles(), case.goal(),
                          f'learned-fixed-{case.case_id}')
    rows, obstacles = [], []
    before = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
    try:
        observation, warmup = env.reset(seed=case.seed, options={'external_max_steps': None})
        result.update(warmup=warmup, formal_initial_state=_state_record(env.world.auv_state),
                      formal_initial_obstacles=_obstacle_records(env.world.obstacle_states))
        result['diagnostic_warmup_control_transitions'] = round(
            warmup['warmup_duration_s'] / project.dynamics.control_dt_s)
        legal = state_is_legal(env.world.auv_state, project)
        duration = reward_sum = path = 0.0
        minimum_clearance = None
        reward_parts: dict[str, float] = {}
        saturation = 0
        rows.append(_node(case.case_id, env, None, None, None))
        for _ in range(MAX_TRANSITIONS_PER_CASE):
            if not np.all(np.isfinite(observation)):
                raise FloatingPointError('固定策略诊断观察含非有限值。')
            with torch.inference_mode():
                batch = torch.as_tensor(observation, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                action = actor.deterministic(batch)[0].cpu().numpy()
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
                raise FloatingPointError('固定策略诊断动作非有限或超出冻结范围。')
            start = env.world.auv_state.position_ned_m.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            result['diagnostic_env_transitions'] += 1
            if not math.isfinite(reward):
                raise FloatingPointError('固定策略诊断reward非有限。')
            if not np.array_equal(action, info['executed_action_normalized']):
                raise RuntimeError('固定B0策略诊断不允许执行过滤。')
            legal &= state_is_legal(env.world.auv_state, project)
            duration += info['elapsed_s']
            reward_sum += reward
            path += float(np.linalg.norm(env.world.auv_state.position_ned_m-start))
            saturation += int(np.any(np.abs(action) >= 1.0-1.0e-6))
            for name, value in info['reward_components'].items():
                reward_parts[name] = reward_parts.get(name, 0.0) + value
            clearance = info['minimum_clearance']
            if math.isfinite(clearance):
                minimum_clearance = (clearance if minimum_clearance is None
                                     else min(minimum_clearance, clearance))
            rows.append(_node(case.case_id, env, action, reward, info))
            obstacles.extend(dict(case_id=case.case_id,
                                  control_step=env.world.control_step_index,
                                  timestamp_s=env.world.timestamp_s, **item)
                             for item in _obstacle_records(env.world.obstacle_states))
            if terminated or truncated:
                if truncated or info['failure_type'] not in PHYSICAL_EVENTS:
                    raise RuntimeError('固定策略诊断仅允许实际任务终止，不允许工程截断。')
                result.update(complete=True, status='COMPLETE',
                              failure_type=info['failure_type'],
                              success=info['failure_type'] == 'goal_success')
                break
        if not result['complete']:
            raise RuntimeError('固定策略诊断未按原1000步任务时域实际终止。')
        unchanged = all(torch.equal(before[name], tensor)
                        for name, tensor in actor.state_dict().items())
        if not unchanged or any(parameter.grad is not None for parameter in actor.parameters()):
            raise RuntimeError('固定策略诊断改变了Actor参数或产生梯度。')
        result.update(
            physical_time_s=duration, reward=reward_sum, reward_components=reward_parts,
            path_length_m=path, path_length_definition='CONTROL_NODE_POLYLINE',
            action_saturation_count=saturation, minimum_clearance_m=minimum_clearance,
            final_goal_distance_m=float(np.linalg.norm(
                case.goal()-env.world.auv_state.position_ned_m)),
            operation_limits_satisfied=bool(legal), actor_parameters_unchanged=unchanged,
            actor_gradient_operations=0,
            scope='Only this registered fixed case; not train-v1 reachability or a new baseline',
        )
    except BaseException as error:
        result.update(status='ERROR', error=repr(error))
        _write_json(summary_path, result)
        raise
    finally:
        if rows:
            with (output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with (output / 'obstacles.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = ['case_id', 'control_step', 'timestamp_s', 'obstacle_id',
                      'position_ned_m', 'velocity_ned_mps', 'radius_m']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(obstacles)
    _write_json(summary_path, result)
    return result


def run_learned_diagnostics(root: str | Path, project: Any,
                            code_version: str) -> dict[str, Any]:
    """可信本批100k/300k模型各测对应三例；推断-only、已登记尝试不自动重复。"""
    import torch
    from scripts.run_b0_reachability import fixed_cases

    from auv_risk_rl.rl.networks import Actor

    root = Path(root)
    state = json.loads((root / 'batch_state.json').read_text(encoding='utf-8'))
    if (state.get('registration_id') != REGISTRATION_ID
            or state.get('experiment_code_commit') != code_version):
        raise ValueError('学习策略诊断必须对应本批实际训练提交。')
    destination = root / 'learned_fixed_diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed in SEEDS:
        for transition in (100000, 300000):
            source = root / 'models' / f'seed_{seed}_{transition}.pt'
            if not source.exists():
                raise FileNotFoundError(f'最终固定策略诊断缺少实际模型: {source}')
            # 此轻量格式仅含本机可信primitive/tensor；不扩大反序列化允许对象集合。
            payload = torch.load(source, map_location='cpu', weights_only=True)
            device = validate_actor_model(payload, seed, transition, code_version)
            if not torch.cuda.is_available():
                raise RuntimeError('固定策略诊断目标CUDA不可用；未静默CPU回退。')
            with torch.random.fork_rng(devices=[]):
                actor = Actor().to(device=device, dtype=torch.float32)
            actor.load_state_dict(payload['actor'], strict=True)
            actor.eval().requires_grad_(False)
            if any(not torch.isfinite(parameter).all() for parameter in actor.parameters()):
                raise FloatingPointError('固定策略诊断模型参数非有限。')
            metadata = dict(training_seed=seed, model_transition=transition,
                            code_version=code_version, method=METHOD,
                            registration_id=REGISTRATION_ID, device=device,
                            model_file=source.relative_to(root).as_posix())
            selected = fixed_cases()[:3] if transition == 100000 else fixed_cases()[3:]
            for case in selected:
                cases.append(_learned_case(case, project, actor, device,
                                           destination / f'seed_{seed}_{transition}', metadata))
            torch.cuda.synchronize()
    result = dict(
        registration_id=REGISTRATION_ID, code_version=code_version,
        status='DIAGNOSTICS_COMPLETE', cases=cases,
        diagnostic_env_transitions=sum(case['diagnostic_env_transitions'] for case in cases),
        diagnostic_warmup_control_transitions=sum(
            case['diagnostic_warmup_control_transitions'] for case in cases),
        diagnostic_sac_updates=0, training_replay_writes=0, actor_gradient_operations=0,
        claim='Fixed-case learned policy diagnostic only; no all-scenario reachability claim',
    )
    _write_json(destination / 'learned_fixed_diagnostics.json', result)
    return result
