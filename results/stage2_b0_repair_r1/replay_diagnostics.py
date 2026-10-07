"""R1登记范围内的冻结策略重放；外部观察积分提议，不改科学转移或写Replay。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import load_project_config
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.local_task import command_to_normalized
from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.networks import Actor
from auv_risk_rl.training.fixed_validation import (
    FixedValidationPool,
    paired_validation_sensor_streams,
)
from auv_risk_rl.training.mvp_analysis import confirmed_records, validate_actor_model

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

V1_COMMIT = '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898'
TASK = ROOT / 'results' / 'stage2_b0_repair_r1'
V1 = ROOT / 'results' / 'stage2_b0_mvp_v1'
TOLERANCE = 1e-8


def clean(value: Any) -> Any:
    """只将未定义无障碍间距写null；不将非有限网络值当合法数值。"""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any, *, exclusive: bool = False) -> None:
    """单份小证据；不覆盖既有attempt或原V1。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(clean(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class BudgetLedger:
    """共享硬上限；先预留整条轨迹容量，异常保留预留量而不重复启动。"""

    def __init__(self, path: Path = TASK / 'diag_budget.json') -> None:
        self.path = path
        self.lock = path.with_suffix('.lock')
        path.parent.mkdir(parents=True, exist_ok=True)

    def _update(self, operation: Any) -> Any:
        for _ in range(1000):
            try:
                descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                break
            except FileExistsError:
                time.sleep(0.01)
        else:
            raise RuntimeError('共享诊断ledger锁未释放；不继续运行。')
        try:
            data = (json.loads(self.path.read_text(encoding='utf-8'))
                    if self.path.exists() else dict(
                registration_id='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
                trajectory_limit=200, control_transition_limit=200000, attempts=[],
                scientific_training_steps=0, scientific_training_updates=0,
                training_replay_writes=0))
            result = operation(data)
            temporary = self.path.with_suffix('.partial')
            write_json(temporary, data)
            os.replace(temporary, self.path)
            return result
        finally:
            self.lock.unlink()

    def reserve(self, trajectory_id: str, maximum_transitions: int) -> None:
        """预留含失败尝试的最坏剩余任务容量，不以结果决定重试。"""
        def apply(data: dict[str, Any]) -> None:
            if any(row['trajectory_id'] == trajectory_id for row in data['attempts']):
                raise RuntimeError(f'已有诊断attempt，不自动重复: {trajectory_id}')
            reserved = sum(row['reserved_transitions'] if row['status'] == 'RUNNING'
                           else row['actual_transitions'] for row in data['attempts'])
            if (len(data['attempts']) >= data['trajectory_limit']
                    or reserved + maximum_transitions > data['control_transition_limit']):
                raise RuntimeError('诊断预算硬上限触发；不再运行。')
            data['attempts'].append(dict(trajectory_id=trajectory_id, status='RUNNING',
                                         reserved_transitions=maximum_transitions,
                                         actual_transitions=0, warmup_control_transitions=0,
                                         created_at=datetime.now(UTC).isoformat()))
        self._update(apply)

    def finish(self, trajectory_id: str, actual: int, warmup: int,
               status: str = 'COMPLETED') -> None:
        """完整运行后释放未用预留；ERROR也真实保存已调用env.step数量。"""
        def apply(data: dict[str, Any]) -> None:
            row = next(item for item in data['attempts'] if item['trajectory_id'] == trajectory_id)
            if actual > row['reserved_transitions'] or row['status'] != 'RUNNING':
                raise ValueError('诊断计数超过预留或attempt重复结束。')
            row.update(status=status, actual_transitions=actual,
                       warmup_control_transitions=warmup,
                       finished_at=datetime.now(UTC).isoformat())
        self._update(apply)


def state_vector(state: Any) -> list[float]:
    """Eq5的8D状态顺序[N,E,D,yaw,pitch,surge,yaw-rate,pitch-rate]。"""
    return [*map(float, state.position_ned_m), float(state.yaw_rad), float(state.pitch_rad),
            float(state.surge_speed_mps), float(state.yaw_rate_rad_s),
            float(state.pitch_rate_rad_s)]


def independent_rotation(yaw: float, pitch: float) -> np.ndarray:
    """直接三角式Body→NED，不调用生产frames函数。"""
    sy, cy, sp, cp = math.sin(yaw), math.cos(yaw), math.sin(pitch), math.cos(pitch)
    return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0., cp]])


def fixed_cases() -> Any:
    """复用原固定案例，目录定位仅用于直接CLI执行。"""
    from scripts.run_b0_reachability import fixed_cases as existing_cases
    return existing_cases()


def los_command(state: Any, goal: np.ndarray, config: Any) -> Any:
    """复用已冻结非学习LOS，不开发新控制器。"""
    from scripts.run_b0_reachability import los_command as existing_los
    return existing_los(state, goal, config)


def scalar_cross(start: float, end: float, low: float, high: float) -> tuple[float, str] | None:
    """独立闭区间首次离开：合法终点等于边界不违规；非法提议定位到边界。"""
    if start < low or start > high:
        return 0., 'lower' if start < low else 'upper'
    if end < low:
        return float((low-start)/(end-start)), 'lower'
    if end > high:
        return float((high-start)/(end-start)), 'upper'
    return None


def boundary_candidates(start: Any, end: Any, project: Any) -> list[dict[str, Any]]:
    """独立计算各空间球包络和标量约束的首次越界分数。"""
    low = np.asarray(project.environment.position_lower_bound_ned_m) + project.risk.auv_radius_m
    high = np.asarray(project.environment.position_upper_bound_ned_m) - project.risk.auv_radius_m
    checks = [(f'position_{axis}', float(start.position_ned_m[index]),
               float(end.position_ned_m[index]), float(low[index]), float(high[index]))
              for index, axis in enumerate(('N', 'E', 'D'))]
    dynamics = project.dynamics
    checks.extend([
        ('pitch', start.pitch_rad, end.pitch_rad, -dynamics.max_pitch_rad,
         dynamics.max_pitch_rad),
        ('surge', start.surge_speed_mps, end.surge_speed_mps, dynamics.min_surge_speed_mps,
         dynamics.max_surge_speed_mps),
        ('yaw_rate', start.yaw_rate_rad_s, end.yaw_rate_rad_s, -dynamics.max_yaw_rate_rad_s,
         dynamics.max_yaw_rate_rad_s),
        ('pitch_rate', start.pitch_rate_rad_s, end.pitch_rate_rad_s,
         -dynamics.max_pitch_rate_rad_s, dynamics.max_pitch_rate_rad_s),
    ])
    result = []
    for name, first, last, lower, upper in checks:
        crossing = scalar_cross(first, last, lower, upper)
        if crossing is not None:
            result.append(dict(constraint=f'{name}_{crossing[1]}', fraction=crossing[0],
                               start=first, proposed_end=last, lower=lower, upper=upper))
    return result


def sphere_entry(start: np.ndarray, end: np.ndarray, center: np.ndarray,
                 radius: float) -> float | None:
    """独立二次根，参考只使用分段线性位置；不调用生产几何。"""
    displacement = np.asarray(start, dtype=np.float64)-np.asarray(center, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64)-np.asarray(start, dtype=np.float64)
    c = float(displacement @ displacement)-radius*radius
    if c <= 0:
        return 0.
    a = float(delta @ delta)
    if a == 0:
        return None
    b = 2*float(displacement @ delta)
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return None
    roots = ((-b-math.sqrt(discriminant))/(2*a), (-b+math.sqrt(discriminant))/(2*a))
    valid = [value for value in roots if 0 <= value <= 1]
    return min(valid) if valid else None


def nearest_goal(start: np.ndarray, end: np.ndarray,
                 center: np.ndarray) -> tuple[float, float]:
    """独立投影到实际执行线段，返回中心距离及最小点分数。"""
    offset, delta = start-center, end-start
    denominator = float(delta @ delta)
    fraction = max(0., min(1., -float(offset @ delta)/denominator)) if denominator else 0.
    return float(np.linalg.norm(offset+fraction*delta)), fraction


def input_audit(env: B0NavigationEnv, observation: np.ndarray,
                action: np.ndarray) -> dict[str, Any]:
    """真实失败快照的独立Body/NED、234D首18字段及命令映射核对。"""
    state, project = env.world.auv_state, env.config
    rotation = independent_rotation(state.yaw_rad, state.pitch_rad).T
    goal_delta = env.goal_position_ned_m-state.position_ned_m
    dynamics = project.dynamics
    low = np.asarray(project.environment.position_lower_bound_ned_m)
    high = np.asarray(project.environment.position_upper_bound_ned_m)
    expected = np.array([
        *(rotation @ goal_delta/100.), *(2*(state.position_ned_m-low)/(high-low)-1),
        2*(state.surge_speed_mps-dynamics.min_surge_speed_mps)
        /(dynamics.max_surge_speed_mps-dynamics.min_surge_speed_mps)-1,
        state.yaw_rate_rad_s/dynamics.max_yaw_rate_rad_s,
        state.pitch_rate_rad_s/dynamics.max_pitch_rate_rad_s,
        math.sin(state.yaw_rad), math.cos(state.yaw_rad),
        math.sin(state.pitch_rad), math.cos(state.pitch_rad),
        *env.previous_normalized,
        1-env.world.control_step_index/project.environment.max_episode_control_steps,
        project.risk.short_horizon_risk_budget/.2,
    ], dtype=np.float32)
    errors = np.abs(expected-observation[:18])
    command = np.array([
        dynamics.min_surge_speed_mps+(dynamics.max_surge_speed_mps-dynamics.min_surge_speed_mps)
        *(float(action[0])+1)/2,
        float(action[1])*dynamics.max_yaw_rate_rad_s,
        float(action[2])*dynamics.max_pitch_rate_rad_s,
    ])
    ordered = sorted(env.world.obstacle_states, key=lambda obstacle: (
        float(np.linalg.norm(obstacle.position_ned_m-state.position_ned_m)), obstacle.obstacle_id))
    slot_error = 0.
    for index, obstacle in enumerate(ordered[:6]):
        expected_slot = np.array([
            *(rotation @ (obstacle.position_ned_m-state.position_ned_m)/project.sensor.range_m),
            *(rotation @ obstacle.velocity_ned_mps), *([0.]*12), obstacle.radius_m, 0., 1.],
            dtype=np.float32)
        slot_error = max(slot_error, float(np.max(np.abs(
            expected_slot-observation[108+21*index:129+21*index]))))
    return dict(goal_body_m=(rotation @ goal_delta).tolist(),
                goal_distance_m=float(np.linalg.norm(goal_delta)),
                self_task_observation_max_abs_error=float(np.max(errors)),
                truth_slot_max_abs_error=slot_error,
                expected_physical_command=command.tolist(), expected_self_task=expected.tolist())


class ProposalObserver:
    """只观察当前实例的小步提议；finally恢复实例，返回值完全不变。"""

    def __init__(self, env: B0NavigationEnv) -> None:
        self.env = env
        self.rows: list[dict[str, Any]] = []
        self.original: Any = None

    def __enter__(self) -> ProposalObserver:
        world = self.env.world
        self.original = world._propose_integration_step

        def observe(start: Any, obstacles: Any, command: Any, timestamp: float) -> Any:
            proposal = self.original(start, obstacles, command, timestamp)
            end, candidates = proposal.auv_end_state, boundary_candidates(
                start, proposal.auv_end_state, self.env.config)
            goal = self.env.goal_position_ned_m
            root = sphere_entry(start.position_ned_m, end.position_ned_m,
                                goal, self.env.config.environment.goal_radius_m)
            collision_roots = []
            for initial, final in zip(obstacles, proposal.obstacle_end_states, strict=True):
                collision = sphere_entry(start.position_ned_m-initial.position_ned_m,
                                         end.position_ned_m-final.position_ned_m,
                                         np.zeros(3), self.env.config.risk.auv_radius_m
                                         +initial.radius_m)
                if collision is not None:
                    collision_roots.append(dict(fraction=collision,
                                                 obstacle_id=initial.obstacle_id))
            independent = [(item['fraction'], 0, 'collision') for item in collision_roots]
            independent += [(item['fraction'], 1, 'boundary') for item in candidates]
            if root is not None:
                independent.append((root, 2, 'success'))
            expected = min(independent) if independent else None
            event = proposal.event
            fraction = 1. if event is None else event.fraction
            actual_end = start.position_ned_m+fraction*(end.position_ned_m-start.position_ned_m)
            distance, closest = nearest_goal(start.position_ned_m, actual_end, goal)
            actual_root = (root/fraction if root is not None and fraction > 0
                           and root <= fraction+TOLERANCE else None)
            event_record = (None if event is None else dict(
                reason=event.event.reason, fraction=event.fraction,
                timestamp_s=event.event.event_timestamp_s, priority=event.priority))
            agreement = (expected is None and event is None) or (
                expected is not None and event is not None
                and expected[2] == event.event.reason
                and abs(expected[0]-event.fraction) <= TOLERANCE)
            self.rows.append(dict(
                timestamp_s=timestamp, start_state=state_vector(start),
                uncommitted_proposed_state=state_vector(end),
                executed_fraction=fraction, independent_boundary_candidates=candidates,
                independent_collision_candidates=collision_roots,
                independent_goal_entry_proposed_fraction=root,
                independent_goal_entry_executed_fraction=actual_root,
                independent_expected_event=expected, production_event=event_record,
                independent_event_agreement=agreement,
                closest_goal_distance_m=distance,
                closest_goal_timestamp_s=timestamp+fraction*closest
                *self.env.config.dynamics.integration_dt_s,
                closest_goal_state=(np.asarray(state_vector(start))+fraction*closest
                                    *(np.asarray(state_vector(end))-state_vector(start))).tolist()))
            return proposal

        world._propose_integration_step = observe
        return self

    def __exit__(self, *args: Any) -> None:
        del self.env.world.__dict__['_propose_integration_step']


def load_actor(seed: int, transition: int) -> tuple[Actor, str]:
    """仅可信V1本机轻量Actor，先核对原科研身份；CUDA不静默回退。"""
    payload = torch.load(V1/'models'/f'seed_{seed}_{transition}.pt',
                         map_location='cpu', weights_only=True)
    device = validate_actor_model(payload, seed, transition, V1_COMMIT)
    if not torch.cuda.is_available():
        raise RuntimeError('冻结策略诊断CUDA不可用。')
    with torch.random.fork_rng(devices=[]):
        actor = Actor().to(device=device, dtype=torch.float32)
    actor.load_state_dict(payload['actor'], strict=True)
    actor.eval().requires_grad_(False)
    return actor, device


def policy_values(actor: Actor, observation: np.ndarray, device: str,
                  ) -> tuple[np.ndarray, dict[str, Any]]:
    """deterministic均值动作且保留原mean/log_std；不采样训练流。"""
    with torch.inference_mode():
        tensor = torch.as_tensor(observation, device=device, dtype=torch.float32)
        mean, log_std = actor(tensor)
        action = mean.tanh().cpu().numpy().copy()
    if not np.all(np.isfinite(action)):
        raise FloatingPointError('冻结策略动作非有限。')
    return action, dict(mean=mean.cpu().tolist(), log_std=log_std.cpu().tolist())


def selected_snapshot(env: B0NavigationEnv) -> str | None:
    """只按登记阈值选择快照，不按接管分支结果改位置。"""
    if abs(env.world.auv_state.pitch_rad) >= math.radians(20.):
        return 'FIRST_ABS_PITCH_GE_20_DEG'
    if np.linalg.norm(env.goal_position_ned_m-env.world.auv_state.position_ned_m) <= 10.:
        return 'FIRST_GOAL_DISTANCE_LE_10_M'
    return None


def action_branch(original: np.ndarray, env: B0NavigationEnv, branch: str) -> np.ndarray:
    """预登记五分支只替换既有LOS分量，不搜索动作、不产生训练示范。"""
    if branch == 'original':
        return original.copy()
    los = command_to_normalized(los_command(env.world.auv_state, env.goal_position_ned_m,
                                           env.config), env.config.dynamics)
    if branch == 'los':
        return los
    action = original.copy()
    action[{'surge_only': 0, 'yaw_only': 1, 'pitch_only': 2}[branch]] = los[
        {'surge_only': 0, 'yaw_only': 1, 'pitch_only': 2}[branch]]
    return action


def run_trajectory(metadata: dict[str, Any], env: B0NavigationEnv, actor: Actor, device: str,
                   *, observation: np.ndarray | None = None, branch: str = 'original',
                   retain_snapshot: bool = False, paired: bool = False,
                   ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """一次冻结轨迹；每步记录真实调用，失败也保留attempt与已执行证据。"""
    identity = metadata['trajectory_id']
    destination = TASK/'diagnostic_replay'/identity
    if destination.exists():
        raise FileExistsError(f'该诊断已有证据，不自动重复: {identity}')
    destination.mkdir(parents=True)
    maximum = 1000 if observation is None else 1000-env.world.control_step_index
    ledger = BudgetLedger()
    ledger.reserve(identity, maximum)
    write_json(destination/'attempt.json', dict(metadata, branch=branch,
                                               maximum_transitions=maximum,
                                               created_at=datetime.now(UTC).isoformat()),
               exclusive=True)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    count = warmup = 0
    summary = dict(metadata, branch=branch, status='RUNNING', transitions=0,
                   science_training_steps=0, science_training_updates=0, training_replay_writes=0,
                   actor_gradient_operations=0, warmup_control_transitions=0,
                   closest_approach=None, first_goal_entry=None, event_mismatches=[],
                   independent_observation_max_abs_error=0., independent_command_max_abs_error=0.,
                   reward_components={}, reward_component_max_abs_error=0.,
                   reward=0., discounted_reward=0., discounted_progress=0.,
                   executed_segment_count=0, goal_success_rewards=0)
    snapshot = fallback = last_legal = None
    rows: list[dict[str, Any]] = []
    context = (paired_validation_sensor_streams(metadata['environment_seed'])
               if paired else nullcontext())
    try:
        with context, gzip.open(destination/'full_trace.jsonl.gz', 'wt', encoding='utf-8') as trace:
            if observation is None:
                observation, reset_info = env.reset(seed=metadata['environment_seed'],
                                                    options={'external_max_steps': None})
                warmup = round(reset_info['warmup_duration_s']/env.config.dynamics.control_dt_s)
                summary.update(warmup=reset_info,
                               formal_initial_state=state_vector(env.world.auv_state),
                               goal_position_ned_m=env.goal_position_ned_m.tolist())
            else:
                summary.update(snapshot_initial_state=state_vector(env.world.auv_state),
                               snapshot_task_step=env.world.control_step_index,
                               goal_position_ned_m=env.goal_position_ned_m.tolist())
            initial_distance = float(np.linalg.norm(
                env.goal_position_ned_m-env.world.auv_state.position_ned_m))
            summary['initial_goal_distance_m'] = initial_distance
            first_reward_step = env.world.control_step_index
            while not env._done:
                if count >= maximum:
                    raise RuntimeError('真实任务未在冻结剩余时域终止，不能追加。')
                # observer只在step内部存在；深拷贝包含干净world和实际感知/RNG历史。
                if retain_snapshot:
                    reason = selected_snapshot(env)
                    current = None
                    if ((snapshot is None and reason is not None)
                            or env.world.control_step_index <= 50):
                        current = dict(env=deepcopy(env), observation=observation.copy(),
                                       task_step=env.world.control_step_index)
                    if env.world.control_step_index < 50:
                        last_legal = current
                    if snapshot is None and reason is not None:
                        if current is None:
                            current = dict(env=deepcopy(env), observation=observation.copy(),
                                           task_step=env.world.control_step_index)
                        snapshot = dict(current, selection_reason=reason)
                    if fallback is None and env.world.control_step_index == 50:
                        fallback = dict(current, selection_reason='CONTROL_STEP_50_FALLBACK')
                action, distribution = policy_values(actor, observation, device)
                audit = input_audit(env, observation, action)
                chosen = action_branch(action, env, branch)
                chosen_audit = input_audit(env, observation, chosen)
                state_before = state_vector(env.world.auv_state)
                previous = env.previous_normalized.copy()
                obs_before = observation.copy()
                timestamp = env.world.timestamp_s
                # 每次实际调用env.step计一次；异常尝试不被隐藏。
                count += 1
                with ProposalObserver(env) as observer:
                    observation, reward, terminated, truncated, info = env.step(chosen)
                actual_command = np.array([
                    env.previous_command.surge_speed_command_mps,
                    env.previous_command.yaw_rate_command_rad_s,
                    env.previous_command.pitch_rate_command_rad_s])
                error = float(np.max(np.abs(
                    actual_command-chosen_audit['expected_physical_command'])))
                summary['independent_command_max_abs_error'] = max(
                    summary['independent_command_max_abs_error'], error)
                summary['independent_observation_max_abs_error'] = max(
                    summary['independent_observation_max_abs_error'],
                    audit['self_task_observation_max_abs_error'], audit['truth_slot_max_abs_error'])
                distance_after = float(np.linalg.norm(
                    env.goal_position_ned_m-env.world.auv_state.position_ned_m))
                reference_parts = dict(
                    progress=audit['goal_distance_m']-distance_after,
                    goal=100.*int(info['failure_type'] == 'goal_success'),
                    time=-.01*(env.world.timestamp_s-timestamp)/.2,
                    smoothness=-.02*sum((float(last)-float(first))**2
                                       for first, last in zip(previous, chosen, strict=True)))
                reward_error = max(abs(reference_parts[key]-info['reward_components'][key])
                                   for key in reference_parts)
                reward_error = max(reward_error, abs(sum(reference_parts.values())-reward))
                summary['reward_component_max_abs_error'] = max(
                    summary['reward_component_max_abs_error'], reward_error)
                summary['reward'] += reward
                discount = .999**(env.world.control_step_index-first_reward_step-1)
                summary['discounted_reward'] += discount*reward
                summary['discounted_progress'] += discount*reference_parts['progress']
                summary['goal_success_rewards'] += int(reference_parts['goal'] != 0)
                for key, value in info['reward_components'].items():
                    summary['reward_components'][key] = (
                        summary['reward_components'].get(key, 0.)+value)
                for segment in observer.rows:
                    summary['executed_segment_count'] += 1
                    if not segment['independent_event_agreement']:
                        summary['event_mismatches'].append(segment)
                    if (summary['closest_approach'] is None
                            or segment['closest_goal_distance_m']
                            < summary['closest_approach']['distance_m']):
                        summary['closest_approach'] = dict(
                            distance_m=segment['closest_goal_distance_m'],
                            timestamp_s=segment['closest_goal_timestamp_s'],
                            state=segment['closest_goal_state'],
                            actor_action=action.tolist(), executed_action=chosen.tolist(),
                            physical_command=actual_command.tolist(),
                            goal_body_m=audit['goal_body_m'], distribution=distribution,
                            task_step=env.world.control_step_index)
                    crossing = segment['independent_goal_entry_executed_fraction']
                    if crossing is not None and summary['first_goal_entry'] is None:
                        summary['first_goal_entry'] = dict(
                            timestamp_s=segment['timestamp_s']+crossing*segment['executed_fraction']
                            *env.config.dynamics.integration_dt_s,
                            task_step=env.world.control_step_index, segment=segment)
                detail = dict(task_step=env.world.control_step_index, timestamp_before_s=timestamp,
                              timestamp_after_s=env.world.timestamp_s, state_before=state_before,
                              state_after=state_vector(env.world.auv_state), observation=obs_before,
                              next_observation=observation, actor=distribution,
                              nominal_actor_action=action, executed_branch_action=chosen,
                              previous_action=previous, physical_command=actual_command,
                              input_audit=audit, reward=reward,
                              reward_components=info['reward_components'],
                              reward_reference=reference_parts,
                              terminated=terminated, truncated=truncated,
                              failure_type=info['failure_type'], integration_segments=observer.rows)
                trace.write(json.dumps(clean(detail), ensure_ascii=False, allow_nan=False)+'\n')
                rows.append(dict(task_step=env.world.control_step_index,
                                 timestamp_s=env.world.timestamp_s,
                                 **{key: value for key, value in zip(
                                     ('north_m', 'east_m', 'down_m', 'yaw_rad', 'pitch_rad',
                                      'surge_mps', 'yaw_rate_rad_s', 'pitch_rate_rad_s'),
                                     state_vector(env.world.auv_state), strict=True)},
                                 action_surge=float(chosen[0]), action_yaw=float(chosen[1]),
                                 action_pitch=float(chosen[2]), goal_distance_m=distance_after,
                                 reward=reward, failure_type=info['failure_type']))
                if terminated or truncated:
                    summary.update(final_event=detail, failure_type=info['failure_type'],
                                   final_state=state_vector(env.world.auv_state),
                                   final_goal_distance_m=distance_after,
                                   complete=terminated and not truncated)
                    if info['failure_type'] == 'operational_boundary_failure':
                        final = observer.rows[-1]
                        fraction = final['production_event']['fraction']
                        tied = [item['constraint']
                                for item in final['independent_boundary_candidates']
                                if abs(item['fraction']-fraction) <= TOLERANCE]
                        summary['boundary_subtypes'] = tied or ['UNRESOLVED']
                    else:
                        summary['boundary_subtypes'] = []
                    break
            unchanged = all(torch.equal(before[name], value)
                            for name, value in actor.state_dict().items())
            if not unchanged or any(parameter.grad is not None for parameter in actor.parameters()):
                raise RuntimeError('冻结重放改变模型或产生梯度。')
            summary.update(status='COMPLETED', actor_parameters_unchanged=unchanged,
                           goal_event_consistent=(bool(summary['first_goal_entry'])
                                                  == (summary['failure_type'] == 'goal_success')),
                           undiscounted_progress_telescoping_error=abs(
                               summary['reward_components']['progress']
                               -(initial_distance-summary['final_goal_distance_m'])))
        if retain_snapshot:
            snapshot = (snapshot or fallback
                        or dict(last_legal, selection_reason='LAST_LEGAL_BEFORE_EVENT'))
            snapshot.update(source_trajectory_id=identity,
                            environment_seed=metadata['environment_seed'], paired=paired,
                            training_seed=metadata['training_seed'],
                            model_transition=metadata['model_transition'])
        ledger.finish(identity, count, warmup)
    except BaseException as error:
        summary.update(status='ERROR', error=repr(error))
        ledger.finish(identity, count, warmup, 'ERROR')
        raise
    finally:
        summary.update(transitions=count, warmup_control_transitions=warmup)
        write_json(destination/'summary.json', summary)
        if rows:
            with (destination/'trajectory.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return summary, snapshot


def v1_success_indices() -> dict[tuple[int, str], int]:
    """既有完整终点Val升序首个成功，不挑最佳seed、不重新抽场景。"""
    output = {}
    for seed in (11, 22, 33):
        for kind, row in confirmed_records(V1, seed, V1_COMMIT, {}):
            if kind != 'validation' or not row['full']:
                continue
            successes = sorted(ep['index'] for ep in row['episodes'] if ep['success'])
            if successes:
                output[seed, row['profile']] = successes[0]
    return output


def write_compact(summaries: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> None:
    """公开小证据与第一事件/近目标反例；完整观察gzip仅本机。"""
    compact = []
    for row in summaries:
        public = {key: value for key, value in row.items() if key != 'final_event'}
        event = row.get('final_event')
        if event is not None:
            public['first_event_chain'] = {key: event[key] for key in (
                'state_before', 'state_after', 'nominal_actor_action', 'executed_branch_action',
                'physical_command', 'input_audit', 'reward', 'reward_components',
                'reward_reference', 'failure_type', 'integration_segments', 'actor')}
        compact.append(public)
    counts = Counter()
    for row in compact:
        for subtype in row.get('boundary_subtypes', []):
            counts[subtype] += 1
    result = dict(registration_id='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
                  status='FROZEN_REPLAY_COMPLETED', cases=compact,
                  trajectory_count=len(compact), boundary_subtype_counts=counts,
                  diagnostic_control_transitions=sum(row['transitions'] for row in compact),
                  diagnostic_warmup_control_transitions=sum(row['warmup_control_transitions']
                                                           for row in compact),
                  event_mismatch_count=sum(len(row['event_mismatches']) for row in compact),
                  missed_goal_count=sum(not row.get('goal_event_consistent', False)
                                        for row in compact),
                  scientific_training_steps=0, scientific_training_updates=0,
                  training_replay_writes=0, actor_gradient_operations=0,
                  claim='Frozen case mechanism diagnostics only; not an independent Test result.')
    write_json(TASK/'frozen_replay_result.json', result)
    snapshot_rows = []
    for row in snapshots:
        env = row['env']
        actor, device = load_actor(row['training_seed'], row['model_transition'])
        original, distribution = policy_values(actor, row['observation'], device)
        snapshot_rows.append(dict(
            source_trajectory_id=row['source_trajectory_id'], task_step=row['task_step'],
            selection_reason=row['selection_reason'], environment_seed=row['environment_seed'],
            training_seed=row['training_seed'], model_transition=row['model_transition'],
            state=state_vector(env.world.auv_state), observation=row['observation'].tolist(),
            actor=distribution, actor_action=original.tolist(),
            independent_input_audit=input_audit(env, row['observation'], original),
            goal_position_ned_m=env.goal_position_ned_m.tolist(),
            paired=row['paired'],
            local_snapshot=f'local_snapshots/{row["source_trajectory_id"]}.pt'))
    write_json(TASK/'frozen_snapshots.json', dict(snapshots=snapshot_rows))


def main() -> int:
    """显式已冻结登记下顺序执行63core、<=6成功对照、30反事实；不训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    registration = ROOT/'docs'/'STAGE2_B0_REPAIR_R1.md'
    if not args.execute or not registration.is_file():
        parser.print_help()
        return 0
    project = load_project_config(ROOT/'configs'/'stage0.yaml')
    scenario = load_training_scenario_config(ROOT/'configs'/'train_scenario_v1.yaml')
    pool = FixedValidationPool(project, scenario)
    summaries, snapshots = [], []
    success_indices = v1_success_indices()
    write_json(TASK/'replay_selection.json', dict(
        selected_success_controls=[dict(seed=seed, profile=profile, index=index)
                                   for (seed, profile), index in success_indices.items()],
        fixed_core=63, max_extra_success=6, max_counterfactual=30), exclusive=True)
    for seed in (11, 22, 33):
        for transition in (100000, 300000):
            actor, device = load_actor(seed, transition)
            selected = fixed_cases()[:3] if transition == 100000 else fixed_cases()
            for case in selected:
                kind = ('original_fixed' if transition == 100000 or case in fixed_cases()[3:]
                        else 'postcv_empty_fixed')
                metadata = dict(trajectory_id=f'{kind}_s{seed}_m{transition}_{case.case_id}',
                                selection=kind, training_seed=seed, model_transition=transition,
                                task_profile='obstacle_free' if case in fixed_cases()[:3]
                                else 'fixed_cv_diagnostic', environment_seed=case.seed,
                                scenario_id=case.case_id)
                env = B0NavigationEnv(project, case.initial_state(), case.obstacles(), case.goal(),
                                      f'learned-fixed-{case.case_id}')
                summary, _ = run_trajectory(metadata, env, actor, device)
                summaries.append(summary)
            for profile in ('obstacle_free', 'cv_train_v1'):
                snapshot_saved = False
                selected_indices = list(range(3))
                if transition == (100000 if profile == 'obstacle_free' else 300000):
                    success = success_indices.get((seed, profile))
                    if success is not None and success not in selected_indices:
                        selected_indices.append(success)
                for index in selected_indices:
                    item = pool.scenario(profile, index)
                    retain = (not snapshot_saved and index < 3
                              and transition == (100000 if profile == 'obstacle_free' else 300000))
                    metadata = dict(
                        trajectory_id=f'val_s{seed}_m{transition}_{profile}_i{index}',
                        selection='VAL_0_1_2' if index < 3 else 'FIRST_SUCCESS_BY_INDEX',
                        training_seed=seed, model_transition=transition, task_profile=profile,
                        environment_seed=pool.environment_seeds[index], scenario_index=index,
                        scenario_id=item.scenario_id,
                        base_scenario_id=pool.base_scenarios[index].scenario_id)
                    env = B0NavigationEnv(project, item.initial_auv_state,
                                          item.initial_obstacle_states,
                                          item.goal_position_ned_m, item.scenario_id)
                    summary, snapshot = run_trajectory(metadata, env, actor, device,
                                                       retain_snapshot=retain, paired=True)
                    summaries.append(summary)
                    if retain and summary['failure_type'] != 'goal_success':
                        snapshots.append(snapshot)
                        snapshot_saved = True
                        target = TASK/'local_snapshots'/f'{snapshot["source_trajectory_id"]}.pt'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(snapshot, target)
            print(f'Completed seed {seed} model {transition}: {len(summaries)} trajectories',
                  flush=True)
    for snapshot in snapshots:
        actor, device = load_actor(snapshot['training_seed'], snapshot['model_transition'])
        for branch in ('original', 'los', 'pitch_only', 'surge_only', 'yaw_only'):
            env, observation = deepcopy(snapshot['env']), snapshot['observation'].copy()
            metadata = dict(
                trajectory_id=f'cf_{snapshot["source_trajectory_id"]}_{branch}',
                source_trajectory_id=snapshot['source_trajectory_id'],
                selection='FIXED_COUNTERFACTUAL',
                training_seed=snapshot['training_seed'],
                model_transition=snapshot['model_transition'],
                environment_seed=snapshot['environment_seed'],
                task_profile=('obstacle_free' if snapshot['model_transition'] == 100000
                              else 'cv_train_v1'),
                snapshot_task_step=snapshot['task_step'],
                snapshot_selection=snapshot['selection_reason'])
            summary, _ = run_trajectory(metadata, env, actor, device,
                                        observation=observation, branch=branch,
                                        paired=snapshot['paired'])
            summaries.append(summary)
        print(f'Completed counterfactual {snapshot["source_trajectory_id"]}', flush=True)
    write_compact(summaries, snapshots)
    print(json.dumps(dict(status='FROZEN_REPLAY_COMPLETED', trajectories=len(summaries),
                          transitions=sum(row['transitions'] for row in summaries))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
