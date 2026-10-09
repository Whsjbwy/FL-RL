"""六个真实300k终点模型的18例固定诊断；批次和实际worker完成前拒绝运行。"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import ProjectConfig, load_project_config
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.rl.networks import Actor, RewardCritic
from auv_risk_rl.training.mvp_batch import atomic_json
from auv_risk_rl.training.r2_registration import APPROVED, REGISTRATION_ID, verify_code_identity
from auv_risk_rl.training.repair_r2 import (
    harness_config,
    new_r2_episode_diagnostics,
    r2_diagnostic_step,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/stage2_b0_r2'
DESTINATION = OUTPUT / 'final_diagnostics'
CODE_VERSION = '929f26af249cbd070dd4913bdf2eb316ae489d2d'
NETWORKS = ('actor', 'q1', 'q2', 'target_q1', 'target_q2')
CASE_IDS = ('R01_horizontal_empty', 'R02_deeper_empty', 'R03_shallower_empty')
FORMAT = 'b0-r2-final-fixed-diagnostics-v1'


def utc() -> str:
    """记录实际UTC时间，不以预期完成时刻代替。"""
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    """只读取本任务真实本机记录。"""
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'记录不是JSON对象：{path}')
    return value


def completion_problem(batch: dict[str, Any], worker: dict[str, Any]) -> str | None:
    """严格核对真实六run终点和实际计算进程结束码，不采用venv启动器状态。"""
    if (batch.get('registration_id') != REGISTRATION_ID
            or batch.get('registration') != APPROVED
            or batch.get('experiment_code_commit') != CODE_VERSION):
        return 'BATCH_IDENTITY_MISMATCH'
    if batch.get('status') != 'COMPLETED':
        return 'BATCH_NOT_COMPLETED'
    if (worker.get('status') != 'EXITED'
            or type(worker.get('actual_worker_exit_code')) is not int
            or worker['actual_worker_exit_code'] != 0
            or type(batch.get('pid')) is not int
            or worker.get('actual_computation_pid') != batch['pid']):
        return 'ACTUAL_COMPUTATION_WORKER_NOT_EXITED_ZERO'
    if batch.get('completed_runs') != APPROVED['run_order']:
        return 'SIX_COMPLETED_RUNS_NOT_CONFIRMED'
    for run_id in APPROVED['run_order']:
        group, seed_text = run_id.split('_seed')
        row = batch.get('runs', {}).get(run_id, {})
        if (row.get('status') != 'COMPLETED' or row.get('transitions') != 300000
                or row.get('updates') != 290001 or row.get('checkpoint_transition') != 300000
                or row.get('group') != group or row.get('seed') != int(seed_text)
                or row.get('reward_config') != APPROVED['groups'][group]):
            return f'FINAL_RUN_IDENTITY_OR_COUNTER_MISMATCH:{run_id}'
    if (batch.get('actual_training_transitions') != 1800000
            or batch.get('actual_sac_updates') != 1740006):
        return 'FINAL_BATCH_COUNTERS_MISMATCH'
    return None


def model_path(group: str, seed: int) -> Path:
    """路径固定到可信本机终点五网络snapshot，不接受外部模型路径。"""
    return OUTPUT / group / 'models' / f'seed_{seed}_300000.pt'


def execution_problem(batch: dict[str, Any], worker: dict[str, Any], *,
                      all_models_present: bool, attempt_exists: bool) -> str | None:
    """在任何模型加载或reset前统一拒绝未完成、缺模型和重复诊断尝试。"""
    problem = completion_problem(batch, worker)
    if problem is None and not all_models_present:
        problem = 'FINAL_MODEL_MISSING'
    if problem is None and attempt_exists:
        problem = 'DIAGNOSTIC_ATTEMPT_ALREADY_EXISTS_NO_RETRY'
    return problem


def preflight() -> dict[str, Any]:
    """只检查状态、路径及Git版本，不创建Actor、环境、Replay或执行前向。"""
    batch = read_json(OUTPUT / 'batch_state.json')
    worker = read_json(OUTPUT / 'actual_training_worker_exit.json')
    inventory = [dict(group=group, seed=seed,
                      path=str(model_path(group, seed).relative_to(ROOT)),
                      exists=model_path(group, seed).is_file())
                 for group in APPROVED['groups'] for seed in APPROVED['training_seeds']]
    problem = execution_problem(
        batch, worker, all_models_present=all(row['exists'] for row in inventory),
        attempt_exists=DESTINATION.exists() and any(DESTINATION.iterdir()))
    verify_code_identity(ROOT, CODE_VERSION)
    return dict(format=FORMAT, checked_at_utc=utc(), ready=problem is None,
                refusal_reason=problem, batch_status=batch.get('status'),
                worker_status=worker.get('status'), actual_worker_exit=worker,
                code_version=CODE_VERSION, model_inventory=inventory,
                case_ids=list(CASE_IDS), planned_cases=18,
                actual_diagnostic_env_transitions=0, actor_forward_calls=0,
                scientific_training_steps=0, scientific_training_updates=0)


def load_cases() -> tuple[tuple[Any, ...], Callable[..., bool]]:
    """仅复用已有固定案例定义，不调用其LOS控制器或可达性运行函数。"""
    path = ROOT / 'scripts/run_b0_reachability.py'
    name = '_r2_existing_fixed_reachability_cases'
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError('已有固定案例loader无法读取。')
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    cases = module.fixed_cases()[:3]
    if tuple(case.case_id for case in cases) != CASE_IDS or any(case.obstacles() for case in cases):
        raise ValueError('已有R01/R02/R03固定无障碍案例身份发生变化。')
    return cases, module.state_is_legal


def load_snapshot(group: str, seed: int) -> dict[str, Any]:
    """仅weights_only读取可信本机snapshot并核对五网络、终点、配置、有限性。"""
    value = torch.load(model_path(group, seed), map_location='cpu', weights_only=True)
    expected = harness_config(APPROVED, group, seed)
    identity = dict(format='b0-r2-network-snapshot-v1', seed=seed, transition=300000,
                    code_version=CODE_VERSION, registration_id=REGISTRATION_ID, group=group,
                    run_kind='scientific_training', method=expected.method,
                    sac_config=asdict(expected.sac), harness_config=expected.to_dict())
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in identity.items()):
        raise ValueError(f'终点snapshot身份或reward配置不符：{group}/seed{seed}')
    models = value.get('models')
    if not isinstance(models, dict) or set(models) != set(NETWORKS):
        raise ValueError('终点snapshot必须包含完整五网络。')
    for name, state in models.items():
        if (not isinstance(state, dict) or not state
                or any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                       or not torch.isfinite(tensor).all() for tensor in state.values())):
            raise ValueError(f'snapshot网络格式/float32/有限性无效：{name}')
    alpha = value.get('log_alpha')
    if not isinstance(alpha, torch.Tensor) or alpha.numel() != 1 or not torch.isfinite(alpha).all():
        raise ValueError('snapshot alpha必须为有限标量。')
    counters = value.get('counters', {})
    if (counters.get('environment_steps') != 300000
            or any(counters.get(key) != 290001 for key in ('gradient_updates', 'actor', 'q1',
                                                         'q2', 'alpha'))):
        raise ValueError('终点snapshot真实环境/更新计数不符。')
    if (set(value.get('actor', {})) != set(models['actor'])
            or any(not torch.equal(value['actor'][key], tensor)
                   for key, tensor in models['actor'].items())):
        raise ValueError('snapshot兼容Actor字段与五网络Actor不一致。')
    return value


def build_networks(value: dict[str, Any]) -> dict[str, torch.nn.Module]:
    """只构造推理网络，无Agent、Replay、优化器、采样器或训练更新。"""
    with torch.random.fork_rng(devices=[]):
        modules = dict(actor=Actor(), q1=RewardCritic(), q2=RewardCritic(),
                       target_q1=RewardCritic(), target_q2=RewardCritic())
    for name, module in modules.items():
        module.load_state_dict(value['models'][name], strict=True)
        module.eval().requires_grad_(False)
    # target仅验证权重兼容；实际诊断Q来自online Q1/Q2，绝不混作Bellman真值。
    return {name: modules[name].to(APPROVED['device']) for name in ('actor', 'q1', 'q2')}


@contextmanager
def count_world_steps(counters: dict[str, int], key: str) -> Any:
    """数实际成功world.step；即使后续观察失败也保留已执行步数，finally恢复。"""
    original = AUVWorld.step

    def observed(world: AUVWorld, *args: Any, **kwargs: Any) -> Any:
        """原step执行一次之后记账；不增添任何仿真步。"""
        result = original(world, *args, **kwargs)
        counters[key] += 1
        return result

    AUVWorld.step = observed
    try:
        yield
    finally:
        AUVWorld.step = original


def inference(networks: dict[str, torch.nn.Module], observation: np.ndarray,
              counters: dict[str, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """一次Actor前向取tanh均值动作；online两个Q各一次只读前向并分项记账。"""
    if observation.shape != (234,) or not np.all(np.isfinite(observation)):
        raise FloatingPointError('诊断观察必须是有限234D。')
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=APPROVED['device'])
    with torch.no_grad():
        counters['actor_forward_calls'] += 1
        mean, log_std = networks['actor'](tensor)
        action = mean.tanh()
        counters['q1_forward_calls'] += 1
        q1 = networks['q1'](tensor, action)
        counters['q2_forward_calls'] += 1
        q2 = networks['q2'](tensor, action)
        if any(not torch.isfinite(item).all() for item in (tensor, mean, log_std, action, q1, q2)):
            raise FloatingPointError('终点诊断网络输入/输出非有限，禁止重试。')
    nominal = action.cpu().numpy().copy()
    record = dict(observation_234=observation.tolist(),
                  network_input_234_float32=tensor.cpu().tolist(),
                  actor_mean=mean.cpu().tolist(), actor_log_std=log_std.cpu().tolist(),
                  nominal_deterministic_action=nominal.tolist(),
                  online_q1_prediction=float(q1.item()), online_q2_prediction=float(q2.item()))
    return nominal, record


def threshold_nodes(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """保留实际首次进入时刻两侧的控制节点；不把节点误称为精确事件插值姿态。"""
    nodes = [diagnostic['_r2_initial_state'], *diagnostic['_r2_trajectory']]
    result = {}
    for field in ('first_within_5m_time_s', 'first_within_3m_time_s', 'first_goal_entry_time_s'):
        timestamp = diagnostic[field]
        if timestamp is None:
            result[field] = None
            continue
        before = next((row for row in reversed(nodes) if row['physical_time_s'] <= timestamp),
                      nodes[0])
        after = next((row for row in nodes if row['physical_time_s'] >= timestamp), nodes[-1])
        result[field] = dict(event_time_s=timestamp,
                             preceding_control_state_ned_8d=before['state_ned_8d'],
                             following_control_state_ned_8d=after['state_ned_8d'],
                             preceding_time_s=before['physical_time_s'],
                             following_time_s=after['physical_time_s'])
    return result


def write_trace(path: Path, record: dict[str, Any], diagnostic: dict[str, Any] | None) -> None:
    """详细输入/轨迹只存本机gzip，模式x防止覆盖既有尝试。"""
    value = dict(record)
    if diagnostic is not None:
        value.update(formal_initial_state=diagnostic['_r2_initial_state'],
                     minimum_goal_state=diagnostic['_r2_minimum_state'],
                     trajectory=diagnostic['_r2_trajectory'])
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def run_case(case: Any, group: str, seed: int, networks: dict[str, torch.nn.Module],
             project: ProjectConfig, totals: dict[str, int],
             state_is_legal: Callable[..., bool]) -> dict[str, Any]:
    """固定案例只运行一次至真实终止或1000时域，不接管动作、不更新或写Replay。"""
    config = harness_config(APPROVED, group, seed)
    scenario_id = f'r2-final-fixed-{case.case_id}'
    env = B0NavigationEnv(project, case.initial_state(), case.obstacles(), case.goal(),
                          scenario_id, task_config=config.task)
    counters = {key: 0 for key in totals}
    record = dict(format=FORMAT, group=group, training_seed=seed, case_id=case.case_id,
                  case_seed=case.seed, scenario_id=scenario_id,
                  run_kind='final_fixed_nonlearning_diagnostic', split='fixed_diagnostic',
                  method=config.method, code_version=CODE_VERSION, transition=300000,
                  model_path=str(model_path(group, seed).relative_to(ROOT)),
                  reward_config=asdict(config.task), goal_position_ned_m=case.goal().tolist(),
                  initial_state=asdict(case.initial_state()), maximum_transitions=1000,
                  initial_state_order=['N', 'E', 'D', 'yaw', 'pitch', 'surge', 'yaw_rate',
                                       'pitch_rate'], attempt=1, status='RUNNING',
                  scientific_training_steps=0, scientific_training_updates=0,
                  actor_samples=0, replay_insertions=0, optimizer_steps=0)
    record['initial_state']['position_ned_m'] = case.initial_state().position_ned_m.tolist()
    filename = f'{group}_seed{seed}_{case.case_id}.json.gz'
    diagnostic = None
    failure = None
    try:
        with count_world_steps(counters, 'warmup_control_transitions'):
            observation, reset_info = env.reset(seed=case.seed)
        if counters['warmup_control_transitions'] != 5 or reset_info['warmup_duration_s'] != 1.0:
            raise ValueError('固定诊断必须执行原一秒/五控制步合法warm-up。')
        diagnostic = new_r2_episode_diagnostics(env)
        record['formal_initial_state'] = deepcopy(diagnostic['_r2_initial_state'])
        record['reset_info'] = reset_info
        reward_parts = dict(progress=0.0, goal=0.0, time=0.0, smoothness=0.0)
        legal = state_is_legal(env.world.auv_state, project)
        original_step = env.step
        for _ in range(1000):
            action, prediction = inference(networks, observation, counters)
            record['pending_policy_inference'] = prediction
            input_state = (diagnostic['_r2_trajectory'][-1] if diagnostic['_r2_trajectory']
                           else diagnostic['_r2_initial_state'])
            prediction['policy_input_state_ned_8d'] = input_state['state_ned_8d']
            prediction['policy_input_physical_time_s'] = input_state['physical_time_s']
            with count_world_steps(counters, 'evaluation_env_transitions'):
                observation, reward, terminated, truncated, info = r2_diagnostic_step(
                    env, diagnostic, action, original_step)
            diagnostic['_r2_trajectory'][-1].update(prediction)
            record.pop('pending_policy_inference')
            if not (np.all(np.isfinite(observation)) and np.isfinite(reward)):
                raise FloatingPointError('真实固定诊断转移产生非有限观察/reward。')
            if (not np.array_equal(action, info['executed_action_normalized'])
                    or info['risk_training'] is not False
                    or info['safety_validation'] is not False):
                raise ValueError('B0固定诊断动作或无风险/无执行过滤身份失效。')
            for name, value in info['reward_components'].items():
                reward_parts[name] += value
            legal &= state_is_legal(env.world.auv_state, project)
            if terminated or truncated:
                break
        if not terminated or truncated:
            raise ValueError('固定案例未在原1000步真实时域内合法结束。')
        record.update(status='COMPLETED', complete=True,
                      event=info['failure_type'], terminated=terminated, truncated=truncated,
                      success=info['failure_type'] == 'goal_success',
                      collision=info['failure_type'] == 'collision',
                      boundary=info['failure_type'] == 'operational_boundary_failure',
                      task_timeout=info['failure_type'] == 'task_horizon',
                      physical_time_s=env.world.timestamp_s-reset_info['episode_start_timestamp_s'],
                      reward_components=reward_parts,
                      undiscounted_return=sum(reward_parts.values()),
                      operation_limits_satisfied=bool(legal),
                      threshold_control_nodes=threshold_nodes(diagnostic),
                      minimum_goal_state=diagnostic['_r2_minimum_state'],
                      final_state=diagnostic['_r2_trajectory'][-1]['state_ned_8d'])
        record.update({key: value for key, value in diagnostic.items()
                       if not key.startswith('_r2_')})
        torch.cuda.synchronize()
    except BaseException as error:
        record.update(status='FAILED', complete=False, error=traceback.format_exc())
        failure = error
    finally:
        record.update(counters=counters, trajectory_file=f'trajectories/{filename}')
        for key, value in counters.items():
            totals[key] += value
        write_trace(DESTINATION / 'trajectories' / filename, record, diagnostic)
    if failure is not None:
        raise RuntimeError(f'固定诊断故障，保留反例且不得自动重跑：{filename}') from failure
    return record


def execute() -> dict[str, Any]:
    """完成所有只读前置校验后独占创建尝试标记，部分失败也永不隐式重跑。"""
    ready = preflight()
    if not ready['ready']:
        raise RuntimeError(f"REFUSED: {ready['refusal_reason']}")
    if torch.__version__.split('+')[0] != '2.11.0' or not torch.cuda.is_available():
        raise RuntimeError('既有Torch2.11.0 CUDA目标环境未就绪；不得静默CPU替代。')
    snapshots = {(group, seed): load_snapshot(group, seed) for group in APPROVED['groups']
                 for seed in APPROVED['training_seeds']}
    project = load_project_config(ROOT / APPROVED['project_config'])
    if (project.environment.max_episode_control_steps != 1000
            or project.dynamics.control_dt_s != 0.2 or project.dynamics.integration_dt_s != 0.05
            or project.environment.goal_radius_m != 2.0):
        raise ValueError('固定诊断必须保持原时域、控制/积分周期、成功半径。')
    cases, state_is_legal = load_cases()
    DESTINATION.mkdir(exist_ok=True)
    guard = DESTINATION / 'execution_state.json'
    state = dict(format=FORMAT, status='RUNNING', started_at_utc=utc(), code_version=CODE_VERSION,
                 batch_completed_at=read_json(OUTPUT / 'batch_state.json')['completed_at'],
                 actual_worker_exit=ready['actual_worker_exit'], completed_cases=[],
                 planned_cases=18, maximum_attempts_per_case=1)
    with guard.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    (DESTINATION / 'trajectories').mkdir()
    totals = dict(evaluation_env_transitions=0, warmup_control_transitions=0,
                  actor_forward_calls=0, q1_forward_calls=0, q2_forward_calls=0)
    rows = []
    try:
        for group in APPROVED['groups']:
            for seed in APPROVED['training_seeds']:
                networks = build_networks(snapshots[group, seed])
                for case in cases:
                    state['current_case'] = f'{group}_seed{seed}_{case.case_id}'
                    atomic_json(guard, dict(state, counters=totals))
                    row = run_case(case, group, seed, networks, project, totals, state_is_legal)
                    rows.append(row)
                    state['completed_cases'].append(state['current_case'])
                    atomic_json(guard, dict(state, counters=totals))
                    print(json.dumps(dict(case=state['current_case'], event=row['event'],
                                          transitions=row['counters']['evaluation_env_transitions']),
                                     ensure_ascii=False), flush=True)
                del networks
        verify_code_identity(ROOT, CODE_VERSION)
        report = dict(state, status='COMPLETED', completed_at_utc=utc(), counters=totals,
                      cases=rows, case_count=len(rows), registration_id=REGISTRATION_ID,
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      device=APPROVED['device'], gpu_name=torch.cuda.get_device_name(0),
                      scientific_training_steps=0, scientific_training_updates=0,
                      optimizer_steps=0, actor_samples=0, replay_insertions=0,
                      q_semantics='ONLINE SOFT-Q PREDICTION; NOT DETERMINISTIC RETURN GROUND TRUTH',
                      trajectory_scope='LOCAL ONLY; 234D INPUTS AND EIGHT-DIMENSIONAL STATES',
                      diagnostic_geometry='EXECUTED RK2 LINEAR SEGMENTS; NOT CONTINUOUS ODE ARC')
        atomic_json(DESTINATION / 'summary.json', report)
        atomic_json(guard, {key: value for key, value in report.items() if key != 'cases'})
        return report
    except BaseException:
        state.update(status='FAILED', failed_at_utc=utc(), counters=totals,
                     error=traceback.format_exc(), retry_permitted=False)
        atomic_json(guard, state)
        raise


def guard_self_test() -> dict[str, Any]:
    """纯字典/路径夹具测试门禁；不加载模型，不创建环境或执行任何前向。"""
    batch = dict(registration_id=REGISTRATION_ID, registration=deepcopy(APPROVED),
                 experiment_code_commit=CODE_VERSION, status='COMPLETED', pid=123,
                 completed_runs=list(APPROVED['run_order']), actual_training_transitions=1800000,
                 actual_sac_updates=1740006, runs={})
    for run_id in APPROVED['run_order']:
        group, seed = run_id.split('_seed')
        batch['runs'][run_id] = dict(status='COMPLETED', transitions=300000, updates=290001,
                                    checkpoint_transition=300000, group=group, seed=int(seed),
                                    reward_config=deepcopy(APPROVED['groups'][group]))
    worker = dict(status='EXITED', actual_worker_exit_code=0, actual_computation_pid=123)
    assert completion_problem(batch, worker) is None
    for field, value in (('status', 'RUNNING'), ('completed_runs', []),
                         ('actual_training_transitions', 1799999)):
        broken = deepcopy(batch)
        broken[field] = value
        assert completion_problem(broken, worker) is not None
    for field, value in (('status', 'WAITING'), ('actual_worker_exit_code', 1),
                         ('actual_computation_pid', 124), ('actual_worker_exit_code', None)):
        broken = deepcopy(worker)
        broken[field] = value
        assert completion_problem(batch, broken) is not None
    broken = deepcopy(batch)
    broken['runs']['C300_seed11']['reward_config'] = dict(w_goal=200.0)
    assert completion_problem(broken, worker) is not None
    assert execution_problem(batch, worker, all_models_present=False,
                             attempt_exists=False) == 'FINAL_MODEL_MISSING'
    assert execution_problem(batch, worker, all_models_present=True,
                             attempt_exists=True) == 'DIAGNOSTIC_ATTEMPT_ALREADY_EXISTS_NO_RETRY'
    return dict(status='PASS', checks=11, fixture_kind='PURE_DICTIONARY_NO_ROLLOUT',
                environment_transitions=0, network_forwards=0, gradient_updates=0)


def main() -> int:
    """默认只读preflight；实际18例必须显式execute和可信本机模型确认。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--trusted-local-models', action='store_true')
    parser.add_argument('--self-test-guards', action='store_true')
    args = parser.parse_args()
    if args.self_test_guards:
        if args.execute:
            parser.error('门禁夹具不得混用execute。')
        print(json.dumps(guard_self_test(), ensure_ascii=False, indent=2))
        return 0
    if args.execute and not args.trusted_local_models:
        parser.error('必须显式确认仅读取本项目可信本机终点weights_only模型。')
    report = execute() if args.execute else preflight()
    print(json.dumps(report if not args.execute else dict(status=report['status'],
                                                         counters=report['counters']),
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
