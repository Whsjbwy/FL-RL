"""可信V1最终critic的有界冻结动作排序/部分soft后果核验；不训练、不校准Q。"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.rl.networks import Actor, RewardCritic
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.training.fixed_validation import paired_validation_sensor_streams
from auv_risk_rl.training.harness import states_equal

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
TASK = ROOT / 'results/stage2_b0_repair_r1'
V1 = ROOT / 'results/stage2_b0_mvp_v1'
V1_COMMIT = '45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898'
BRANCHES = ('original', 'los', 'pitch_only', 'surge_only', 'yaw_only')


def helper() -> Any:
    """复用另一个只读诊断程序的预算/五分支，不复制环境实现。"""
    specification = importlib.util.spec_from_file_location(
        'r1_replay_diagnostics', TASK / 'replay_diagnostics.py')
    if specification is None or specification.loader is None:
        raise RuntimeError('缺少已登记冻结重放helper。')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def independent_input(snapshot: dict[str, Any]) -> dict[str, Any]:
    """按原生旋转分量独立重算选定真实失败输入，不调用生产编码或另一次build。"""
    env, actual = snapshot['env'], np.asarray(snapshot['observation'])
    state, config = env.world.auv_state, env.config
    cy, sy = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    cp, sp = math.cos(state.pitch_rad), math.sin(state.pitch_rad)
    body = np.array([[cy * cp, sy * cp, -sp], [-sy, cy, 0.0],
                     [cy * sp, sy * sp, cp]])
    low = np.array(config.environment.position_lower_bound_ned_m)
    high = np.array(config.environment.position_upper_bound_ned_m)
    expected = np.zeros(234, dtype=np.float64)
    goal = body @ (env.goal_position_ned_m - state.position_ned_m)
    expected[:18] = [*(goal / 100.0),
                     *(2.0 * (state.position_ned_m - low) / (high - low) - 1.0),
                     2.0 * (state.surge_speed_mps - 0.3) / 1.2 - 1.0,
                     state.yaw_rate_rad_s / 0.35, state.pitch_rate_rad_s / 0.25,
                     sy, cy, sp, cp, *env.previous_normalized,
                     1.0 - env.world.control_step_index / 1000.0,
                     config.risk.short_horizon_risk_budget / 0.2]
    expected[18:63] = env.rays.ranges_m / config.sensor.range_m
    expected[63:108] = env.rays.valid_masks
    ordered = sorted(env.world.obstacle_states, key=lambda item: (
        math.sqrt(sum(float(value) ** 2 for value in
                      item.position_ned_m - state.position_ned_m)), item.obstacle_id))
    for index, obstacle in enumerate(ordered[:6]):
        offset = 108 + 21 * index
        expected[offset:offset + 3] = body @ (
            obstacle.position_ned_m - state.position_ned_m) / config.sensor.range_m
        expected[offset + 3:offset + 6] = body @ obstacle.velocity_ned_mps
        expected[offset + 18:offset + 21] = [obstacle.radius_m, 0.0, 1.0]
    expected32 = expected.astype(np.float32)
    error = np.abs(expected32 - actual)
    tolerance = 1.0e-5 + 1.0e-5 * np.abs(expected32)
    if actual.shape != (234,) or not np.all(error <= tolerance):
        raise AssertionError('实际失败快照观察不符独立234维参考。')
    return dict(source_trajectory_id=snapshot['source_trajectory_id'],
                model_transition=snapshot['model_transition'],
                training_seed=snapshot['training_seed'],
                task_step=env.world.control_step_index,
                goal_body_m=goal.tolist(),
                goal_distance_m=float(np.linalg.norm(env.goal_position_ned_m
                                                      - state.position_ned_m)),
                observation_max_abs_error=float(error.max()),
                observed_covariance_max_abs=float(max(
                    np.max(np.abs(actual[114 + 21 * index:126 + 21 * index]))
                    for index in range(6))),
                interpretation='Independent current-state encoding; no learned-quality claim.')


def load_final_networks(
    seed: int,
) -> tuple[Actor, RewardCritic, RewardCritic, float, dict[str, Any]]:
    """仅本项目可信300k完整恢复点，抽取模型后释放Replay，不能冒充100kcritic。"""
    source = V1 / f'seed_{seed}' / 'latest_resume.pt'
    state = torch.load(source, map_location='cpu', weights_only=False)
    if (state['code_version'] != V1_COMMIT or state['config']['training_seed'] != seed
            or state['scheduler']['transitions'] != 300000
            or state['torch_version'] != str(torch.__version__)
            or state['agent']['counters']['gradient_updates'] != 290001):
        raise ValueError('最终critic身份与V1真实300k恢复点不符。')
    with torch.random.fork_rng(devices=[]):
        actor, q1, q2 = Actor(), RewardCritic(), RewardCritic()
    for name, network in (('actor', actor), ('q1', q1), ('q2', q2)):
        network.load_state_dict(state['agent']['models'][name], strict=True)
        network.to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)
    alpha = float(state['agent']['log_alpha'].exp())
    identity = dict(source=str(source.relative_to(ROOT)), code_version=state['code_version'],
                    model_transition=300000, gradient_updates=290001,
                    torch_version=state['torch_version'], alpha=alpha,
                    replay_size_at_load=state['agent']['replay']['size'],
                    replay_not_used_or_copied=True)
    del state
    gc.collect()
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise FloatingPointError('冻结最终alpha必须有限正值。')
    return actor, q1, q2, alpha, identity


def rollout(snapshot: dict[str, Any], actor: Actor, alpha: float,
            initial: np.ndarray, seed: int, branch: str, repetition: int,
            diagnostic_seed: int, module: Any) -> dict[str, Any]:
    """Q初始动作无熵项；后续每时刻累加gamma^t(r_t-alpha logpi_t)的部分和。"""
    identity = f'critic_s{seed}_m300000_{branch}_rep{repetition}'
    env, observation = deepcopy(snapshot['env']), snapshot['observation'].copy()
    remaining = 1000 - env.world.control_step_index
    maximum = min(50, remaining)
    if maximum < 1 or env._done:
        raise ValueError('critic诊断快照必须仍有合法可执行时域。')
    ledger = module.BudgetLedger()
    ledger.reserve(identity, maximum)
    output = TASK / 'math' / 'critic_rollouts' / f'{identity}.json'
    if output.exists():
        raise FileExistsError('禁止覆盖或重复critic诊断attempt。')
    before_env = deepcopy(snapshot['env'])
    count, reward_sum, partial_soft = 0, 0.0, 0.0
    entropy_sum = 0.0
    rows = []
    initial_distance = float(np.linalg.norm(env.goal_position_ned_m
                                            - env.world.auv_state.position_ned_m))
    generator = torch.Generator(device='cuda').manual_seed(diagnostic_seed)
    initial_generator = generator.get_state().clone()
    context = (paired_validation_sensor_streams(snapshot['environment_seed'])
               if snapshot['paired'] else nullcontext())
    try:
        with context, torch.inference_mode():
            while count < maximum and not env._done:
                if count == 0:
                    action, log_pi = initial.copy(), None
                else:
                    tensor = torch.as_tensor(observation, device='cuda', dtype=torch.float32)
                    sampled, density = actor.sample(tensor, generator)
                    action, log_pi = sampled.cpu().numpy().copy(), float(density)
                    if not math.isfinite(log_pi):
                        raise FloatingPointError('冻结诊断logpi非有限。')
                if not np.isfinite(action).all():
                    raise FloatingPointError('冻结诊断动作非有限。')
                count += 1
                observation, reward, terminated, truncated, info = env.step(action)
                if not math.isfinite(reward):
                    raise FloatingPointError('冻结诊断reward非有限。')
                exponent = count - 1
                entropy = 0.0 if log_pi is None else -alpha * log_pi
                reward_sum += reward
                partial_soft += 0.999 ** exponent * (reward + entropy)
                entropy_sum += 0.999 ** exponent * entropy
                if not all(math.isfinite(value) for value in
                           (reward_sum, partial_soft, entropy_sum)):
                    raise FloatingPointError('冻结诊断累计reward/entropy非有限。')
                rows.append(dict(offset=exponent, task_step=env.world.control_step_index,
                                 state=module.state_vector(env.world.auv_state),
                                 action=action.tolist(), reward=reward,
                                 reward_components=info['reward_components'],
                                 log_pi=log_pi, entropy_term=entropy,
                                 discount=0.999 ** exponent,
                                 failure_type=info['failure_type'],
                                 terminated=terminated, truncated=truncated))
        if not states_equal(before_env, snapshot['env']):
            raise AssertionError('诊断改变了原冻结快照。')
        if count == 1 and not torch.equal(initial_generator, generator.get_state()):
            raise AssertionError('固定初始动作不应消耗诊断策略随机流。')
        ledger.finish(identity, count, 0)
        result = dict(trajectory_id=identity, training_seed=seed, branch=branch,
                      repetition=repetition, diagnostic_seed=diagnostic_seed,
                      maximum_transitions=maximum, actual_transitions=count,
                      warmup_control_transitions=0, alpha=alpha,
                      initial_action_entropy_term_included=False,
                      subsequent_entropy_position='gamma**t * -alpha*logpi(a_t|s_t), t>=1',
                      partial_soft_return=partial_soft,
                      discounted_subsequent_entropy_sum=entropy_sum,
                      undiscounted_reward_sum=reward_sum,
                      initial_goal_distance_m=initial_distance,
                      final_goal_distance_m=float(np.linalg.norm(
                          env.goal_position_ned_m - env.world.auv_state.position_ned_m)),
                      terminated=env._done, failure_type=rows[-1]['failure_type'],
                      remaining_task_tail_omitted=not env._done,
                      scope='FINITE_PARTIAL_SOFT_RETURN_NOT_EXACT_Q_CALIBRATION',
                      loss_or_update_executed=False, numerical_finite=True,
                      transitions=rows)
        module.write_json(output, result, exclusive=True)
        return {key: value for key, value in result.items() if key != 'transitions'}
    except BaseException as error:
        ledger.finish(identity, count, 0, 'ERROR')
        module.write_json(output, dict(status='ERROR', trajectory_id=identity,
                                       actual_transitions=count, error=repr(error),
                                       transitions=rows),
                          exclusive=True)
        raise


def main() -> int:
    """一次登记的3seed×5动作×4随机后续；不自动重试、不写Replay。"""
    module = helper()
    inventory = json.loads((TASK / 'frozen_snapshots.json').read_text(encoding='utf-8'))
    if (TASK / 'math' / 'critic_result.json').exists():
        raise FileExistsError('critic诊断已有结果，不自动重跑。')
    source_rows = inventory['snapshots']
    inputs, records, sources = [], [], []
    seed_root = 20261007
    effective_seeds = {seed: [int(SeedManager(seed_root).get_rng(
        f'r1/frozen-soft-return/seed-{seed}/stream-{repetition}').integers(0, 2**63))
        for repetition in range(4)] for seed in (11, 22, 33)}
    module.write_json(TASK / 'math' / 'critic_registration.json', dict(
        registered_at=datetime.now(UTC).isoformat(), diagnostic_root_seed=seed_root,
        effective_seeds=effective_seeds, branches=BRANCHES,
        branch_coupling='same repeated diagnostic stream for each fixed initial action',
        maximum_control_transitions=3000, maximum_trajectories=60,
        initial_action_entropy=False, subsequent_entropy='t>=1, gamma**t*(-alpha*logpi_t)',
        frozen_model_scope='FINAL300K_ONLY', no_train_replay_write=True), exclusive=True)
    # 六个冻结失败快照均独立重算输入；只把300kCV快照用于最终critic核验。
    for row in source_rows:
        snapshot = torch.load(TASK / row['local_snapshot'], map_location='cpu', weights_only=False)
        inputs.append(independent_input(snapshot))
        del snapshot
    module.write_json(TASK / 'math' / 'actual_failure_input_review.json', dict(
        snapshots=inputs, tolerance=dict(atol=1.0e-5, rtol=1.0e-5),
        independently_recomputed=True, scientific_training_steps=0))
    for seed in (11, 22, 33):
        selected = [row for row in source_rows if row['training_seed'] == seed
                    and row['model_transition'] == 300000]
        if len(selected) != 1:
            raise ValueError('每seed必须恰有预登记的最终CV首失败快照。')
        snapshot = torch.load(TASK / selected[0]['local_snapshot'],
                              map_location='cpu', weights_only=False)
        actor, q1, q2, alpha, source = load_final_networks(seed)
        saved_actor, device = module.load_actor(seed, 300000)
        if any(not torch.equal(tensor, saved_actor.state_dict()[key])
               for key, tensor in actor.state_dict().items()):
            raise AssertionError('300k完整critic恢复点与原300k小Actor参数不一致。')
        frozen = [deepcopy(network.state_dict()) for network in (actor, q1, q2)]
        with torch.inference_mode():
            original, _ = module.policy_values(actor, snapshot['observation'], device)
            observation = torch.as_tensor(snapshot['observation'], device=device,
                                          dtype=torch.float32)
            values = []
            for branch in BRANCHES:
                action = module.action_branch(original, snapshot['env'], branch)
                tensor_action = torch.as_tensor(action, device=device, dtype=torch.float32)
                first = float(q1(observation, tensor_action))
                second = float(q2(observation, tensor_action))
                if not math.isfinite(first) or not math.isfinite(second):
                    raise FloatingPointError('冻结最终critic非有限。')
                values.append(dict(branch=branch, initial_action=action.tolist(),
                                   q1=first, q2=second, minimum_q=min(first, second)))
                for repetition in range(4):
                    record = rollout(snapshot, actor, alpha, action, seed, branch, repetition,
                                     effective_seeds[seed][repetition], module)
                    record.update(q1=first, q2=second, minimum_q=min(first, second))
                    records.append(record)
        ranking = sorted(values, key=lambda value: -value['minimum_q'])
        source.update(source_trajectory_id=snapshot['source_trajectory_id'],
                      task_step=snapshot['task_step'], actions=values,
                      ranking_descending_min_q=ranking)
        sources.append(source)
        for network, before in zip((actor, q1, q2), frozen, strict=True):
            if not states_equal(before, network.state_dict()) or any(
                    parameter.grad is not None for parameter in network.parameters()):
                raise AssertionError('critic诊断改变网络参数或产生梯度。')
        del snapshot, actor, saved_actor, q1, q2, frozen
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(status='FROZEN_FINAL_CRITIC_DIAGNOSTIC_COMPLETED', sources=sources,
                  rollouts=records, trajectory_count=len(records),
                  diagnostic_control_transitions=sum(row['actual_transitions'] for row in records),
                  warmup_control_transitions=0, scientific_training_steps=0,
                  scientific_training_updates=0, training_replay_writes=0, gradient_operations=0,
                  actor_and_critic_parameters_unchanged=True,
                  evidence_level='PLAUSIBLE_HYPOTHESIS_OR_SUPPORTED_LOCAL_SHORT_HORIZON_MECHANISM',
                  claim='Q rankings and finite partial soft returns; not exact Q calibration.',
                  missing_weights=['No retained100kcritic', 'No retained50k/75kweights'],
                  finished_at=datetime.now(UTC).isoformat())
    module.write_json(TASK / 'math' / 'critic_result.json', result, exclusive=True)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('sources', 'rollouts')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
