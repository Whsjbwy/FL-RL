"""R1真实结束后审计小模型和终止分层；只读原始证据，不运行环境或学习。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(TASK)]
ACTUAL_COMPUTATION_PID = 37516
LAUNCHER_PID = 38168
TRAINING_COMMAND_NAME = 'r1_scientific_batch'
SEEDS = (11, 22, 33)
MODEL_POINTS = (25000, 50000, 75000, 100000)
COMPONENTS = ('progress', 'goal', 'time', 'smoothness')
DISTANCE_FIELDS = ('initial_distance_m', 'final_distance_m', 'minimum_goal_distance_m',
                   'minimum_goal_distance_time_s', 'first_within_10m_time_s',
                   'first_goal_entry_time_s')
NETWORKS = ('actor', 'q1', 'q2', 'target_q1', 'target_q2')


def read_json(path: Path) -> Any:
    """缺证据即失败，不创建缺失的回执或修改历史内容。"""
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, message: str) -> None:
    """实际状态不符时立即拒绝最终证据PASS。"""
    if not condition:
        raise ValueError(message)


def completion_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    """批次完成、实际计算PID及venv启动器退出三者均真实成立才继续。"""
    from analyze_r1 import REGISTRATION_ID, validate_completed_state

    state = validate_completed_state()
    require(state.get('pid') == ACTUAL_COMPUTATION_PID, '实际batch计算PID不是冻结的37516。')
    actual_path = TASK/'actual_training_worker_exit.json'
    actual = read_json(actual_path)
    require(actual.get('status') == 'EXITED'
            and actual.get('actual_computation_pid') == ACTUAL_COMPUTATION_PID
            and actual.get('actual_worker_exit_code') == 0,
            '真实计算PID37516尚未退出0；不得用启动器退出或预期完成代替。')
    launcher_path = TASK/f'{TRAINING_COMMAND_NAME}.worker.json'
    launcher = read_json(launcher_path)
    require(launcher.get('status') == 'EXITED' and launcher.get('worker_pid') == LAUNCHER_PID
            and launcher.get('actual_worker_exit_code') == 0,
            'venv启动器38168未真实退出0或回执身份不符。')
    require(launcher.get('code_commit') == state['experiment_code_commit'],
            '启动器代码版本与实际批次不符。')
    expected_command = [
        '.venv-b1/Scripts/python.exe', 'scripts/run_b0_training.py',
        '--repair-registration', 'configs/stage2_b0_repair_r1.yaml', '--execute',
        '--run-kind', 'scientific_training', '--transition-budget', '300000',
        '--output-dir', 'results/stage2_b0_repair_r1',
    ]
    require(launcher.get('command') == expected_command, '启动器实际命令与本轮批次不符。')
    commands = []
    for line in (TASK/'commands.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get('name') == TRAINING_COMMAND_NAME:
                commands.append(row)
    require(len(commands) == 1, '实际训练命令结束记录缺失或重复，拒绝混合attempt。')
    command = commands[0]
    require(command.get('exit_code') == 0 and command.get('actual_worker_pid') == LAUNCHER_PID
            and command.get('code_commit') == state['experiment_code_commit']
            and command.get('command') == expected_command
            and command.get('start_utc') == launcher['start_utc'],
            'commands.jsonl实际结束记录与启动器回执不一致。')
    start = datetime.fromisoformat(launcher['start_utc'])
    launcher_end = datetime.fromisoformat(launcher['end_utc'])
    command_end = datetime.fromisoformat(command['end_utc'])
    require(all(stamp.tzinfo is not None for stamp in (start, launcher_end, command_end)),
            '实际开始/结束时间必须含时区，不能比较含糊的本地时间。')
    # record_command先写启动器结束回执，再单独取时追加commands；这不是同一次取时。
    require(start <= launcher_end <= command_end,
            '结束时间必须不早于开始，且符合启动器回执先于commands记录的实际写入顺序。')
    require(state['registration_id'] == REGISTRATION_ID, '实际批次R1登记身份不符。')
    return state, dict(
        status='PASS', actual_computation_pid=ACTUAL_COMPUTATION_PID,
        actual_computation_exit_code=actual['actual_worker_exit_code'],
        launcher_pid=LAUNCHER_PID, launcher_exit_code=launcher['actual_worker_exit_code'],
        actual_process_receipt=actual_path.relative_to(ROOT).as_posix(),
        launcher_receipt=launcher_path.relative_to(ROOT).as_posix(),
        confirmed_command_record=command,
        launcher_end_utc=launcher['end_utc'], command_end_utc=command['end_utc'],
        timestamp_rule='start <= launcher end <= commands end; recorder uses separate clock reads',
        note='Windows进程句柄记录实际计算进程退出；启动器退出另行核对，不互相替代。')


def model_shapes(name: str) -> dict[str, tuple[int, ...]]:
    """按已冻结234/237→256→256结构独立列明期望，避免新建随机网络。"""
    input_dimension, output_dimension = (234, 6) if name == 'actor' else (237, 1)
    return {'net.0.weight': (256, input_dimension), 'net.0.bias': (256,),
            'net.2.weight': (256, 256), 'net.2.bias': (256,),
            'net.4.weight': (output_dimension, 256),
            'net.4.bias': (output_dimension,)}


def audit_models(state: dict[str, Any], *, trusted_local: bool) -> list[dict[str, Any]]:
    """只weights_only读取本项目12个小快照，验证身份/参数/温度，不加载Replay。"""
    require(trusted_local, '须显式--trusted-local确认只读本项目本机小模型。')
    import torch
    from analyze_r1 import REGISTRATION_ID

    from auv_risk_rl.training.repair_r1 import harness_config
    from auv_risk_rl.training.repair_registration import APPROVED

    rows = []
    for seed in SEEDS:
        expected_config = harness_config(APPROVED, seed).sac
        expected_sac = vars(expected_config)
        for point in MODEL_POINTS:
            path = TASK/'models'/f'seed_{seed}_{point}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            expected = dict(format='b0-repair-r1-network-snapshot-v1',
                            registration_id=REGISTRATION_ID, seed=seed, transition=point,
                            code_version=state['experiment_code_commit'],
                            run_kind='scientific_training', method='B0_FULL_STATE_ORDINARY_SAC',
                            branch='L2_COMMON_LEARNING_RATE')
            require(all(payload.get(key) == value for key, value in expected.items()),
                    f'小模型实际格式/seed/时点/代码/登记/方法不符: {path.name}')
            require(payload.get('sac_config') == expected_sac,
                    f'小模型完整SAC配置或有效派生seed不符: {path.name}')
            counters = payload['counters']
            expected_updates = point-9999
            require(counters.get('environment_steps') == point
                    and counters.get('gradient_updates') == expected_updates
                    and all(counters.get(key) == expected_updates
                            for key in ('actor', 'q1', 'q2', 'alpha')),
                    f'小模型真实transition/update/optimizer计数不符: {path.name}')
            require(set(payload['models']) == set(NETWORKS),
                    f'小模型必须齐全包含Actor/两个Q/两个target: {path.name}')
            network_rows = []
            for name in NETWORKS:
                network = payload['models'][name]
                shapes = model_shapes(name)
                require(set(network) == set(shapes), f'{path.name}的{name}参数键不符。')
                for key, shape in shapes.items():
                    tensor = network[key]
                    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape
                            and tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
                            and bool(torch.isfinite(tensor).all()),
                            f'{path.name}的{name}.{key}形状/dtype/有限性不符。')
                network_rows.append(dict(name=name, parameter_tensor_count=len(network),
                                         parameter_count=sum(t.numel() for t in network.values()),
                                         stored_dtype='float32', all_finite=True))
            require(set(payload['actor']) == set(payload['models']['actor'])
                    and all(torch.equal(payload['actor'][key], tensor)
                            for key, tensor in payload['models']['actor'].items()),
                    f'Actor便捷导出与完整小模型Actor不一致: {path.name}')
            log_alpha = payload['log_alpha']
            require(isinstance(log_alpha, torch.Tensor) and log_alpha.numel() == 1
                    and log_alpha.dtype == torch.float32 and bool(torch.isfinite(log_alpha).all()),
                    f'小模型log_alpha缺失或非有限: {path.name}')
            alpha = float(log_alpha.exp())
            require(math.isfinite(alpha) and alpha > 0, f'小模型alpha非正/非有限: {path.name}')
            rows.append(dict(seed=seed, transition=point, status='PASS',
                             path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                             complete_sac_updates=expected_updates,
                             optimizer_steps=sum(counters[key]
                                                 for key in ('actor', 'q1', 'q2', 'alpha')),
                             networks=network_rows, alpha=alpha,
                             sac_config=payload['sac_config'],
                             code_version=payload['code_version'],
                             replay_loaded=False, gradient_operations=0))
    require(len(rows) == 12, '必须实际核对全部3seed×4时点小模型。')
    return rows


def event_label(row: dict[str, Any]) -> str:
    """未完成片段不伪装成功/物理终止；保留原有failure字段。"""
    if row['complete']:
        require(row['failure_type'] in ('goal_success', 'collision',
                                      'operational_boundary_failure', 'task_horizon'),
                'complete日志包含未知物理终止类型。')
        return row['failure_type']
    return 'INCOMPLETE_FRAGMENT'


def stratified_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分层只统计实际测量，首次接近时间明确以真正进入该球的episode为条件。"""
    from analyze_r1 import moment

    total = len(rows)
    measured = {name: sum(name in row for row in rows) for name in DISTANCE_FIELDS}
    moments = {}
    for name in DISTANCE_FIELDS:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        moments[name] = dict(moment(values), measured_episodes=measured[name],
                             unmeasured_episodes=total-measured[name],
                             measured_null_episodes=measured[name]-len(values))
    subtype = Counter(row.get('boundary_subtype') or 'NOT_RECORDED' for row in rows
                      if row['failure_type'] == 'operational_boundary_failure')
    labels = Counter(label for row in rows for label in (row.get('boundary_constraints') or []))
    return dict(
        episode_count=total, complete_physical_episode_count=sum(row['complete'] for row in rows),
        episode_control_transitions=sum(row['steps'] for row in rows),
        boundary_subtypes=dict(subtype), boundary_constraint_membership=dict(labels),
        constraint_counting_note='同刻多个约束各计一次，membership总和可大于边界episode数。',
        fragment_flags={name: sum(bool(row.get(name)) for row in rows)
                        for name in ('phase_boundary', 'budget_stop', 'external_truncation')},
        target_distance_and_time=moments,
        first_approach_time_note='first10m/first2m的均值仅以实际进入该球的episode为条件。',
        entered_10m_count=(sum(row.get('first_within_10m_time_s') is not None for row in rows)
                           if measured['first_within_10m_time_s'] else None),
        entered_2m_count=(sum(row.get('first_goal_entry_time_s') is not None for row in rows)
                          if measured['first_goal_entry_time_s'] else None),
        reward=moment([float(row['reward']) for row in rows]),
        reward_components={name: moment([float(row['reward_components'][name]) for row in rows])
                           for name in COMPONENTS})


def audit_termination_strata(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """仅确认的C原100k与R1日志，按seed/时点/物理终止分层，不运行策略。"""
    from analyze_r1 import CONTROL_COMMIT, V1, bin_stop

    from auv_risk_rl.training.mvp_analysis import confirmed_records
    from auv_risk_rl.training.repair_registration import REGISTRATION_ID

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    confirmation: dict[str, Any] = {}
    actual_counts: dict[str, Any] = {}
    for group, root, code, registration in (
        ('C_3e-4', V1, CONTROL_COMMIT, 'STAGE2_B0_MVP_BATCH_V1'),
        ('R1_1e-4', TASK, state['experiment_code_commit'], REGISTRATION_ID),
    ):
        for seed in SEEDS:
            diagnostics: dict[str, Any] = {}
            transitions, updates, points = 0, 0, []
            for kind, row in confirmed_records(root, seed, code, diagnostics,
                                               registration_id=registration):
                if kind == 'episode' and row['task_profile'] == 'obstacle_free':
                    require(row['last_transition'] <= 100000, '不能混入100k以后空场景。')
                    transitions += row['steps']
                    key = (group, 'training_episode', seed, bin_stop(row['last_transition']),
                           'whole_episode_by_final_transition', event_label(row))
                    groups[key].append(row)
                elif kind == 'validation' and row['profile'] == 'obstacle_free':
                    points.append((row['at_transition'], row['full']))
                    scopes = ('all_registered', 'paired_monitor30') if row['full'] else (
                        'all_registered',)
                    for scope in scopes:
                        for episode in row['episodes']:
                            if scope == 'paired_monitor30' and episode['index'] >= 30:
                                continue
                            key = (group, 'Val300' if row['full'] else 'monitor30', seed,
                                   row['at_transition'], scope, event_label(episode))
                            groups[key].append(episode)
                elif kind == 'update' and row['transition'] <= 100000:
                    require(row['transition'] == 10000+updates, '已确认更新并非连续UTD=1。')
                    updates += 1
            require(transitions == 100000 and updates == 90001,
                    f'{group} seed{seed}确认100k/90001实测计数不符。')
            require(sorted(points) == [(0, False), (25000, False), (50000, False),
                                       (75000, False), (100000, True)],
                    f'{group} seed{seed}固定验证点缺失/重复。')
            confirmation[f'{group}_seed_{seed}'] = diagnostics
            actual_counts[f'{group}_seed_{seed}'] = dict(transitions=transitions, updates=updates)
    strata = []
    for (group, source, seed, point, scope, event), episodes in sorted(groups.items()):
        strata.append(dict(group=group, source=source, training_seed=seed,
                           transition_point_or_bin_end=point, scope=scope, termination=event,
                           **stratified_summary(episodes)))
    return strata, dict(confirmed_records=confirmation, actual_counts=actual_counts)


def main() -> int:
    """只有所有实际结束证据齐全后才写本轮结果补充，不触碰原始证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trusted-local', action='store_true')
    args = parser.parse_args()
    state, completion = completion_gate()
    models = audit_models(state, trusted_local=args.trusted_local)
    strata, confirmation = audit_termination_strata(state)
    from analyze_r1 import csv_write, json_write

    result = dict(
        status='FINAL_EVIDENCE_AUDIT_PASS', generated_at=datetime.now(UTC).isoformat(),
        registration_id=state['registration_id'],
        experiment_code_commit=state['experiment_code_commit'],
        completion_gate=completion, small_model_audits=models,
        termination_strata=strata, raw_log_confirmation=confirmation,
        operation_counts=dict(environment_transitions=0, replay_writes=0,
                              complete_sac_updates=0, optimizer_steps=0,
                              policy_replays=0, checkpoint_replay_loads=0),
        interpretation_boundaries=[
            '仅证明真实结束和必要证据齐全，不等于Stage2科学GO。',
            'C与R1仍为独立run，固定100k主终点不变，不选择best模型。',
            'C历史训练最小目标距离/首次接近/边界子类未测，保持null或NOT_RECORDED。',
            '无观测值不以0代替；首次接近时间仅对已进入对应球的episode求均值。',
            '整episode结束分箱并非该区间逐transition统计，不混淆二者。'])
    output = TASK/'analysis'
    json_write(output/'final_evidence_audit.json', result)
    csv_write(output/'termination_strata.csv', strata)
    print(json.dumps(dict(status=result['status'], actual_computation_pid=ACTUAL_COMPUTATION_PID,
                          actual_computation_exit_code=0, audited_small_models=len(models),
                          termination_strata_count=len(strata), environment_transitions=0,
                          complete_sac_updates=0), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
