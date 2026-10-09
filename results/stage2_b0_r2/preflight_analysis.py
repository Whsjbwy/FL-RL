"""R2登记冻结后的只读离线审核；不加载策略、不重放环境、不执行学习。"""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
R1 = ROOT/'results'/'stage2_b0_repair_r1'
SEEDS = (11, 22, 33)
THRESHOLDS = (10., 5., 3.)
GAMMA = .999
REWARD_TOLERANCE = 1e-8
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def read(path: Path) -> Any:
    """只读实际UTF-8证据，缺失不补造。"""
    return json.loads(path.read_text(encoding='utf-8'))


def dot(a: list[float], b: list[float]) -> float:
    """独立标量内积，不调用生产几何或奖励函数。"""
    return math.fsum(x*y for x, y in zip(a, b, strict=True))


def distance(a: list[float], b: list[float]) -> float:
    """NED欧氏距离，单位m。"""
    return math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b, strict=True)))


def protocol_evidence() -> dict[str, Any]:
    """直接读取本机Word原生表，只保留奖励范围与Stage2有限修复短依据。"""
    path = ROOT/'handoff'/'protocol'/'LOCAL_v2_0.docx'
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read('word/document.xml')).find('w:body', NS)
    if body is None:
        raise ValueError('Word原生正文缺失。')
    tables = body.findall('w:tbl', NS)

    def rows(index: int) -> list[list[str]]:
        return [[''.join(cell.itertext()) for cell in row.findall('w:tc', NS)]
                for row in tables[index].findall('w:tr', NS)]

    weights = rows(54)
    goal = next(row for row in weights if row[0] == '到达 wg')
    if goal[1:3] != ['100', '50–200']:
        raise ValueError('真实Word到达奖励表与R2登记不符，必须人工核对。')
    stage2 = dict(rows(88)[1:])
    return dict(source=path.relative_to(ROOT).as_posix(), source_part='word/document.xml',
                reward_table_index_zero_based=54, goal_row=goal,
                reward_rows=weights, stage2_table_index_zero_based=88,
                stage2_repair_action=stage2['Repair action'],
                stage2_maximum_repair_attempts=stage2['Maximum repair attempts'],
                stage2_stop_condition=stage2['Stop condition'],
                goal100_to200_within_original_validation_range=True,
                interpretation='真实允许区间50–200；100与200都是允许值，不把原区间改写成100–200。')


def confirmed_r1(kind: str) -> list[dict[str, Any]]:
    """仅读取已经安全checkpoint确认的episode/validation，不扫update或加载Replay。"""
    result = []
    code = read(R1/'batch_state.json')['experiment_code_commit']
    for seed in SEEDS:
        seen = set()
        for segment in read(R1/f'seed_{seed}'/'segments.json'):
            cutoff = segment['valid_log_sequence']
            if cutoff is None:
                raise ValueError('R1日志尚未确认，不能用于R2审核。')
            path = (R1/segment['path']/f'{kind}.jsonl').resolve()
            path.relative_to(R1.resolve())
            with path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sequence = row['log_sequence']
                    if sequence > cutoff:
                        continue
                    expected = dict(registration_id='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
                                    training_seed=seed, code_version=code,
                                    segment_id=segment['segment_id'],
                                    method='B0_FULL_STATE_ORDINARY_SAC',
                                    run_kind='scientific_training')
                    if any(row.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'历史日志身份不符: {path}:{line_number}')
                    if sequence in seen:
                        raise ValueError('历史确认日志序号重复。')
                    seen.add(sequence)
                    row['_source'] = dict(path=path.relative_to(ROOT).as_posix(),
                                          line=line_number, valid_log_sequence=cutoff)
                    result.append(row)
    return result


def reward_audit(training: list[dict[str, Any]],
                 validations: list[dict[str, Any]]) -> dict[str, Any]:
    """同一行为离线重计100/200；失败差0、成功差100，不声称行为或梯度会改善。"""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fragments = Counter()
    for row in training:
        if row['complete']:
            grouped['training_complete', row['training_seed'], None, None].append(row)
        else:
            fragments[str(row['training_seed'])] += 1
    for evaluation in validations:
        if evaluation['profile'] != 'obstacle_free':
            raise ValueError('本审核不混入CV任务。')
        key = ('Val300' if evaluation['full'] else 'monitor30', evaluation['training_seed'],
               evaluation['at_transition'], evaluation['count'])
        grouped[key].extend(evaluation['episodes'])
    results, residual = [], 0.
    for (scope, seed, point, n), episodes in sorted(grouped.items(), key=str):
        original, modified, deltas, pieces = [], [], [], []
        success_steps = successes = 0
        for row in episodes:
            if not row['complete']:
                raise ValueError('固定物理评价中不应混入未完成片段。')
            parts = row['reward_components']
            success = row['failure_type'] == 'goal_success'
            before = math.fsum(parts.values())
            after = math.fsum((parts['progress'], 2*parts['goal'], parts['time'],
                               parts['smoothness']))
            residual = max(residual, abs(before-row['reward']),
                           abs(parts['goal']-(100. if success else 0.)),
                           abs((after-before)-(100. if success else 0.)))
            if row.get('initial_distance_m') is not None:
                residual = max(residual, abs(parts['progress']-(
                    row['initial_distance_m']-row['final_distance_m'])))
            residual = max(residual, abs(parts['time']+.01*row['physical_time_s']/.2))
            original.append(before)
            modified.append(after)
            deltas.append(after-before)
            pieces.append(parts)
            successes += int(success)
            success_steps += row['steps'] if success else 0
        count = len(episodes)
        results.append(dict(scope=scope, seed=seed, at_transition=point,
                            expected_evaluation_count=n, complete_episodes=count,
                            successful_complete_episodes=successes,
                            successful_episode_transitions=success_steps,
                            successful_terminal_transitions=successes,
                            all_complete_episode_transitions=sum(r['steps'] for r in episodes),
                            reward100_mean=math.fsum(original)/count,
                            same_behavior_reward200_mean=math.fsum(modified)/count,
                            mean_delta=math.fsum(deltas)/count,
                            expected_mean_delta=100*successes/count,
                            reward100_component_means={
                                name: math.fsum(r[name] for r in pieces)/count
                                                      for name in ('progress', 'goal', 'time',
                                                                   'smoothness')}))
    if residual > REWARD_TOLERANCE:
        raise ValueError(f'独立离线奖励核对超预先固定容差: {residual}')
    return dict(source='R1 confirmed episode/validation JSONL; no update/Replay load',
                whole_episode_reward_residual_max_abs=residual,
                predeclared_float64_reward_tolerance=REWARD_TOLERANCE,
                incomplete_training_fragments_excluded=dict(fragments), scopes=results,
                limitation='是同一既有轨迹的奖励重计，不是reward200的实际训练/策略表现。')


def sphere_entry(start: list[float], end: list[float], goal: list[float],
                 radius: float) -> float | None:
    """独立二次式求直线分段首次进入球的比例，不推进世界。"""
    offset = [a-b for a, b in zip(start, goal, strict=True)]
    delta = [b-a for a, b in zip(start, end, strict=True)]
    c, a = dot(offset, offset)-radius**2, dot(delta, delta)
    if c <= 0:
        return 0.
    if a == 0:
        return None
    b = 2*dot(offset, delta)
    disc = b*b-4*a*c
    if disc < 0:
        return None
    root = (-b-math.sqrt(disc))/(2*a)
    return root if 0 <= root <= 1 else None


def capture_entry(row: dict[str, Any], segment: dict[str, Any], fraction: float,
                  goal: list[float], origin: float) -> dict[str, Any]:
    """实际执行0.05s线性分段上的插值状态；当前指令来自所属真实控制步。"""
    first, proposed = segment['start_state'], segment['uncommitted_proposed_state']
    state = [a+fraction*(b-a) for a, b in zip(first, proposed, strict=True)]
    north, east, down = (a-b for a, b in zip(goal, state[:3], strict=True))
    yaw, pitch = state[3:5]
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    body = [cy*cp*north+sy*cp*east-sp*down, -sy*north+cy*east,
            cy*sp*north+sy*sp*east+cp*down]
    desired_yaw = math.atan2(east, north)
    desired_pitch = -math.atan2(down, math.hypot(north, east))
    yaw_error = math.atan2(math.sin(desired_yaw-yaw), math.cos(desired_yaw-yaw))
    return dict(task_step=row['task_step'], time_since_formal_start_s=(
        segment['timestamp_s']+.05*fraction-origin), distance_m=distance(state[:3], goal),
        state_ned_yaw_pitch_surge_rates=state, goal_body_m=body,
        yaw_error_rad=yaw_error, pitch_error_rad=desired_pitch-pitch,
        normalized_command=row['executed_branch_action'], physical_command=row['physical_command'],
        actual_surge_yaw_rate_pitch_rate=state[5:8],
        command_minus_response=[a-b for a, b in zip(
            row['physical_command'], state[5:8], strict=True)])


def detailed_trace(summary: dict[str, Any], group: str) -> dict[str, Any]:
    """只读已存在gzip详细轨迹；每条扫描一次，不重新调用Actor或环境。"""
    path = R1/'diagnostic_replay'/summary['trajectory_id']/'full_trace.jsonl.gz'
    goal = summary['goal_position_ned_m']
    captures: dict[str, Any] = {str(int(radius)): None for radius in THRESHOLDS}
    max_reward_error = max_command_error = 0.
    discounted = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        origin = None
        for row in map(json.loads, stream):
            if origin is None:
                origin = row['timestamp_before_s']
            before = distance(row['state_before'][:3], goal)
            after = distance(row['state_after'][:3], goal)
            action, previous = row['executed_branch_action'], row['previous_action']
            reference = dict(progress=before-after,
                             goal=100. if row['failure_type'] == 'goal_success' else 0.,
                             time=-.01*(row['timestamp_after_s']-row['timestamp_before_s'])/.2,
                             smoothness=-.02*math.fsum((a-b)**2 for a, b in zip(
                                 action, previous, strict=True)))
            max_reward_error = max(max_reward_error,
                                   abs(math.fsum(reference.values())-row['reward']),
                                   *(abs(value-row['reward_components'][name])
                                     for name, value in reference.items()))
            expected = [.3+1.2*(action[0]+1)/2, .35*action[1], .25*action[2]]
            max_command_error = max(max_command_error, *(abs(a-b) for a, b in zip(
                expected, row['physical_command'], strict=True)))
            factor = GAMMA**(row['task_step']-1)
            for name, value in reference.items():
                discounted[name].append(factor*value)
            for segment in row['integration_segments']:
                for radius in THRESHOLDS:
                    key = str(int(radius))
                    if captures[key] is not None:
                        continue
                    fraction = sphere_entry(segment['start_state'][:3],
                                            segment['uncommitted_proposed_state'][:3], goal, radius)
                    if fraction is not None and fraction <= segment['executed_fraction']:
                        captures[key] = capture_entry(row, segment, fraction, goal, origin)
    if max(max_reward_error, max_command_error) > REWARD_TOLERANCE:
        raise ValueError(f'详细历史轨迹出现独立reward/action不一致: {path}')
    discounted_parts = {name: math.fsum(values) for name, values in discounted.items()}
    return dict(group=group, trajectory_id=summary['trajectory_id'],
                source=path.relative_to(ROOT).as_posix(), seed=summary['training_seed'],
                model_transition=summary['model_transition'], failure_type=summary['failure_type'],
                closest_goal_distance_m=summary['closest_approach']['distance_m'],
                boundary_subtypes=summary['boundary_subtypes'], first_capture=captures,
                max_abs_independent_reward_error=max_reward_error,
                max_abs_independent_action_mapping_error=max_command_error,
                discounted_task_reward_components=discounted_parts,
                reward200_minus100_discounted=discounted_parts['goal'],
                discount_origin='first formal reward exponent 0; gamma=.999 per control transition',
                response_scope='0.05s executed linear segment interpolation; no new integration')


def node_success(episode: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """已保留成功控制节点轨迹折扣对照；不把任务return当soft Q校准真值。"""
    previous_position = episode['initial_position_ned_m']
    goal = episode['goal_position_ned_m']
    sums = defaultdict(list)
    residual = 0.
    for index, node in enumerate(episode['trajectory']):
        progress = distance(previous_position, goal)-distance(node['position_ned_m'], goal)
        residual = max(residual, abs(progress-node['reward_components']['progress']))
        for name, value in node['reward_components'].items():
            sums[name].append(GAMMA**index*value)
        previous_position = node['position_ned_m']
    if residual > REWARD_TOLERANCE:
        raise ValueError('保留成功轨迹的独立进展重计不符。')
    parts = {name: math.fsum(values) for name, values in sums.items()}
    return dict(seed=episode['training_seed'], at_transition=evaluation['at_transition'],
                index=episode['index'], scenario_id=episode['scenario_id'],
                source=evaluation['_source'], steps=episode['steps'],
                discounted_reward100_components=parts,
                discounted_goal200=2*parts['goal'], goal200_minus100=parts['goal'],
                discounted_goal100_to_progress_ratio=(parts['goal']/parts['progress']
                                                      if parts['progress'] != 0 else None),
                independent_progress_error=residual,
                limitation='仅原任务折扣奖励，不含后续策略熵，不是随机策略soft Q的真值。')


def main() -> int:
    """已冻结登记后运行一次只读审核并写小型报告；0训练、0重放。"""
    started = datetime.now(UTC).isoformat()
    git = ['git', '-c', f'safe.directory={ROOT.as_posix()}']
    commit = subprocess.run(git+['rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    protocol = protocol_evidence()
    training, validations = confirmed_r1('episode'), confirmed_r1('validation')
    rewards = reward_audit(training, validations)
    traces = []
    for seed in SEEDS:
        for case in ('R01_horizontal_empty', 'R02_deeper_empty', 'R03_shallower_empty'):
            summary = read(R1/'diagnostic_replay'/f'r1_final_s{seed}_{case}'/'summary.json')
            traces.append(detailed_trace(summary, 'R1_1e-4_fixed100k'))
        for index in (0, 1, 2):
            identifier = f'val_s{seed}_m100000_obstacle_free_i{index}'
            summary = read(R1/'diagnostic_replay'/identifier/'summary.json')
            traces.append(detailed_trace(summary, 'V1_3e-4_Val_index012100k_HISTORICAL'))
    successful = []
    for seed in SEEDS:
        ordered = sorted((evaluation for evaluation in validations
                          if evaluation['training_seed'] == seed),
                         key=lambda row: row['at_transition'])
        actual = retained = None
        for evaluation in ordered:
            for episode in sorted(evaluation['episodes'], key=lambda row: row['index']):
                if episode['failure_type'] != 'goal_success':
                    continue
                if actual is None:
                    actual = dict(point=evaluation['at_transition'], index=episode['index'],
                                  source=evaluation['_source'],
                                  detailed_nodes_retained='trajectory' in episode)
                if retained is None and 'trajectory' in episode:
                    retained = node_success(episode, evaluation)
        successful.append(dict(seed=seed, earliest_actual_success=actual,
                               earliest_retained_success_nodes=retained))
    models = read(R1/'analysis'/'final_evidence_audit.json')['small_model_audits']
    inventory = []
    for row in models:
        path = ROOT/row['path']
        inventory.append({key: row[key] for key in ('seed', 'transition', 'status', 'path',
                                                    'complete_sac_updates', 'alpha')}
                         | dict(exists_now=path.is_file(), bytes_now=path.stat().st_size,
                                new_model_forward_or_backward_operations=0))
    result = dict(
        task='STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET',
        registration_id='STAGE2_B0_R2_CONTROLLED_REWARD_AND_BUDGET',
        start_utc=started, end_utc=datetime.now(UTC).isoformat(), audit_code_commit=commit,
        preregistration_commit='07e9e5a', actual_command=[sys.executable, *sys.argv],
        exit_code=0, protocol=protocol, reward100_vs200_same_behavior=rewards,
        selected_existing_detailed_traces=traces,
        r1_earliest_success_discounted_reward=successful, model_inventory=inventory,
        confirmed_new_critical_l1_defects=[], unresolved_substantive_protocol_conflicts=[],
        confirmed_critical_L1_defects=[], unresolved_scientific_conflicts=[],
        training_permitted=True,
        permission_scope='Only frozen R2 registration; all preceding actual audit checks completed',
        evidence_level='SUPPORTED_LOCAL_RANGE_AND_SAME_BEHAVIOR_REWARD_ARITHMETIC',
        limitation='未发现新关键错误限于此只读审核；不等于全域无bug或reward200必定有效。',
        missing_evidence=[
            'R1 Val0/1/2控制节点轨迹没有完整8D/Actor分布；不从旧V1冒造R1状态。',
            '没有保存的更早成功详细轨迹不重构；最早实际成功与最早保留成功分别登记。',
            '旧100k Replay实际采样频率/RNG不可由终点数量恢复。'],
        operation_counts=dict(new_environment_transitions=0, new_policy_replays=0,
                              new_scientific_training_steps=0, new_complete_sac_updates=0,
                              new_model_forward_operations=0, new_backward_operations=0))
    (TASK/'PRETRAIN_MODEL_REWARD_AUDIT.json').write_text(json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [
        '# R2运行前模型／奖励一致性审核', '',
        f'冻结登记07e9e5a后执行；本次审计代码{commit}。只读历史证据，不重放、不训练。', '',
        'LOCAL原生奖励表54（0-based）到达Nominal100，Validation区间**50–200**。',
        '100→200在原文区间；共同任务reward、最多两轮预注册修复、测试集不调仍适用。',
        'Stage2有限修复两轮，不因奖励数值核对通过而自动宣称科学GO。', '',
        '## 已有详细轨迹的10／5／3m捕获', '',
        '| 原组／seed／案例 | 最小距离m | 首次10m / 5m / 3m时间s | 终止 |',
        '| --- | ---: | --- | --- |']
    for row in traces:
        times = ' / '.join('null' if row['first_capture'][str(int(r))] is None else
                           f"{row['first_capture'][str(int(r))]['time_since_formal_start_s']:.3f}"
                           for r in THRESHOLDS)
        lines.append(f"| {row['group']}/{row['seed']}/{row['trajectory_id']} | "
                     f"{row['closest_goal_distance_m']:.4f} | {times} | {row['failure_type']} |")
    lines += ['', '详细状态、Body目标、yaw/pitch误差、名义/物理指令与实际响应在同名JSON。',
              '时间是已记录0.05s执行线性分段的首次球进入；不是重新仿真。',
              'V1补充有明确历史身份；R1 Val控制节点缺8D，不把V1状态冒充R1。', '',
              '## 同行为奖励100／200离线重计', '',
              f"完整episode最大独立残差{rewards['whole_episode_reward_residual_max_abs']:.3g}，",
              f'运行前固定float64核对容差{REWARD_TOLERANCE:g}。',
              '失败其他分量不变且差0，成功终点仅额外100；分母按训练／验证点／seed独立。',
              '成功episode全部transition与单个成功终点分别计数，见JSON，不推造Replay抽样频率。',
              '折扣以首个正式reward为gamma^0；进展与终点分别计，不与未折扣回报混用。',
              '原顺序最早实际成功与最早有保留节点的成功分开；缺失详细轨迹写null。',
              '奖励200的离线重计保持行为固定，不能证明它会改变学习或保证到达。', '',
              '## 模型与停止条件', '',
              '12个R1小模型现场存在；沿用上一轮真实weights_only身份/有限性审计，',
              '本轮不重新加载网络推断、不计算梯度；不能把历史审计称成本轮模型运算。',
              '此范围未发现新关键L1或实质协议冲突；并非全域无缺陷证明。',
              '本工具新环境步／重放／训练／前向／反向均0。',
              '训练授权和科学Gate由主任务按冻结R2登记处理，本工具不启动训练。']
    (TASK/'PRETRAIN_MODEL_REWARD_AUDIT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='READ_ONLY_PREFLIGHT_AUDIT_COMPLETE',
                          historical_detailed_traces_scanned=len(traces),
                          reward_max_abs_error=rewards['whole_episode_reward_residual_max_abs'],
                          new_training_steps=0, new_environment_transitions=0), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
