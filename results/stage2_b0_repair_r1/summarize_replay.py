"""只读冻结重放的因果层级证据；不启动环境、不重写原V1结果。"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
V1 = ROOT/'results'/'stage2_b0_mvp_v1'


def angular_errors(state: list[float], goal: list[float]) -> dict[str, float]:
    """在独立最小距离分段点重算航向/俯仰误差，不把控制节点近似当精确点。"""
    delta = [last-first for first, last in zip(state[:3], goal, strict=True)]
    desired_yaw = math.atan2(delta[1], delta[0])
    desired_pitch = -math.atan2(delta[2], math.hypot(delta[0], delta[1]))
    return dict(yaw_error_rad=math.atan2(math.sin(desired_yaw-state[3]),
                                       math.cos(desired_yaw-state[3])),
                pitch_error_rad=desired_pitch-state[4],
                surge_speed_mps=state[5], yaw_rate_rad_s=state[6], pitch_rate_rad_s=state[7])


def write_json(path: Path, value: Any) -> None:
    """只创建本轮新分析证据，不覆盖既有分析结论。"""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main() -> int:
    """分开核心诊断、成功对照与反事实；不混成导航表现分母。"""
    result = json.loads((TASK/'frozen_replay_result.json').read_text(encoding='utf-8'))
    cases = result['cases']
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        groups[row['selection']].append(row)
    summaries = {}
    rows = []
    for label, entries in groups.items():
        boundaries: Counter[str] = Counter()
        for row in entries:
            boundaries.update(row['boundary_subtypes'])
        summaries[label] = dict(
            count=len(entries), transitions=sum(row['transitions'] for row in entries),
            events=dict(Counter(row['failure_type'] for row in entries)),
            boundary_subtypes=dict(boundaries),
            independent_event_mismatches=sum(len(row['event_mismatches']) for row in entries),
            entered_goal_ball=sum(row['first_goal_entry'] is not None for row in entries))
    for row in cases:
        close = row['closest_approach']
        angles = angular_errors(close['state'], row['goal_position_ned_m'])
        rows.append(dict(
            trajectory_id=row['trajectory_id'], seed=row['training_seed'],
            model_transition=row['model_transition'], profile=row['task_profile'],
            selection=row['selection'], branch=row['branch'], transitions=row['transitions'],
            final_event=row['failure_type'], boundary_subtypes=';'.join(row['boundary_subtypes']),
            closest_goal_distance_m=close['distance_m'], closest_timestamp_s=close['timestamp_s'],
            first_goal_entry_timestamp_s=(row['first_goal_entry']['timestamp_s']
                                          if row['first_goal_entry'] else None),
            final_goal_distance_m=row['final_goal_distance_m'],
            reward=row['reward'], progress=row['reward_components']['progress'],
            goal_reward=row['reward_components']['goal'],
            time_reward=row['reward_components']['time'],
            smoothness_reward=row['reward_components']['smoothness'],
            discounted_progress=row['discounted_progress'],
            progress_telescoping_error=row['undiscounted_progress_telescoping_error'],
            **angles))
    with (TASK/'replay_failure_breakdown.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {row['trajectory_id']: row for row in cases}
    counterfactuals = []
    cf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in groups.get('FIXED_COUNTERFACTUAL', []):
        cf_groups[row['source_trajectory_id']].append(row)
    for source, branches in cf_groups.items():
        original = index[source]
        cf_original = next(row for row in branches if row['branch'] == 'original')
        snapshot_step = cf_original['snapshot_task_step']
        exact = (original['transitions']-snapshot_step == cf_original['transitions']
                 and original['failure_type'] == cf_original['failure_type']
                 and original['final_state'] == cf_original['final_state'])
        counterfactuals.append(dict(
            source=source, snapshot_step=snapshot_step,
            snapshot_selection=cf_original['snapshot_selection'],
            snapshot_initial_states_identical=all(
                row['snapshot_initial_state'] == cf_original['snapshot_initial_state']
                for row in branches), original_continuation_matches_source_exactly=exact,
            outcomes=[dict(branch=row['branch'], event=row['failure_type'],
                           boundary_subtypes=row['boundary_subtypes'],
                           transitions=row['transitions'],
                           minimum_goal_distance_m=row['closest_approach']['distance_m'],
                           reward=row['reward']) for row in branches],
            claim='接管分支仅为冻结案例机制诊断，不计学习策略成功'))
    # 原固定18例的实际状态/事件与V1同一模型结果直接比较，非文件摘要。
    old = json.loads((V1/'learned_fixed_diagnostics'/'learned_fixed_diagnostics.json').read_text(
        encoding='utf-8'))
    fixed_comparisons = []
    for previous in old['cases']:
        source = next(row for row in groups['original_fixed']
                      if row['training_seed'] == previous['training_seed']
                      and row['model_transition'] == previous['model_transition']
                      and row['scenario_id'] == previous['case_id'])
        checks = dict(
            transitions_equal=source['transitions'] == previous['diagnostic_env_transitions'],
            event_equal=source['failure_type'] == previous['failure_type'],
            final_goal_distance_abs_error=abs(source['final_goal_distance_m']
                                               -previous['final_goal_distance_m']),
            reward_abs_error=abs(source['reward']-previous['reward']))
        fixed_comparisons.append(dict(trajectory_id=source['trajectory_id'], **checks))
    # V1公开CSV中同一终点的指标来自已确认原日志；新交叉profile诊断不伪称原结果。
    with (V1/'public_evidence'/'validation_episodes.csv').open(encoding='utf-8') as stream:
        fields = list(csv.DictReader(stream))
    validation_comparisons = []
    for source in groups['VAL_0_1_2']+groups.get('FIRST_SUCCESS_BY_INDEX', []):
        wanted_step = 100000 if source['task_profile'] == 'obstacle_free' else 300000
        if source['model_transition'] != wanted_step:
            continue
        prior = next(row for row in fields
                     if int(row['training_seed']) == source['training_seed']
                     and row['task_profile'] == source['task_profile']
                     and int(row['at_transition']) == wanted_step
                     and int(row['index']) == source['scenario_index'])
        validation_comparisons.append(dict(
            trajectory_id=source['trajectory_id'],
            event_equal=source['failure_type'] == prior['failure_type'],
            transitions_equal=source['transitions'] == int(prior['steps']),
            reward_abs_error=abs(source['reward']-float(prior['reward'])),
            final_goal_distance_abs_error=abs(
                source['final_goal_distance_m']-float(prior['final_distance_m']))))
    evidence = dict(
        registration_id='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
        groups=summaries, fixed_v1_direct_comparisons=fixed_comparisons,
        validation_v1_direct_comparisons=validation_comparisons,
        frozen_counterfactuals=counterfactuals,
        max_observation_error=max(row['independent_observation_max_abs_error'] for row in cases),
        max_command_error=max(row['independent_command_max_abs_error'] for row in cases),
        max_reward_reference_error=max(row['reward_component_max_abs_error'] for row in cases),
        max_progress_telescoping_error=max(
            row['undiscounted_progress_telescoping_error'] for row in cases),
        independent_segment_count=sum(row['executed_segment_count'] for row in cases),
        independent_event_mismatch_count=sum(len(row['event_mismatches']) for row in cases),
        missed_goal_count=sum(not row['goal_event_consistent'] for row in cases),
        actor_parameters_unchanged=all(row['actor_parameters_unchanged'] for row in cases),
        evidence_grade='SUPPORTED_MECHANISM_WITHIN_REGISTERED_CASES',
        scientific_training_updates=0, training_replay_writes=0,
        limitations=[
            '有限冻结案例不是全体Val/训练episode细分边界的统计代表。',
            '成功/失败接管只支持该快照因果诊断，不是新导航方法结果。',
            '50k/75k权重和100kcritic缺失，不能给其Q精确校准结论。',
            '无L1在此处表示已核对范围未发现，不证明任意潜在实现均无误。'])
    write_json(TASK/'replay_mechanism_evidence.json', evidence)
    print(json.dumps(dict(status='READ_ONLY_REPLAY_SUMMARY_COMPLETE',
                          trajectories=len(cases), groups=summaries,
                          counterfactual_continuity_all_exact=all(
                              row['original_continuation_matches_source_exactly']
                              for row in counterfactuals)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
