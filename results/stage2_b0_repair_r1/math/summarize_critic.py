"""从已保存冻结critic结果生成紧凑表；不重新运行策略或改变原结果。"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def main() -> int:
    """逐seed/固定初始动作，保留4次全部部分回报，不按最好分支筛选。"""
    output = Path(__file__).resolve().parent
    result = json.loads((output / 'critic_result.json').read_text(encoding='utf-8'))
    rows = []
    for source in result['sources']:
        seed = int(source['source_trajectory_id'].split('_')[1][1:])
        for action in source['actions']:
            rollouts = [row for row in result['rollouts']
                        if row['training_seed'] == seed and row['branch'] == action['branch']]
            if len(rollouts) != 4:
                raise ValueError('固定初始动作缺少登记的4个诊断随机流，不能替代缺失数据。')
            rows.append(dict(
                training_seed=seed, model_transition=300000,
                source_trajectory_id=source['source_trajectory_id'],
                snapshot_task_step=source['task_step'], branch=action['branch'],
                q1=action['q1'], q2=action['q2'], minimum_q=action['minimum_q'],
                frozen_alpha=source['alpha'], independent_diagnostic_streams=4,
                short_rollout_control_transitions=sum(
                    row['actual_transitions'] for row in rollouts),
                mean_goal_progress_m=statistics.mean(row['initial_goal_distance_m']
                                                     - row['final_goal_distance_m']
                                                     for row in rollouts),
                mean_partial_soft_return=statistics.mean(row['partial_soft_return']
                                                          for row in rollouts),
                sample_sd_partial_soft_return=statistics.stdev(row['partial_soft_return']
                                                               for row in rollouts),
                true_terminal_count=sum(row['terminated'] for row in rollouts),
                omitted_remaining_tail_count=sum(row['remaining_task_tail_omitted']
                                                  for row in rollouts),
                interpretation='FINITE_PARTIAL_RETURN_NOT_EXACT_Q_CALIBRATION'))
    with (output / 'critic_action_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), rollouts=result['trajectory_count'],
                          diagnostic_control_transitions=result['diagnostic_control_transitions']),
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
