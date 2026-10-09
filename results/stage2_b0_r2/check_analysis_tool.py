"""结果工具的小字典/临时JSONL自检；不创建环境、模型或任何科研训练记录。"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import analyze_r2 as analysis


def episode(*, success: bool, weight: float) -> dict:
    """明确标记的合成结构夹具，不能被视为真实环境episode。"""
    parts = dict(progress=1.0, goal=weight if success else 0.0, time=-0.01, smoothness=0.0)
    return dict(
        training_seed=11, task_profile='obstacle_free', split='SYNTHETIC_TOOL_FIXTURE_ONLY',
        scenario_id='synthetic-tool-fixture', scenario_index=0, scenario_root_seed=0,
        episode_id=0, env_slot=0, steps=1, complete=True,
        failure_type='goal_success' if success else 'operational_boundary_failure',
        success=success, action_saturation_count=0, physical_time_s=0.2, path_length_m=1.0,
        minimum_clearance_m=None, reward_components=parts, reward=sum(parts.values()),
        first_goal_entry_time_s=0.2 if success else None, first_within_10m_time_s=0.0,
        minimum_goal_distance_m=2.0 if success else 15.0,
        boundary_subtype=None if success else 'pitch_lower', start_transition=1,
        last_transition=1, at_transition=1, task_config=dict(
            w_progress=1.0, w_goal=weight, w_time=0.01, w_smooth=0.02, d_C=0.05))


class AnalysisToolChecks(unittest.TestCase):
    """仅验证分析分母、单位/共同效用、有效前缀和完成门禁。"""

    def test_denominator_utility_and_sample_sd(self) -> None:
        """未完整片段排除；goal200只改变原效用，共同goal100可比。"""
        fragment = dict(steps=7, complete=False, failure_type='none', budget_stop=True)
        c_rows = [episode(success=True, weight=100.0), episode(success=False, weight=100.0)]
        g_rows = [episode(success=True, weight=200.0), episode(success=False, weight=200.0)]
        c = analysis.episode_stats([*c_rows, fragment], 100.0)
        g = analysis.episode_stats([*g_rows, fragment], 200.0)
        self.assertEqual(c['complete_physical_episodes'], 2)
        self.assertEqual(c['fragments'], 1)
        self.assertEqual(c['success_rate'], 0.5)
        self.assertEqual(c['boundary_subtypes'], {'pitch_lower': 1})
        self.assertAlmostEqual(c['common100utility']['mean'], g['common100utility']['mean'])
        self.assertAlmostEqual(analysis.moment([1.0, 3.0])['sample_sd'], math.sqrt(2.0))
        self.assertEqual(analysis.moment([]), dict(n=0, mean=None, sample_sd=None))
        self.assertEqual(len(analysis.expected_points()), 15)

    def test_running_batch_is_never_complete(self) -> None:
        """运行中状态只允许partial；不依赖任何真实PID或日志。"""
        state = dict(status='RUNNING')
        self.assertEqual(analysis.completion_gate(Path('.'), state, partial=True)['status'],
                         'PARTIAL_ONLY')
        with self.assertRaises(ValueError):
            analysis.completion_gate(Path('.'), state, partial=False)

    def test_validation_index_csv_bridge(self) -> None:
        """验证权威index必须显式进入紧凑CSV，不依赖重复scenario_index字段。"""
        parent = analysis.TASK/'development'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='analysis_index_bridge_', dir=parent) as temporary:
            root = Path(temporary).resolve()
            root.relative_to(analysis.TASK.resolve())
            validation = dict(at_transition=300000, evaluation_key='obstacle_free:300000:full',
                              full=True, registration_id=analysis.REGISTRATION_ID,
                              code_version='0'*40, run_kind='scientific_training',
                              method=analysis.METHOD, segment_id='fixture', log_sequence=1,
                              _log_source=dict(path='fixture.jsonl', line_number=1,
                                               valid_log_sequence=1))
            writer = analysis.RowWriter(root/'all_episodes.csv.gz', analysis.EPISODE_FIELDS,
                                        compressed=True)
            try:
                for index in range(300):
                    item = episode(success=True, weight=100.0)
                    item.pop('scenario_index')
                    item['index'] = index
                    writer.add(analysis.public_episode(item, 'C300', validation))
            finally:
                writer.close()
            selected = analysis.compact_endpoint_episodes(root)
            self.assertEqual(len(selected['C300', 11]), 300)
            self.assertEqual({int(row['scenario_index']) for row in selected['C300', 11]},
                             set(range(300)))

    def test_actual_schema_and_confirmed_prefix(self) -> None:
        """按真实结构读取临时日志，特别覆盖update无task_profile与未确认尾部。"""
        parent = analysis.TASK/'development'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='analysis_tool_', dir=parent) as temporary:
            root = Path(temporary).resolve()
            root.relative_to(analysis.TASK.resolve())
            segment = root/'C300'/'seed_11'/'segments'/'segment_0001'
            segment.mkdir(parents=True)
            inventory = [dict(segment_id='segment_0001',
                              path='seed_11/segments/segment_0001',
                              valid_log_sequence=5, status='RUNNING')]
            (segment.parents[1]/'segments.json').write_text(json.dumps(inventory), encoding='utf-8')
            common = dict(registration_id=analysis.REGISTRATION_ID, run_kind='scientific_training',
                          method=analysis.METHOD, training_seed=11, segment_id='segment_0001',
                          code_version='0'*40, group='C300', reward_config=episode(
                              success=True, weight=100.0)['task_config'])
            training = dict(episode(success=True, weight=100.0), **common, log_sequence=2)
            (segment/'episode.jsonl').write_text(json.dumps(training)+'\n', encoding='utf-8')
            updates = [dict(common, log_sequence=number+3, transition=10000+number,
                            metrics={name: 0.2 if name == 'alpha' else 1.0
                                     for name in analysis.UPDATE_FIELDS}) for number in range(3)]
            updates.append(dict(updates[-1], log_sequence=6, group='UNCONFIRMED_WRONG_GROUP'))
            (segment/'update.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in updates),
                                               encoding='utf-8')
            episodes = []
            for index in range(30):
                row = copy.deepcopy(episode(success=True, weight=100.0))
                row.update(index=index, scenario_index=index, scenario_id=f'fixture-{index}',
                           base_scenario_id=f'base-{index}', actual_scenario_id=f'fixture-{index}',
                           environment_seed=index, initial_position_ned_m=[0.0, 0.0, 0.0],
                           goal_position_ned_m=[3.0, 0.0, 0.0])
                if index < 3:
                    row.update(trajectory_retention_reasons=['preregistered_index'],
                               formal_initial_state=dict(position_ned_m=[0.0, 0.0, 0.0],
                                                         physical_time_s=0.0),
                               trajectory=[dict(task_step=1, position_ned_m=[1.0, 0.0, 0.0],
                                                physical_time_s=0.2, action=[0.0, 0.0, 0.0],
                                                reward=row['reward'],
                                                reward_components=row['reward_components'])])
                episodes.append(row)
            validation = dict(common, log_sequence=1, task_profile='obstacle_free',
                              profile='obstacle_free', at_transition=0, full=False, count=30,
                              episodes=episodes, validation_root_seed=20261006,
                              evaluation_key='obstacle_free:0:monitor', indices=list(range(30)),
                              training_state_unchanged=True, r2_diagnostic_state_unchanged=True,
                              evaluation_env_transitions=30, warmup_control_transitions=150)
            (segment/'validation.jsonl').write_text(json.dumps(validation)+'\n', encoding='utf-8')
            writers = dict(episodes=analysis.RowWriter(root/'episodes.csv',
                                                       analysis.EPISODE_FIELDS),
                           trajectories=analysis.RowWriter(root/'trajectories.csv',
                                                            analysis.TRAJECTORY_FIELDS))
            try:
                result = analysis.collect_run(root, 'C300', 11, '0'*40, partial=True,
                                               writers=writers, canonical={})
                self.assertEqual(result['confirmed_complete_sac_updates'], 3)
                self.assertEqual(result['confirmed_optimizer_steps'], 12)
                self.assertEqual(result['observed_validation_points'], 1)
                self.assertEqual(len(result['missing_validation_points']), 14)
                self.assertEqual(writers['episodes'].count, 31)
                self.assertEqual(writers['trajectories'].count, 6)
                self.assertEqual(result['segment_diagnostics']['excluded_unconfirmed_rows'],
                                 {'update': 1})
            finally:
                for writer in writers.values():
                    writer.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
