"""合成原始日志验证MVP分析分母、确认前缀、身份与可比曲线；不运行环境或梯度。"""

from __future__ import annotations

import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from auv_risk_rl.training.mvp_analysis import (
    METHOD,
    REGISTRATION_ID,
    EpisodeAccumulator,
    Moments,
    _aggregate_points,
    analyze_batch,
    confirmed_records,
    summarize_episodes,
    validate_actor_model,
    validation_point,
    write_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('b0_mvp_analysis_cli',
                                              ROOT / 'scripts' / 'analyze_b0_mvp.py')
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def episode(event: str = 'goal_success', **changes: Any) -> dict[str, Any]:
    """解析可计数夹具；真实task horizon即使truncated也按实际事件分类。"""
    value = dict(steps=10, complete=True, failure_type=event,
                 reward=3.0, reward_components={'progress': 1.0}, physical_time_s=2.0,
                 path_length_m=1.0, minimum_clearance_m=None, action_saturation_count=2,
                 progress_m=0.5, index=0)
    value.update(changes)
    return value


def record(sequence: int, **changes: Any) -> dict[str, Any]:
    """权威日志身份；纯合成，不将夹具冒充实际训练transition。"""
    value = dict(registration_id=REGISTRATION_ID, run_kind='scientific_training',
                 method=METHOD, training_seed=11, segment_id='segment_0001',
                 code_version='synthetic-code', log_sequence=sequence)
    value.update(changes)
    return value


def segment_fixture(root: Path, rows: dict[str, list[dict[str, Any]]],
                    cutoff: int | None = 2) -> None:
    """仅生成小型文本日志，不建立物理世界、Replay、模型或虚构计数器。"""
    directory = root / 'seed_11' / 'segments' / 'segment_0001'
    directory.mkdir(parents=True)
    segments = [dict(segment_id='segment_0001', path=directory.relative_to(root).as_posix(),
                     parent_checkpoint=None, valid_log_sequence=cutoff, status='PAUSED')]
    (root / 'seed_11' / 'segments.json').write_text(json.dumps(segments), encoding='utf-8')
    for kind, records in rows.items():
        (directory / f'{kind}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')


def validation(full: bool = False, seed: int = 11, reward: float = 3.0,
               ) -> dict[str, Any]:
    """固定base索引集合；所有期望在运行前由解析夹具给定。"""
    count = 300 if full else 30
    return record(1, profile='obstacle_free', training_seed=seed, count=count, full=full,
                  validation_root_seed=20261006, training_state_unchanged=True,
                  at_transition=100000 if full else 0,
                  evaluation_key=f'obstacle_free:{100000 if full else 0}:fixture',
                  episodes=[episode(index=index, reward=reward) for index in range(count)])


def model(**changes: Any) -> dict[str, Any]:
    """只测试模型身份，不创建torch.Tensor或调用torch.load。"""
    value = dict(format='b0-mvp-actor-v1', registration_id=REGISTRATION_ID,
                 method=METHOD, run_kind='scientific_training', seed=11, transition=100000,
                 code_version='synthetic-code', actor={'fixture': 'not a real tensor'},
                 sac_config=dict(gamma=.999, tau=.005, learning_rate=3e-4, batch_size=256,
                                 replay_capacity=500000, learning_starts=10000, utd=1,
                                 initial_alpha=.2, target_entropy=-3.0, device='cuda'))
    value.update(changes)
    return value


def test_sample_standard_deviation_and_unmeasured_null() -> None:
    """n−1标准差有独立手算依据，N=1不伪造零SD。"""
    moments = Moments()
    assert moments.result() == dict(n=0, mean=None, sample_sd=None)
    moments.add(1)
    assert moments.result() == dict(n=1, mean=1.0, sample_sd=None)
    moments.add(3)
    assert moments.result()['mean'] == 2.0
    assert moments.result()['sample_sd'] == pytest.approx(math.sqrt(2), abs=1e-12)


def test_event_denominator_keeps_all_physical_events_excludes_fragments() -> None:
    """四种物理完整终止共享分母4；工程/课程/预算三片段均不得算失败或成功。"""
    rows = [episode(event) for event in
            ('goal_success', 'collision', 'operational_boundary_failure', 'task_horizon')]
    rows.extend([episode('external_truncation', complete=False),
                 episode('none', complete=False, phase_boundary=True),
                 episode('none', complete=False, budget_stop=True)])
    result = summarize_episodes(rows)
    assert result['complete_physical_episodes'] == 4
    assert result['logged_training_steps'] == 70
    assert result['fragments'] == 3
    assert result['external_truncations'] == 1
    assert result['phase_boundary_fragments'] == 1
    assert result['budget_stop_fragments'] == 1
    assert [result[key] for key in ('success_rate', 'collision_rate', 'boundary_rate',
                                    'timeout_rate')] == [.25, .25, .25, .25]
    assert result['moments']['penalized_time_s']['mean'] == 150.5
    assert result['moments']['successful_path_length_m']['n'] == 1
    assert result['action_saturation_transition_rate'] == .2


def test_true_task_horizon_is_physical_not_external_cutoff() -> None:
    """历史接口同时标truncated时，task_horizon仍为物理timeout；不用布尔误分类。"""
    result = summarize_episodes([episode('task_horizon', terminated=True, truncated=True)])
    assert result['timeout_rate'] == 1.0
    assert result['external_truncations'] == 0
    assert result['moments']['penalized_time_s']['mean'] == 200.0


def test_no_complete_episode_and_no_obstacle_are_not_zero_metrics() -> None:
    """无episode和无障碍间距保留不适用标识。"""
    empty = summarize_episodes([episode('none', complete=False, budget_stop=True)])
    assert empty['success_rate'] is None
    assert empty['moments']['reward']['mean'] is None
    actual = summarize_episodes([episode()])
    assert actual['moments']['minimum_clearance_m']['mean'] is None
    assert actual['near_miss_rate_on_clearance_applicable'] is None


def test_near_miss_distinguishes_collision_and_zero_point_five_boundary() -> None:
    """无碰撞且净间距严格小于.5m；碰撞和等于.5m不计near miss。"""
    result = summarize_episodes([episode(minimum_clearance_m=.49),
                                 episode(minimum_clearance_m=.5),
                                 episode('collision', minimum_clearance_m=-.1)])
    assert result['near_miss_count'] == 1
    assert result['near_miss_rate_on_clearance_applicable'] == 1/3


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_raw_result_is_rejected(value: float) -> None:
    """损坏科学标量不得用裁剪、零值或重试隐藏。"""
    with pytest.raises(ValueError, match='非有限'):
        summarize_episodes([episode(reward=value)])


def test_inconsistent_fragment_physical_event_is_rejected() -> None:
    """课程/预算片段不能伪装物理失败。"""
    with pytest.raises(ValueError, match='物理事件'):
        EpisodeAccumulator().add(episode('collision', complete=False))


def test_checkpoint_cutoff_excludes_preserved_abandoned_tail(tmp_path: Path) -> None:
    """旧尾部仍在文件中，但只有安全checkpoint确认的前缀进入权威统计。"""
    segment_fixture(tmp_path, {'episode': [record(1), record(3)]}, cutoff=1)
    diagnostics: dict[str, Any] = {}
    rows = list(confirmed_records(tmp_path, 11, 'synthetic-code', diagnostics))
    assert [row['log_sequence'] for _, row in rows] == [1]
    assert diagnostics['excluded_unconfirmed_rows']['episode'] == 1
    assert (tmp_path / 'seed_11/segments/segment_0001/episode.jsonl').read_text().count('\n') == 2


def test_null_checkpoint_cutoff_does_not_invent_confirmed_results(tmp_path: Path) -> None:
    """尚无安全保存的segment全部为provisional，不冒充已确认成果。"""
    segment_fixture(tmp_path, {'episode': [record(1)]}, cutoff=None)
    assert list(confirmed_records(tmp_path, 11, 'synthetic-code', {})) == []


def test_duplicate_authoritative_sequence_is_rejected(tmp_path: Path) -> None:
    """恢复后权威序号不能重复计数。"""
    segment_fixture(tmp_path, {'episode': [record(1)], 'update': [record(1)]})
    with pytest.raises(ValueError, match='重复'):
        list(confirmed_records(tmp_path, 11, 'synthetic-code', {}))


@pytest.mark.parametrize('field,value', [('run_kind', 'engineering_smoke'),
                                        ('method', 'FULL_CONSTRAINED_SAC'),
                                        ('code_version', 'other-code')])
def test_other_run_identity_is_rejected(tmp_path: Path, field: str, value: str) -> None:
    """工程日志和其他代码/方法不能混入本批科学结果。"""
    segment_fixture(tmp_path, {'episode': [record(1, **{field: value})]})
    with pytest.raises(ValueError, match='身份'):
        list(confirmed_records(tmp_path, 11, 'synthetic-code', {}))


def test_partial_batch_is_not_complete_or_scientific_go(tmp_path: Path) -> None:
    """没有真实日程/原始结果时输出NOT_COMPLETE和缺项，不根据历史PASS补造结果。"""
    state = dict(registration_id=REGISTRATION_ID, experiment_code_commit='synthetic-code',
                 status='RUNNING', completed_jobs=[], seeds={})
    (tmp_path / 'batch_state.json').write_text(json.dumps(state), encoding='utf-8')
    analysis = analyze_batch(tmp_path)
    assert analysis['status'] == 'NOT_COMPLETE'
    assert analysis['stage2_scientific_decision'] == 'REQUIRES_REVIEW_OF_RAW_SEED_EVIDENCE'
    assert len(analysis['seed_summaries']) == 3
    assert len(analysis['seed_summaries'][0]['missing_registered_validations']) == 14
    assert analysis['validation_points'] == []
    write_analysis(tmp_path / 'analysis', analysis)
    CLI.write_report(analysis, tmp_path / 'analysis')
    assert 'NOT_COMPLETE' in (tmp_path / 'analysis/ANALYSIS_REPORT.md').read_text()


def test_val300_preserved_with_same_monitor30_subset() -> None:
    """Val300全部episode保留；曲线只复用相同index0..29，不混淆分母。"""
    row = validation(full=True)
    for item in row['episodes'][30:]:
        item.update(failure_type='collision', reward=-9)
    result = validation_point(row)
    assert result['all_registered_episodes']['complete_physical_episodes'] == 300
    assert result['all_registered_episodes']['success_rate'] == .1
    assert result['paired_monitor30']['complete_physical_episodes'] == 30
    assert result['paired_monitor30']['success_rate'] == 1
    assert result['paired_monitor30']['moments']['reward']['mean'] == 3


def test_missing_fixed_index_and_incomplete_validation_are_rejected() -> None:
    """不得重抽、少保留失败场景或把截断算成正式完整评估。"""
    row = validation()
    row['episodes'][1]['index'] = 0
    with pytest.raises(ValueError, match='完整覆盖'):
        validation_point(row)
    row = validation()
    row['episodes'][0]['complete'] = False
    with pytest.raises(ValueError, match='未完成'):
        validation_point(row)


def test_cross_seed_mean_sd_uses_seed_results_not_pooled_episodes() -> None:
    """三seed均值2、样本SD1有独立解析依据；不是90个独立训练seed。"""
    points = [validation_point(validation(seed=seed, reward=reward))
              for seed, reward in zip((11, 22, 33), (1., 2., 3.), strict=True)]
    result = _aggregate_points(points)[0]
    assert result['independent_training_seed_count'] == 3
    assert result['metrics']['reward'] == dict(n=3, mean=2.0, sample_sd=1.0)
    with pytest.raises(ValueError, match='重复验证'):
        _aggregate_points(points + [deepcopy(points[0])])


@pytest.mark.parametrize('field,value', [('run_kind', 'engineering_smoke'),
                                        ('method', 'COMPLETE_LOCAL_CONSTRAINED_SAC'),
                                        ('seed', 22), ('transition', 300000),
                                        ('code_version', 'other')])
def test_learned_model_identity_boundary(field: str, value: Any) -> None:
    """错误模型身份在任何环境运行前被拒绝。"""
    with pytest.raises(ValueError, match='身份'):
        validate_actor_model(model(**{field: value}), 11, 100000, 'synthetic-code')


def test_model_cuda_contract_and_production_parameters() -> None:
    """不静默CPU回退，不接受降低batch/起步数的工程模型。"""
    payload = model()
    assert validate_actor_model(payload, 11, 100000, 'synthetic-code') == 'cuda'
    payload['sac_config']['batch_size'] = 4
    with pytest.raises(ValueError, match='生产配置'):
        validate_actor_model(payload, 11, 100000, 'synthetic-code')
    payload = model()
    payload['sac_config']['device'] = 'cpu'
    with pytest.raises(ValueError, match='设备回退'):
        validate_actor_model(payload, 11, 100000, 'synthetic-code')


def test_report_cli_default_does_not_execute_environment_or_learning() -> None:
    """默认纯分析；学习策略固定诊断需要独立显式选项和可信模型声明。"""
    args = CLI.parser().parse_args([])
    assert not args.learned_diagnostics
    assert not args.trusted_local_models
    assert not args.plot_only
    assert args.batch_root == ROOT / 'results' / 'stage2_b0_mvp_v1'
    with pytest.raises(ValueError, match='plot-only'):
        CLI.main(['--plot-only'])


def test_raw_trajectory_ned_and_event_prefix_time_are_preserved() -> None:
    """深度Down正方向与不足.2s末段时间直接使用原始证据，不猜测或平滑。"""
    value = dict(initial_position_ned_m=[10., 20., 8.], trajectory=[
        dict(elapsed_s=.2, position_ned_m=[10.1, 20.2, 8.3]),
        dict(elapsed_s=.05, position_ned_m=[10.15, 20.21, 8.4]),
    ])
    series = CLI.trajectory_series(value)
    assert series == dict(time_s=[0., .2, .25], north_m=[10., 10.1, 10.15],
                          east_m=[20., 20.2, 20.21], down_m=[8., 8.3, 8.4])
    value['trajectory'][0]['elapsed_s'] = 0
    with pytest.raises(ValueError, match='实际正执行时间'):
        CLI.trajectory_series(value)


def test_trajectory_references_do_not_duplicate_all_raw_points(tmp_path: Path) -> None:
    """报告只保存checkpoint确认行引用，绘图时读真实完整节点。"""
    row = validation()
    row['episodes'][0].update(initial_position_ned_m=[10., 20., 8.],
                               goal_position_ned_m=[20., 20., 8.],
                               trajectory=[dict(elapsed_s=.2,
                                                position_ned_m=[10.1, 20., 8.])])
    segment_fixture(tmp_path, {'validation': [row]}, cutoff=1)
    confirmed = list(confirmed_records(tmp_path, 11, 'synthetic-code', {}))[0][1]
    point = validation_point(confirmed)
    assert point['retained_trajectory_indices'] == [0]
    assert 'trajectory' not in point
    analysis = dict(batch_root=str(tmp_path), registration_id=REGISTRATION_ID,
                    experiment_code_commit='synthetic-code', validation_points=[point])
    actual = CLI._retained_validation_rows(analysis)
    assert len(actual) == 1
    assert actual[0]['episodes'][0]['trajectory'] == row['episodes'][0]['trajectory']
