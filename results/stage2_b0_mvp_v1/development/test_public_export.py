"""纯合成日志解析测试，不执行环境、网络或科研训练。"""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
from copy import deepcopy
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / 'export_public_evidence.py'
SPEC = importlib.util.spec_from_file_location('public_export', HELPER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CODE = 'SYNTHETIC_TEST_NOT_SCIENTIFIC_EVIDENCE'


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')


def fixture(root: Path) -> tuple[dict, dict]:
    state = dict(status='COMPLETED', registration_id=MODULE.REGISTRATION,
                 experiment_code_commit=CODE, registration={'synthetic_fixture_only': True},
                 completed_jobs=[dict(seed=seed, stop=stop) for seed in MODULE.SEEDS
                                 for stop in (100000, 300000)],
                 seeds={str(seed): dict(transitions=300000, updates=290001)
                        for seed in MODULE.SEEDS})
    analysis = dict(status='BATCH_DATA_COMPLETE', registration_id=MODULE.REGISTRATION,
                    experiment_code_commit=CODE, seed_summaries=[])
    for seed in MODULE.SEEDS:
        segment = dict(segment_id='synthetic', path=f'seed_{seed}/segments/synthetic',
                       valid_log_sequence=99, status='COMPLETED')
        write(root / f'seed_{seed}/segments.json', [segment])
        analysis['seed_summaries'].append(dict(
            training_seed=seed, status='COMPLETE', authoritative_logged_steps=300000,
            authoritative_logged_updates=290001, segment_diagnostics={'segments': [segment]}))
        common = dict(registration_id=MODULE.REGISTRATION, method=MODULE.METHOD,
                      run_kind='scientific_training', training_seed=seed,
                      code_version=CODE, segment_id='synthetic')
        item = dict(steps=1, complete=True, failure_type='goal_success',
                    task_profile='obstacle_free', physical_time_s=0.2, minimum_clearance_m=None,
                    action_saturation_count=0, reward=0.0, path_length_m=0.1,
                    reward_components=dict.fromkeys(MODULE.REWARD_FIELDS, 0.0),
                    initial_position_ned_m=[0.0, 0.0, 1.0], goal_position_ned_m=[1.0, 0.0, 1.0],
                    trajectory=[dict(task_step=1, elapsed_s=0.2, position_ned_m=[0.1, 0.0, 1.0],
                                     action=[0.0, 0.0, 0.0])])
        training = [dict(item, **common, log_sequence=1, steps=100000),
                    dict(item, **common, log_sequence=2, steps=200000, task_profile='cv_train_v1'),
                    dict(item, log_sequence=100, code_version='UNCONFIRMED_OTHER_IDENTITY')]
        points = []
        for sequence, (profile, step, full) in enumerate(sorted(MODULE.POINTS), 3):
            count = 300 if full else 30
            episodes = []
            for index in range(count):
                episode = dict(item, index=index, task_profile=profile,
                               base_scenario_id=f'synthetic-base-{index}',
                               scenario_id=f'synthetic-{profile}-{index}',
                               failure_type='collision' if index == 3 else 'goal_success')
                if index not in (0, 1, 2, 3):
                    episode.pop('trajectory')
                episodes.append(episode)
            points.append(dict(common, profile=profile, at_transition=step, full=full,
                               count=count, validation_root_seed=20261006,
                               training_state_unchanged=True, evaluation_key=f'{profile}:{step}',
                               log_sequence=sequence, episodes=episodes))
        directory = root / segment['path']
        directory.mkdir(parents=True)
        for kind, rows in [('episode', training), ('validation', points)]:
            (directory / (kind + '.jsonl')).write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    write(root / 'batch_state.json', state)
    write(root / 'analysis/batch_analysis.json', analysis)
    return state, analysis


def must_fail(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f'{label} did not fail')


def main() -> None:
    checks = []
    with tempfile.TemporaryDirectory(prefix='public-export-synthetic-',
                                     dir=Path(__file__).resolve().parent) as directory:
        root = Path(directory)
        state, analysis = fixture(root)
        report = MODULE.export(root)
        assert report['counts'] == dict(training_episodes=6, validation_episodes=2880,
                                        selected_trajectories=168, trajectory_nodes=336,
                                        validation_points=42)
        assert report['excluded_unconfirmed_rows'] == {'episode': 3}
        with (root / 'public_evidence/validation_episodes.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]['minimum_clearance_m'] == 'null'
        assert rows[0]['event_denominator_eligible'] == 'true'
        assert rows[0]['source_path'].startswith('seed_11/')
        checks += ['complete synthetic parse', 'confirmed cutoff excludes foreign tail',
                   'null and event denominator preserved', '42 points and all 2880 original rows',
                   'registered retained trajectories and initial nodes']
        failed_state = dict(state, status='RUNNING')
        write(root / 'batch_state.json', failed_state)
        must_fail(lambda: MODULE.export(root), 'running batch')
        write(root / 'batch_state.json', state)
        bad_analysis = deepcopy(analysis)
        bad_analysis['experiment_code_commit'] = 'DIFFERENT_CODE'
        write(root / 'analysis/batch_analysis.json', bad_analysis)
        must_fail(lambda: MODULE.export(root), 'analysis code identity')
        write(root / 'analysis/batch_analysis.json', analysis)
        bad_segments = deepcopy(analysis['seed_summaries'][0]['segment_diagnostics']['segments'])
        bad_segments[0]['valid_log_sequence'] = 98
        write(root / 'seed_11/segments.json', bad_segments)
        must_fail(lambda: MODULE.export(root), 'changed cutoff')
        write(root / 'seed_11/segments.json',
              analysis['seed_summaries'][0]['segment_diagnostics']['segments'])
        source = root / 'seed_11/segments/synthetic/episode.jsonl'
        original = source.read_text(encoding='utf-8')
        source.write_text(original.replace(CODE, 'WRONG_CODE', 1), encoding='utf-8')
        must_fail(lambda: MODULE.export(root), 'confirmed record identity')
        checks += ['RUNNING refuses final export', 'different analysis identity refuses',
                   'changed cutoff refuses', 'confirmed wrong record identity refuses']
    print(json.dumps(dict(status='PASS', fixture_kind='SYNTHETIC_LOG_ONLY', checks=checks,
                          environment_transitions=0, sac_updates=0, scientific_training_steps=0),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
