"""从真实小回执生成R2公开清单；不执行Git、测试、训练或文件摘要。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[1]
FINAL = {
    'final_pytest_all': 'pytest_all.xml',
    'final_phase_a': 'pytest_phase_a.xml',
    'final_ruff': None,
    'final_compileall': None,
}
COMMANDS = {*FINAL, 'r2_initial_scoped_ruff', 'r2_preflight_cli',
            'pretrain_code_push', 'pretrain_remote_ref', 'publication_helper_ruff',
            'publication_helper_ruff_final', 'publication_prepare', 'publication_prepare_final',
            'r2_scientific_batch', 'r2_result_analysis',
            'r2_final_evidence_audit'}


def read_json(path: Path) -> Any:
    """缺少回执为None，不能用计划数字替代实测。"""
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def write_json(path: Path, value: Any) -> None:
    """本helper只写当前results中的公开派生小结果，保留原始记录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8')


def public_text(value: str) -> str:
    """公开文本只折叠本项目机器路径；不改变原始命令/worker记录。"""
    return value.replace(str(ROOT), '<PROJECT_ROOT>').replace(ROOT.as_posix(), '<PROJECT_ROOT>')


def junit_counts(path: Path) -> dict[str, int] | None:
    """按真实testcase计数，错误/跳过不得当作通过。"""
    if not path.is_file():
        return None
    cases = list(ET.parse(path).getroot().iter('testcase'))
    failed = sum(case.find('failure') is not None for case in cases)
    errors = sum(case.find('error') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    return dict(total=len(cases), passed=len(cases)-failed-errors-skipped,
                failed=failed, errors=errors, skipped=skipped)


def test_summary() -> dict[str, Any]:
    """只有已退出worker和相应真实JUnit可确认最终测试完成。"""
    checks = {}
    for name, xml_name in FINAL.items():
        worker = read_json(TASK / f'{name}.worker.json')
        counts = junit_counts(TASK / xml_name) if xml_name else None
        if worker is None:
            status = 'NOT_RUN'
        elif worker.get('status') != 'EXITED':
            status = worker.get('status', 'UNKNOWN')
        elif worker.get('direct_child_exit_code') != 0:
            status = 'FAIL'
        elif xml_name and (counts is None or counts['failed'] or counts['errors']):
            status = 'INVALID_OR_FAILED_JUNIT'
        else:
            status = 'PASS'
        checks[name] = dict(status=status, worker_receipt=f'{name}.worker.json',
                            exit_code=worker.get('direct_child_exit_code') if worker else None,
                            code_commit=worker.get('code_commit') if worker else None,
                            start_utc=worker.get('start_utc') if worker else None,
                            end_utc=worker.get('end_utc') if worker else None,
                            junit_file=xml_name, junit_counts=counts)
    return dict(generated_at_utc=datetime.now(UTC).isoformat(), checks=checks,
                all_final_checks_pass=all(row['status'] == 'PASS' for row in checks.values()),
                observation_counters_file='pytest_all_counts.json',
                counter_note='Nested env/World calls must not be added; optimizer unit tests '
                             'are not scientific navigation training.',
                development_missing_checkpoint_reproducer=dict(
                    before=junit_counts(TASK/'development/review_r2_missing_checkpoint_red.xml'),
                    after=junit_counts(TASK/'development/review_r2_green.xml'),
                    scope='No scientific training; mock recovery and actual paired initialization'))


def environment_summary() -> dict[str, Any]:
    """公开必要软硬件/工作盘余量，不复制完整进程列表或其他磁盘用量。"""
    identity = read_json(TASK/'environment_identity.json') or {}
    resources = read_json(TASK/'resources_preflight.json') or {}
    return dict(observed_at_utc=resources.get('observed_at_utc'),
                python=identity.get('python'), pytorch=identity.get('pytorch'),
                cuda=identity.get('cuda_build'), gpu=identity.get('gpu'),
                platform=identity.get('platform'), cpu=resources.get('cpu_model'),
                logical_cores=resources.get('logical_cores'), ram=resources.get('ram'),
                gpu_observation=resources.get('nvidia_smi'),
                workspace_disk=resources.get('disks', {}).get('D:/'),
                replay_schema=resources.get('replay_schema'),
                storage_estimate=resources.get('six_run_storage_estimate'),
                pip_check_exit_code=identity.get('pip_check_exit_code'),
                existing_environment_reused=identity.get('existing_environment_reused'),
                cuda_probe=identity.get('cuda_probe'),
                cuda_probe_result=identity.get('cuda_smoke_result'),
                cuda_probe_synchronized=identity.get('cuda_synchronized'),
                caveat='Resource observations and storage estimates, not throughput measurement.')


def pretrain_result(summary: dict[str, Any]) -> dict[str, Any]:
    """训练前检查与已启动批次分开，绝不把运行中科研结果写成PASS。"""
    audit = read_json(TASK/'PRETRAIN_MODEL_REWARD_AUDIT.json') or {}
    push = read_json(TASK/'pretrain_code_push.worker.json') or {}
    remote = read_json(TASK/'pretrain_remote_ref.worker.json') or {}
    batch = read_json(TASK/'batch_state.json') or {}
    launcher = read_json(TASK/'launcher_identity.json') or {}
    worker = read_json(TASK/'r2_scientific_batch.worker.json') or {}
    code_commits = {row['code_commit'] for row in summary['checks'].values()}
    commit = next(iter(code_commits)) if len(code_commits) == 1 else None
    ref_file = TASK/'pretrain_remote_ref.stdout.txt'
    refs = [line.split() for line in ref_file.read_text(encoding='utf-8').splitlines()
            if line.strip()] if ref_file.is_file() else []
    published = (commit is not None and push.get('direct_child_exit_code') == 0
                 and remote.get('direct_child_exit_code') == 0
                 and [commit, 'refs/heads/codex/stage2-b0-r2'] in refs)
    initial_ruff = read_json(TASK/'r2_initial_scoped_ruff.worker.json') or {}
    failure_history = [dict(
        name='initial_scoped_ruff', exit_code=initial_ruff.get('direct_child_exit_code'),
        record='r2_initial_scoped_ruff.worker.json', raw_output='r2_initial_scoped_ruff.stdout.txt',
        resolution='Manual import/long-line corrections; final Ruff independently rerun.')]
    for name in ('review_r2_missing_checkpoint_red', 'reward_tests_initial'):
        counts = junit_counts(TASK/f'development/{name}.xml')
        if counts is not None:
            failure_history.append(dict(name=name, junit_counts=counts,
                                        local_record=f'development/{name}.xml',
                                        note='Historical development result retained unchanged; '
                                             'not counted as a passing final test.'))
    preflight_ok = (audit.get('confirmed_critical_L1_defects') == []
                    and audit.get('unresolved_scientific_conflicts') == []
                    and audit.get('training_permitted') is True)
    return dict(
        generated_at_utc=datetime.now(UTC).isoformat(),
        status=('PRETRAIN_ENGINEERING_CHECKS_PASS'
                if summary['all_final_checks_pass'] and published and preflight_ok
                else 'PRETRAIN_CHECKS_INCOMPLETE_OR_BLOCKED'),
        tested_code_commit=commit, branch='codex/stage2-b0-r2',
        github_repository='https://github.com/Whsjbwy/FL-RL',
        github_code_commit=(f'https://github.com/Whsjbwy/FL-RL/commit/{commit}'
                            if commit else None),
        code_push_confirmed=published, remote_refs=refs,
        test_results=summary['checks'],
        audit=dict(path='PRETRAIN_MODEL_REWARD_AUDIT.json',
                   actual_code_commit=audit.get('audit_code_commit'),
                   actual_exit_code=audit.get('exit_code'),
                   confirmed_critical_L1_defects=audit.get('confirmed_critical_L1_defects'),
                   unresolved_scientific_conflicts=audit.get('unresolved_scientific_conflicts'),
                   training_permitted=audit.get('training_permitted')),
        failure_history=failure_history,
        scientific_batch_snapshot=dict(
            observed_at_utc=datetime.now(UTC).isoformat(),
            status=batch.get('status', 'NOT_STARTED'),
            experiment_code_commit=batch.get('experiment_code_commit'),
            actual_computation_pid=batch.get('pid'), launcher_pid=launcher.get('launcher_pid'),
            direct_child_pid=worker.get('direct_child_pid'),
            worker_exit_code=worker.get('direct_child_exit_code'),
            current_run=batch.get('current_run'), completed_runs=batch.get('completed_runs'),
            actual_training_transitions=batch.get('actual_training_transitions'),
            actual_sac_updates=batch.get('actual_sac_updates'),
            note='Running snapshot only; not the final six-run result or scientific Gate.'),
        scientific_stage2_decision='AWAITING_COMPLETED_R2_RESULT_ANALYSIS',
        extra_file_hashes_generated=False)


def refresh_receipts() -> None:
    """显式完整生成时刷新派生回执；仅清单模式不调用本函数。"""
    summary = test_summary()
    write_json(TASK/'TEST_RESULTS_SUMMARY.json', summary)
    write_json(TASK/'PRETRAIN_RESULT.json', pretrain_result(summary))
    write_json(TASK/'PUBLIC_ENVIRONMENT_SUMMARY.json', environment_summary())
    records = []
    source = TASK/'commands.jsonl'
    if source.is_file():
        records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()
                   if line.strip()]
    lines = ['R2 actual selected command receipts',
             'Project-machine root is displayed as <PROJECT_ROOT>; original receipts unchanged.',
             'No command is executed by this publication helper.', '']
    for row in records:
        if row.get('name') not in COMMANDS:
            continue
        selected = {key: row.get(key) for key in (
            'name', 'command', 'start_utc', 'end_utc', 'exit_code',
            'code_commit', 'direct_child_pid', 'actual_worker_pid')}
        lines.append(public_text(json.dumps(selected, ensure_ascii=False)))
    worker = read_json(TASK/'r2_scientific_batch.worker.json')
    if worker and not any(row.get('name') == 'r2_scientific_batch' for row in records):
        lines.extend(['', 'RUNNING COMMAND RECEIPT (no exit code yet):',
                      public_text(json.dumps(worker, ensure_ascii=False))])
    lines.extend(['', 'Failure history is preserved in PRETRAIN_RESULT.json and local raw records.',
                  'A failed development check is never counted as a final PASS.'])
    (TASK/'COMMANDS.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> int:
    """仅清单模式保留全部历史回执；缺失文件列出而不伪造完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-only', action='store_true',
                        help='只刷新候选路径清单，不改历史时间和任何现有验收回执。')
    args = parser.parse_args()
    if not args.paths_only:
        refresh_receipts()
    names = ['prepare_publication.py', 'COMMANDS.txt', 'TEST_RESULTS_SUMMARY.json',
             'PRETRAIN_RESULT.json',
             'PUBLIC_ENVIRONMENT_SUMMARY.json', 'PRETRAIN_MODEL_REWARD_AUDIT.json',
             'PRETRAIN_MODEL_REWARD_AUDIT.md', 'preflight_analysis.py', 'environment_preflight.py',
             'record_command.py', 'watch_actual_worker.py', 'pytest_counts.py',
             'pytest_all.xml', 'pytest_phase_a.xml', 'pytest_all_counts.json',
             'RUNNING_RECEIPT.json', 'RESULT_TOOLS_REVIEW.json', 'RUN_OPERATIONS.md',
             'analyze_r2.py', 'check_analysis_tool.py', 'final_diagnostics.py',
             'analysis/R2_RESULT.json', 'analysis/R2_RESULT_REPORT.md', 'launcher_identity.json',
             'storage_usage.json']
    for name in FINAL:
        names.extend(f'{name}.{suffix}' for suffix in ('stdout.txt', 'stderr.txt', 'worker.json'))
    entries, missing = [], []
    for name in names:
        path = TASK/name
        if path.is_file():
            entries.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size))
        else:
            missing.append(name)
    write_json(TASK/'PUBLICATION_PATHS.json', dict(
        generated_at_utc=datetime.now(UTC).isoformat(), recommended_existing_files=entries,
        missing_future_artifacts=missing, total_existing_bytes=sum(row['bytes'] for row in entries),
        requires_manual_staging_review=True, git_operations_performed=False, hashes_generated=False,
        excludes=['C300/seed_*/ full Replay/segments', 'G200/seed_*/ full Replay/segments',
                  'models and checkpoints', 'validation_manifest.json', 'runtime/', 'test_temp_*/',
                  'protocol Word/full extracted text', 'raw commands.jsonl',
                  'raw GPU process lists', 'redundant environment probe stdout/stderr',
                  'development/ temporary copies and raw fixture output', 'partial_analysis/',
                  'live batch_state.json', 'live actual_training_worker_exit.json / WAITING'],
        note='Explicit small-file suggestion, not git add and not a scientific Gate. '
             'Future result files require size/content review before publication.'))
    print(json.dumps(dict(status='PUBLICATION_SUGGESTION_WRITTEN',
                          files=len(entries), bytes=sum(row['bytes'] for row in entries))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
