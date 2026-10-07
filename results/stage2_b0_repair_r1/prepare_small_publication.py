"""保留本地完整提交，创建仅含小证据的公开分支；不改科学文件。"""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/stage2_b0_repair_r1'
GIT = ['D:/Program Files/Git/cmd/git.exe', '-c', f'safe.directory={ROOT.as_posix()}']
BASE = 'f89bfa4af3cad0c6da9ab64b79da7ae1b267dbf7'
LOCAL = 'ca4e67b88941a19ee1974be8150b58e7a443e5dd'
BRANCH = 'codex/stage2-b0-repair-r1-public'


def git(*args: str, env=None) -> str:
    """只在本仓库调用普通Git，不强推、不重写已有分支。"""
    return subprocess.run(GIT + list(args), cwd=ROOT, env=env, text=True,
                          capture_output=True, check=True).stdout.strip()


def main() -> None:
    """独立临时index选文件，原工作区及完整本地历史均保持。"""
    selected = ['.gitignore', 'docs/STAGE2_B0_REPAIR_R1.md',
                'configs/stage2_b0_repair_r1.yaml', 'scripts/run_b0_training.py',
                'src/auv_risk_rl/training/config.py', 'src/auv_risk_rl/training/mvp_batch.py',
                'src/auv_risk_rl/training/mvp_analysis.py',
                'src/auv_risk_rl/training/repair_r1.py',
                'src/auv_risk_rl/training/repair_registration.py',
                'tests/test_b0_repair_entry.py', 'tests/test_b0_repair_r1.py',
                'tests/test_b0_r1_numerics.py']
    for directory in (OUT, OUT / 'math', OUT / 'offline'):
        selected.extend(path.relative_to(ROOT).as_posix() for path in directory.iterdir()
                        if path.is_file() and path.suffix in ('.py', '.md', '.csv'))
    names = ['branch_decision.json', 'control_comparability.json', 'diag_budget.json',
             'replay_mechanism_evidence.json', 'replay_selection.json', 'frozen_snapshots.json',
             'pytest_all_counts.json', 'pytest_phase_a_counts.json',
             'pytest_all.xml', 'pytest_phase_a.xml', 'targeted_final.xml',
             'r1_pytest_all.stdout.txt', 'r1_pytest_all.stderr.txt', 'r1_pytest_all.worker.json',
             'r1_phase_a.stdout.txt', 'r1_phase_a.stderr.txt', 'r1_phase_a.worker.json',
             'r1_static_final.stdout.txt', 'r1_static_final.worker.json',
             'r1_compileall.stdout.txt', 'r1_compileall.worker.json',
             'repair_goal_diagnostic_roundoff_counterexample.stdout.txt',
             'repair_goal_diagnostic_roundoff_counterexample.xml',
             'commands.jsonl', 'publication_plan.json']
    names += ['math/' + name for name in (
        'actual_failure_input_review.json', 'critic_registration.json', 'critic_result.json',
        'equation53_source.json', 'math_result.json', 'numerics.xml', 'commands.json')]
    selected += ['results/stage2_b0_repair_r1/' + name for name in names]
    selected = sorted(set(selected))
    plan = dict(reason='Automatic public-push review rejected broad diagnostic artifacts.',
                local_full_commit_preserved=LOCAL, public_parent=BASE, branch=BRANCH,
                no_existing_history_rewritten=True, scientific_code_unchanged=True,
                original_data_untouched=True, selected_paths=selected,
                local_only=['frozen_replay_result.json', 'offline/v1_offline_analysis.json',
                            'diagnostic_replay/', 'models/', 'local_snapshots/'])
    (OUT / 'publication_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)
                                           + '\n', encoding='utf-8')
    for name in selected:
        if not (ROOT / name).is_file() or (ROOT / name).stat().st_size > 150000:
            raise ValueError(f'公开文件缺失或超过150kB人工审阅界限：{name}')
    index = OUT / 'development/publication.index'
    index.unlink(missing_ok=True)
    env = {**os.environ, 'GIT_INDEX_FILE': str(index)}
    git('read-tree', BASE, env=env)
    git('add', '--', *selected, env=env)
    tree = git('write-tree', env=env)
    commit = git('commit-tree', tree, '-p', BASE, '-m',
                 'Publish minimal R1 code, registration and diagnostic evidence')
    git('branch', BRANCH, commit)
    git('symbolic-ref', 'HEAD', 'refs/heads/' + BRANCH)
    git('read-tree', commit)
    protected = ['src', 'tests', 'scripts', 'configs', 'docs/STAGE2_B0_REPAIR_R1.md']
    if git('diff', '--name-only', LOCAL, commit, '--', *protected):
        raise RuntimeError('公开提交科学文件与实际被测版本不一致。')
    print(json.dumps(dict(public_code_commit=commit, branch=BRANCH,
                          protected_git_diff_from_tested_commit=[],
                          selected_files=len(selected)), ensure_ascii=False))


if __name__ == '__main__':
    main()
