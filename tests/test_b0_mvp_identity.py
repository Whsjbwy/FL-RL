"""正常Git提交与工作树身份守卫的纯mock测试；不计算文件摘要或运行环境。"""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from auv_risk_rl.training import mvp_registration


def test_clean_git_identity_uses_commit_and_protected_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """使用已有commit及标准diff/untracked查询，不靠摘要清单或历史PASS。"""
    calls: list[tuple[str, ...]] = []

    def query(root: Path, *arguments: str) -> str:
        """纯mock查询记录，不读写实际Git或用户分支。"""
        assert root == tmp_path
        calls.append(arguments)
        return ''

    monkeypatch.setattr(mvp_registration, 'git', query)
    mvp_registration.verify_code_identity(tmp_path, 'actual-experiment-commit')
    assert calls[0] == ('cat-file', '-e', 'actual-experiment-commit^{commit}')
    for arguments in calls[1:]:
        protected = {'src', 'tests', 'scripts', 'configs', mvp_registration.DOCUMENT,
                     'pyproject.toml', 'requirements-b1-tools.txt'}
        assert protected <= set(arguments)
        assert 'results' not in arguments
    assert calls[1][:3] == ('diff', '--name-only', 'actual-experiment-commit')
    assert calls[2][:3] == ('ls-files', '--others', '--exclude-standard')


@pytest.mark.parametrize('path', [
    'src/auv_risk_rl/training/curriculum.py', 'tests/test_b0_curriculum.py',
    'configs/stage2_b0_mvp_v1.yaml', 'docs/STAGE2_B0_MVP_V1.md',
    'pyproject.toml', 'requirements-b1-tools.txt',
])
def test_git_tracked_changes_refuse_execution_or_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str,
) -> None:
    """源码/测试/配置/登记发生变化必须拒绝；不自动回退或修复用户工作。"""
    def query(root: Path, *arguments: str) -> str:
        """只注入实际Git diff形式的结果。"""
        return path if arguments[0] == 'diff' else ''

    monkeypatch.setattr(mvp_registration, 'git', query)
    with pytest.raises(ValueError, match='偏离'):
        mvp_registration.verify_code_identity(tmp_path, 'fixed-code')


@pytest.mark.parametrize('path', [
    'src/auv_risk_rl/rl/fedavg.py', 'src/auv_risk_rl/rl/transformer_policy.py',
    'tests/test_new_unregistered_method.py',
])
def test_git_untracked_protected_files_are_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str,
) -> None:
    """模拟未登记文件名称不等于实现这些算法，守卫必须明确拒绝。"""
    def query(root: Path, *arguments: str) -> str:
        """只返回一条模拟untracked路径，实际磁盘无新科学源码。"""
        return path if arguments[0] == 'ls-files' else ''

    monkeypatch.setattr(mvp_registration, 'git', query)
    with pytest.raises(ValueError, match='偏离'):
        mvp_registration.verify_code_identity(tmp_path, 'fixed-code')


def test_missing_experiment_commit_propagates_git_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """本机缺少原实验提交不能伪装相同代码继续恢复。"""
    def query(root: Path, *arguments: str) -> str:
        """模拟正常Git不能解析commit，而不是伪造守卫返回PASS。"""
        raise subprocess.CalledProcessError(128, ['git', *arguments])

    monkeypatch.setattr(mvp_registration, 'git', query)
    with pytest.raises(subprocess.CalledProcessError):
        mvp_registration.verify_code_identity(tmp_path, 'missing-code')


def test_git_wrapper_preserves_normal_safe_directory_and_failure_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """只设置当前仓库safe.directory，不修改全局身份或绕过Git命令错误。"""
    def execute(command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        """检查子进程参数，不调用真实Git或网络。"""
        assert command == ['git', '-c', f'safe.directory={tmp_path.as_posix()}',
                           'rev-parse', 'HEAD']
        assert options == dict(cwd=tmp_path, text=True, capture_output=True, check=True)
        return subprocess.CompletedProcess(command, 0, stdout=' fixed-code\n')

    monkeypatch.setattr(subprocess, 'run', execute)
    assert mvp_registration.git(tmp_path, 'rev-parse', 'HEAD') == 'fixed-code'
