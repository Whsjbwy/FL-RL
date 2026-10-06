"""CM13真实性检查：测试映射真实存在、Ruff不可静默跳过；纯摘要检查已退役。"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def audit_module():
    """读取当前审计脚本的检查函数，不执行main或另一个训练器。"""

    script = Path(__file__).resolve().parents[1] / "scripts/run_stage0_audit.py"
    spec = importlib.util.spec_from_file_location("b1_audit_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equation_map_rejects_nonexistent_test_function(
    audit_module, tmp_path, monkeypatch
) -> None:
    """映射字符串非空但函数不存在时，不能记为PASS。"""

    monkeypatch.setattr(audit_module, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_real.py").write_text("def test_real():\n    pass\n", encoding="utf-8")
    mapping = tmp_path / "docs/equation_code_map.csv"
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["equation_id", "test_evidence"])
        writer.writeheader()
        writer.writerow({"equation_id": "Eq. (38)",
                         "test_evidence": "tests/test_real.py::test_missing"})
    assert audit_module._check_equation_test_map()[0] == "FAIL"
    mapping.write_text(mapping.read_text().replace("test_missing", "test_real"), encoding="utf-8")
    assert audit_module._check_equation_test_map()[0] == "PASS"


def test_missing_ruff_is_quality_failure(audit_module, monkeypatch) -> None:
    """明确缺少必需Ruff时，审计不得降为WARNING后给出GO。"""

    monkeypatch.setattr(audit_module.shutil, "which", lambda name: None)
    status, reason = audit_module._check_ruff_if_available()
    assert status == "FAIL"
    assert "Ruff" in reason
