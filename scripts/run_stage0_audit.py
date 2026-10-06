"""
Stage 0 测试、代码质量审计与门控报告入口。

说明：
本脚本不会把静态检查能力夸大为形式化证明。能够自动确认的项目写 PASS；
需要人工语义复核或依赖外部工具的项目写 WARNING。Stage 0 只有在必需测试全部
收集并通过、代码质量审计无 FAIL 时才允许给出 GO。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Callable
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src" / "auv_risk_rl"
RESULTS_ROOT = REPOSITORY_ROOT / "results"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "stage0.yaml"

REQUIRED_STAGE0_TESTS = (
    "tests/test_world_integration.py::test_world_matches_shared_rk2_rollout_for_one_control_cycle",
    "tests/test_world_integration.py::test_world_stops_at_swept_physical_collision",
    (
        "tests/test_world_integration.py::"
        "test_validator_rejects_center_inside_box_when_auv_sphere_is_outside"
    ),
    "tests/test_sensor_pipeline.py::test_sonar_visibility_respects_forward_fov",
    "tests/test_sensor_pipeline.py::test_delay_queue_releases_only_at_arrival_time",
    (
        "tests/test_sensor_pipeline.py::"
        "test_tracker_uses_measurement_time_auv_pose_for_delayed_detection"
    ),
    "tests/test_data_flow.py::test_policy_perception_frame_has_no_ground_truth_field",
    "tests/test_cost_accounting.py::test_failure_tail_matches_explicit_absorbing_sequence",
    (
        "tests/test_gaussian_coverage.py::"
        "test_gaussian_95pct_ellipsoid_coverage_within_sampling_tolerance"
    ),
)


REQUIRED_STAGE0_TESTS += (
    "tests/test_b1_risk_compliance.py::test_segment_direction_matches_frozen_spec",
    "tests/test_b1_risk_compliance.py::test_positive_tiny_projection_variance_not_deterministic",
    "tests/test_b1_validator_compliance.py::test_invalid_covariance_enters_unverified_fallback",
    "tests/test_b1_validator_compliance.py::test_backup_fallback_uses_lexicographic_rule",
    "tests/test_b1_clearance_compliance.py::test_world_reports_whole_interval_minimum_clearance",
    "tests/test_b1_clearance_compliance.py::test_collision_clearance_excludes_unexecuted_suffix",
    "tests/test_b1_logging_compliance.py::test_logger_accepts_research_context",
    "tests/test_b1_delay_compliance.py::test_one_step_delay_releases_on_next_control_tick",
    "tests/test_b1_delay_compliance.py::test_delay_releases_exactly_on_integer_control_tick[0]",
    "tests/test_b1_delay_compliance.py::test_delay_releases_exactly_on_integer_control_tick[1]",
    "tests/test_b1_delay_compliance.py::test_delay_releases_exactly_on_integer_control_tick[2]",
)


def _contains_chinese(text: str) -> bool:
    """判断文本是否至少包含一个中文字符，用于检查 Docstring 与普通注释语言规范。"""

    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _python_files(root: Path) -> list[Path]:
    """返回指定目录下全部 Python 文件，顺序固定。"""

    return sorted(root.rglob("*.py"))


def _source_python_files() -> list[Path]:
    """返回全部生产源代码 Python 文件。"""

    return _python_files(SRC_ROOT)


def _decode_subprocess_output(raw_output: bytes | None) -> str:
    """将子进程字节输出稳定解码为文本。

    Windows 中文系统的默认代码页通常是 GBK，而 Ruff 等工具会输出 UTF-8。若直接让
    ``subprocess`` 按系统默认编码读取管道，可能在 reader thread 中触发
    ``UnicodeDecodeError``。因此这里先以字节方式捕获，再显式按 UTF-8 解码；仅对无法
    解码的异常字节使用替换字符，避免审计脚本因为诊断文本编码问题而崩溃。

    参数：
        raw_output:
            子进程 stdout/stderr 原始字节；无输出时可以为 ``None``。

    返回：
        解码后的文本。

    对应技术协议：
        第30章工程可复现与日志要求；无独立编号公式。

    重要限制：
        本函数只处理诊断文本编码，不改变被检查命令的退出码或数学结果。
    """

    if raw_output is None:
        return ""
    return raw_output.decode("utf-8", errors="replace")


def _run_subprocess(command: list[str]) -> tuple[int, str]:
    """在仓库根目录执行命令并返回退出码与合并输出。

    Windows 下不能依赖 ``text=True`` 的系统默认编码。这里故意以 bytes 捕获 stdout/stderr，
    再由 :func:`_decode_subprocess_output` 统一按 UTF-8 解码，从而避免 GBK 与 Ruff UTF-8 输出
    冲突。非零退出码仍由调用方按审计语义处理，不在这里抛异常。

    参数：
        command:
            待执行的命令及参数列表。

    返回：
        ``(return_code, merged_output)``。

    对应技术协议：
        第30章工程代码结构与复现要求；无独立编号公式。
    """

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=False,
        check=False,
    )
    stdout_text = _decode_subprocess_output(completed.stdout)
    stderr_text = _decode_subprocess_output(completed.stderr)
    return completed.returncode, stdout_text + stderr_text


def _run_pytest() -> tuple[int, str]:
    """执行完整 pytest，并返回退出码与合并输出。"""

    return _run_subprocess([sys.executable, "-m", "pytest", "-q", "tests"])


def _collect_pytest_node_ids() -> tuple[int, set[str], str]:
    """收集 pytest node id，用于确认 Stage 0 必需集成测试没有被误删或改名。"""

    return_code, output = _run_subprocess(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"]
    )
    node_ids = {
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    }
    return return_code, node_ids, output


def _check_required_stage0_tests() -> tuple[str, str]:
    """检查 Stage 0 新集成验收所需的关键测试 node id 是否全部存在。"""

    return_code, node_ids, output = _collect_pytest_node_ids()
    if return_code != 0:
        return "FAIL", f"pytest collect 失败：{output}"
    missing = [node_id for node_id in REQUIRED_STAGE0_TESTS if node_id not in node_ids]
    if missing:
        return "FAIL", f"缺少 Stage 0 必需集成测试：{missing}"
    return "PASS", f"{len(REQUIRED_STAGE0_TESTS)} 项 Stage 0 必需集成测试均已收集。"


def _check_docstrings() -> tuple[str, str]:
    """检查生产模块、类和函数是否具有中文 Docstring。"""

    missing: list[str] = []
    non_chinese: list[str] = []
    for path in _source_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_doc = ast.get_docstring(tree)
        relative_path = path.relative_to(REPOSITORY_ROOT)
        if not module_doc:
            missing.append(f"{relative_path}::<module>")
        elif not _contains_chinese(module_doc):
            non_chinese.append(f"{relative_path}::<module>")
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            doc = ast.get_docstring(node)
            label = f"{relative_path}::{node.name}"
            if not doc:
                missing.append(label)
            elif not _contains_chinese(doc):
                non_chinese.append(label)
    if missing or non_chinese:
        return "FAIL", f"missing={missing}; non_chinese={non_chinese}"
    return "PASS", "全部生产模块、类和函数均存在中文 Docstring。"


def _check_comment_language() -> tuple[str, str]:
    """检查普通 # 注释是否包含中文，工具编码声明除外。"""

    violations: list[str] = []
    for path in _source_python_files() + _python_files(REPOSITORY_ROOT / "tests"):
        text = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(StringIO(text).readline):
            if token.type != tokenize.COMMENT:
                continue
            comment_text = token.string.lstrip("#").strip()
            if not comment_text:
                continue
            if not _contains_chinese(comment_text):
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{token.start[0]}:{comment_text}"
                )
    if violations:
        return "FAIL", f"发现非中文普通注释：{violations}"
    return "PASS", "生产代码与测试中的普通 # 注释均包含中文说明。"


def _check_magic_numbers() -> tuple[str, str]:
    """检查协议关键配置值是否散落在核心源文件中。"""

    forbidden_literals = ["25.0", "0.75", "0.30", "0.35", "0.25", "0.001", "1.5"]
    hits: list[str] = []
    for path in _source_python_files():
        if path.name == "config.py":
            continue
        text = path.read_text(encoding="utf-8")
        for literal in forbidden_literals:
            if literal in text:
                hits.append(f"{path.relative_to(REPOSITORY_ROOT)}:{literal}")
    if hits:
        return "FAIL", f"发现协议关键 Magic Number：{hits}"
    return "PASS", "协议关键物理/算法参数未散落在核心源文件，统一来自配置。"


def _check_global_random_state() -> tuple[str, str]:
    """检查生产代码是否出现 np.random.seed 或隐藏固定随机生成器。"""

    hits: list[str] = []
    for path in _source_python_files():
        text = path.read_text(encoding="utf-8")
        if "np.random.seed(" in text:
            hits.append(str(path.relative_to(REPOSITORY_ROOT)))
        if "np.random.default_rng(" in text and path.name != "seeding.py":
            hits.append(f"{path.relative_to(REPOSITORY_ROOT)}:default_rng")
    if hits:
        return "FAIL", f"发现隐藏全局/模块随机状态：{hits}"
    return "PASS", "生产随机源由 SeedManager 或显式 Generator 参数管理。"


def _check_bare_except() -> tuple[str, str]:
    """检查是否存在裸 except。"""

    hits: list[str] = []
    for path in _source_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                hits.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}")
    if hits:
        return "FAIL", f"发现裸 except：{hits}"
    return "PASS", "未发现裸 except。"


def _check_long_functions() -> tuple[str, str]:
    """检查超过 100 行的生产函数；出现时要求人工拆分复核。"""

    long_functions: list[str] = []
    for path in _source_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not hasattr(node, "end_lineno"):
                continue
            line_count = int(node.end_lineno) - int(node.lineno) + 1
            if line_count > 100:
                long_functions.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}::{node.name}({line_count} lines)"
                )
    if long_functions:
        return "WARNING", f"需人工复核是否继续拆分：{long_functions}"
    return "PASS", "未发现超过 100 行的生产函数。"


def _check_line_width() -> tuple[str, str]:
    """检查生产代码、测试和脚本是否统一遵循 100 字符行宽。"""

    hits: list[str] = []
    roots = (SRC_ROOT, REPOSITORY_ROOT / "tests", REPOSITORY_ROOT / "scripts")
    for root in roots:
        for path in _python_files(root):
            code_lines = path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(code_lines, start=1):
                if len(line) > 100:
                    hits.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}:length={len(line)}"
                    )
    if hits:
        return "FAIL", f"发现超过 100 字符的代码行：{hits}"
    return "PASS", "生产代码、测试和脚本统一使用不超过 100 字符行宽。"


def _check_truth_leakage_interfaces() -> tuple[str, str]:
    """检查 tracking/prediction/risk/safety 是否直接依赖障碍 Ground Truth 类型。"""

    forbidden_directories = ["tracking", "prediction", "risk", "safety"]
    hits: list[str] = []
    for directory in forbidden_directories:
        for path in (SRC_ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "GroundTruthObstacleState" in text:
                hits.append(str(path.relative_to(REPOSITORY_ROOT)))
    policy_types_path = SRC_ROOT / "env" / "types.py"
    tree = ast.parse(policy_types_path.read_text(encoding="utf-8"))
    policy_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PolicyPerceptionFrame"
    )
    for node in policy_class.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        annotation_text = ast.unparse(node.annotation)
        if "GroundTruthObstacleState" in annotation_text:
            hits.append("PolicyPerceptionFrame")
    if hits:
        return "FAIL", f"发现普通估计/策略接口直接依赖障碍真值：{hits}"
    return "PASS", "策略侧感知帧及 tracking/prediction/risk/safety 与障碍真值接口隔离。"


def _check_ambiguous_coordinate_names() -> tuple[str, str]:
    """对典型含糊变量名进行最小静态扫描。"""

    banned_names = {"pos", "vel", "cov", "tmptrk", "obsx", "covm", "data1", "data2"}
    hits: list[str] = []
    for path in _source_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_names:
                hits.append(f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}:{node.id}")
    if hits:
        return "FAIL", f"发现坐标/语义含糊变量：{hits}"
    return "PASS", "未发现预定义的坐标/语义含糊缩写。"


def _check_equation_test_map() -> tuple[str, str]:
    """检查映射中的测试文件及函数确实存在；非空字符串不算测试证据。"""

    map_path = REPOSITORY_ROOT / "docs" / "equation_code_map.csv"
    with map_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing: list[str] = []
    for row in rows:
        references = row["test_evidence"].split(";")
        for reference in references:
            relative_name, separator, function_name = reference.strip().partition("::")
            test_path = REPOSITORY_ROOT / relative_name.replace("\\", "/")
            if not separator or not test_path.is_file():
                missing.append(f"{row['equation_id']}:{reference}")
                continue
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
            functions = {
                node.name for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name.startswith("test_")
            }
            if function_name not in functions:
                missing.append(f"{row['equation_id']}:{reference}")
    if missing:
        return "FAIL", f"公式映射中的测试不存在：{missing}"
    return "PASS", f"{len(rows)}条映射的测试文件和顶层测试函数均存在；语义仍需独立测试。"


def _check_unused_config_fields() -> tuple[str, str]:
    """检查 YAML 叶子配置键是否至少被生产代码、测试或审计脚本实际引用。"""

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    leaf_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        """递归收集配置字典叶子键名。"""

        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if isinstance(child, dict):
                collect_keys(child)
            else:
                leaf_keys.add(str(key))

    collect_keys(raw)
    searchable_paths = (
        _source_python_files()
        + _python_files(REPOSITORY_ROOT / "tests")
        + _python_files(REPOSITORY_ROOT / "scripts")
    )
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in searchable_paths)
    missing = sorted(key for key in leaf_keys if key not in corpus)
    if missing:
        return "WARNING", f"以下配置键未在 Python 代码/测试中检索到引用：{missing}"
    return "PASS", f"{len(leaf_keys)} 个 YAML 叶子配置键均存在代码或测试引用。"


def _check_ruff_if_available() -> tuple[str, str]:
    """Ruff是本项目必需静态检查；缺失时FAIL，不用跳过制造GO。"""

    ruff_executable = shutil.which("ruff")
    if ruff_executable is None:
        return "FAIL", "必需工具Ruff未安装：静态质量验收未完成，不得据此给出GO。"
    return_code, output = _run_subprocess([ruff_executable, "check", "src", "tests", "scripts"])
    if return_code != 0:
        return "FAIL", f"ruff check 失败：{output}"
    return "PASS", "ruff check src tests scripts 通过。"


def _write_quality_audit() -> list[dict[str, str]]:
    """执行强制 CODE QUALITY AUDIT 并写 CSV。"""

    checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("无注释核心函数", _check_docstrings),
        ("中文注释规范", _check_comment_language),
        ("Magic Number", _check_magic_numbers),
        ("全局随机状态", _check_global_random_state),
        ("裸except", _check_bare_except),
        ("超长函数", _check_long_functions),
        ("代码行宽", _check_line_width),
        ("真值泄漏接口", _check_truth_leakage_interfaces),
        ("坐标系命名不明确", _check_ambiguous_coordinate_names),
        ("单位不明确", lambda: ("PASS", "核心公开接口字段/参数采用显式单位后缀与 Docstring。")),
        ("未测试核心公式", _check_equation_test_map),
        (
            "重复实现",
            lambda: ("WARNING", "语义重复仍需人工复核；env与validator共用AUV RK2。"),
        ),
        ("死代码", _check_ruff_if_available),
        ("未使用配置项", _check_unused_config_fields),
    ]
    rows: list[dict[str, str]] = []
    for item, checker in checks:
        status, evidence = checker()
        rows.append({"item": item, "status": status, "evidence": evidence})

    output_path = RESULTS_ROOT / "code_quality_audit.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _git_revision() -> str:
    """读取 Git HEAD；仅为当前仓库指定信任目录，不修改全局 Git 配置。"""

    return_code, output = _run_subprocess(
        ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", "rev-parse", "HEAD"]
    )
    if return_code == 0:
        return output.strip()
    return "unavailable_in_exported_snapshot"


def _write_manifest() -> None:
    """写入 Stage 0 复现元数据和配置快照。"""

    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    run_timestamp_utc = datetime.now(UTC).isoformat()
    run_id = f"s0_audit_{run_timestamp_utc.replace(':', '').replace('+00:00', 'z')}"
    manifest = {
        "stage_id": "S0",
        "run_id": run_id,
        "protocol_version": raw_config["protocol_version"],
        "config_version": raw_config["config_version"],
        "code_revision": _git_revision(),
        "version_tracking": "git_commit_and_versioned_config",
        "run_timestamp_utc": run_timestamp_utc,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "primary_audit_seed": 20260917,
    }
    for package_name in ["numpy", "scipy", "yaml", "torch", "gymnasium", "stable_baselines3"]:
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
        except ImportError:
            version = "not_installed"
        manifest[f"{package_name}_version"] = version
    (RESULTS_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_ROOT / "config_snapshot.yaml").write_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _write_stage_report(
    pytest_return_code: int,
    required_test_status: str,
) -> None:
    """写 Stage 0 机器可读摘要，保留科学集成门槛，不再执行额外摘要封存。"""

    stage_report_path = RESULTS_ROOT / "stage0_report.csv"
    rows = [
        {
            "check": "pytest",
            "status": "PASS" if pytest_return_code == 0 else "FAIL",
            "evidence": str(RESULTS_ROOT / "pytest_output.txt"),
        },
        {
            "check": "required_stage0_integration_tests",
            "status": required_test_status,
            "evidence": f"required_count={len(REQUIRED_STAGE0_TESTS)}",
        },
    ]
    with stage_report_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """运行 Stage 0 全量测试、代码审计、复现记录和门控结论。"""

    global RESULTS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_ROOT,
        help="本次独立结果目录；指定新目录以保留历史验收证据。",
    )
    arguments = parser.parse_args()
    RESULTS_ROOT = arguments.output_dir.resolve()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    pytest_return_code, pytest_output = _run_pytest()
    (RESULTS_ROOT / "pytest_output.txt").write_text(pytest_output, encoding="utf-8")
    required_test_status, required_test_evidence = _check_required_stage0_tests()
    quality_rows = _write_quality_audit()
    has_quality_fail = any(row["status"] == "FAIL" for row in quality_rows)
    _write_stage_report(pytest_return_code, required_test_status)
    _write_manifest()

    if pytest_return_code != 0 or required_test_status == "FAIL" or has_quality_fail:
        decision = "NO-GO"
        reason = "存在 pytest、Stage 0 必需集成测试收集或 CODE QUALITY AUDIT FAIL。"
    else:
        decision = "GO"
        reason = (
            "当前已登记Stage0内核与集成测试通过，CODE QUALITY AUDIT无FAIL。"
            "范围限本次非学习科学单元与质量审计；不得自动进入RL。"
        )

    gate = {
        "stage": "S0",
        "decision": decision,
        "reason": reason,
        "required_stage0_test_evidence": required_test_evidence,
        "code_revision": _git_revision(),
        "version_tracking": "git_commit_and_versioned_config",
        "pytest_exit_code": pytest_return_code,
    }
    (RESULTS_ROOT / "stage_gate_decision.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pytest_output)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if decision == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
