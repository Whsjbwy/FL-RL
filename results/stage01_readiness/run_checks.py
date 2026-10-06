"""本轮非学习验收记录器；不运行训练、benchmark 或额外摘要检查。"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    """顺序运行实际检查并保存命令、退出码及完整标准输出/错误。"""

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPYCACHEPREFIX", None)
    env["AUV_AUDIT_PROJECT_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    commands = [
        ("pytest_all", [sys.executable, "-m", "pytest", "tests", "-q", "-p",
                        "no:cacheprovider", "-p", "results.stage01_readiness.pytest_counters",
                        f"--basetemp={OUT / 'development/pytest_workspace'}",
                        f"--junitxml={OUT / 'pytest_all.xml'}"]),
        ("pytest_phase_a", [sys.executable, "-m", "pytest",
                            "handoff/reference_tests/test_phase_a_conformance.py", "-q",
                            "-p", "no:cacheprovider",
                            f"--junitxml={OUT / 'pytest_phase_a.xml'}"]),
        ("stage1_acceptance", [sys.executable, "scripts/run_stage1_acceptance.py",
                               "--output-dir", str(OUT / "stage1")]),
        ("stage0_interface_smoke", [sys.executable,
                                    "scripts/run_stage0_integration_smoke.py"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("compileall", [sys.executable, "-X",
                        f"pycache_prefix={OUT / 'development/compile_cache'}",
                        "-m", "compileall", "-q", "src", "tests", "scripts"]),
        ("stage0_quality_diagnostic", [sys.executable, "-c",
            "import importlib.util,json,pathlib; "
            "s=importlib.util.spec_from_file_location('audit','scripts/run_stage0_audit.py'); "
            "a=importlib.util.module_from_spec(s); s.loader.exec_module(a); "
            "a.RESULTS_ROOT=pathlib.Path('results/stage01_readiness/quality'); "
            "a.RESULTS_ROOT.mkdir(parents=True,exist_ok=True); "
            "r=a._write_quality_audit(); "
            "status,evidence=a._check_required_stage0_tests(); "
            "print(json.dumps({'quality':r,'required_tests_status':status,"
            "'required_tests_evidence':evidence},ensure_ascii=False,indent=2)); "
            "raise SystemExit(int(status=='FAIL' or any(x['status']=='FAIL' for x in r)))"]),
    ]
    results = []
    for name, command in commands:
        record = {"name": name, "command": command, "cwd": str(ROOT),
                  "tested_commit": revision, "started_at": datetime.now(UTC).isoformat()}
        print(f"START {name}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, check=False)
        (OUT / f"{name}.stdout.txt").write_bytes(completed.stdout)
        (OUT / f"{name}.stderr.txt").write_bytes(completed.stderr)
        record.update(exit_code=completed.returncode, ended_at=datetime.now(UTC).isoformat())
        results.append(record)
        (OUT / "commands.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"END {name} exit={completed.returncode}", flush=True)
    (OUT / "commands.txt").write_text("\n\n".join(
        f"{r['name']}\ncwd={r['cwd']}\n{r['command']}\nexit_code={r['exit_code']}"
        for r in results), encoding="utf-8")
    (OUT / "run_identity.json").write_text(json.dumps({
        "tested_commit": revision, "interpreter": sys.executable,
        "python_version": sys.version, "os": platform.platform(),
        "source_origin": str(ROOT), "extra_digest_checks": "RETIRED / NOT RUN",
        "config_paths": ["configs/stage0.yaml", "configs/train_scenario_v1.yaml"],
        "workload": "scientific unit/integration checks plus nonlearning acceptance",
    }, indent=2), encoding="utf-8")
    return int(any(r["exit_code"] for r in results
                   if r["name"] != "stage0_quality_diagnostic"))


if __name__ == "__main__":
    raise SystemExit(main())
