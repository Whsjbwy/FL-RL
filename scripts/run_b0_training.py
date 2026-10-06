"""B0入口默认preflight；只允许真实登记匹配的科研批次或独立工程夹具。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from auv_risk_rl.training.config import B0HarnessConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_run_config(path: Path) -> tuple[B0HarnessConfig, dict[str, Any]]:
    """严格区分科学配置与工程夹具，未知字段由dataclass拒绝。"""
    from auv_risk_rl.rl.config import SACConfig
    from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config

    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if set(raw) != {"project_config", "scenario_config", "harness", "preparation"}:
        raise ValueError("运行YAML顶层字段不匹配")
    values = dict(raw["harness"])
    values["sac"] = derived_sac_config(
        SACConfig(**values["sac"]), training_seed=values["training_seed"],
        run_kind=values["run_kind"],
    )
    return B0HarnessConfig(**values), raw


def git_code_version() -> str:
    """读取正常Git提交标识，不计算额外文件摘要。"""
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """科研批次使用完整登记守卫；旧未登记准备配置仍不能执行。"""
    from auv_risk_rl.config import load_project_config
    from auv_risk_rl.env.scenario_generator import load_training_scenario_config
    from auv_risk_rl.training.harness import B0TrainingHarness

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage2_b0.yaml")
    parser.add_argument("--run-kind", choices=("engineering_smoke", "scientific_training"))
    parser.add_argument("--transition-budget", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--trusted-local", action="store_true")
    parser.add_argument("--mvp-registration", type=Path)
    parser.add_argument("--resume-batch", action="store_true")
    args = parser.parse_args(argv)
    if args.mvp_registration is not None:
        from auv_risk_rl.training.mvp_batch import finalize_results, run_batch
        from auv_risk_rl.training.mvp_registration import validate_registration

        try:
            registration = validate_registration(
                ROOT, args.mvp_registration, run_kind=args.run_kind,
                budget=args.transition_budget, output=args.output_dir, execute=args.execute,
            )
        except ValueError as error:
            parser.error(str(error))
        if args.resume is not None or args.trusted_local:
            parser.error("批次只从自身可信最新恢复点继续，请使用--resume-batch")
        if not args.execute:
            print(json.dumps(dict(status="PREFLIGHT_ONLY", registration=registration,
                                  code_version=git_code_version()), ensure_ascii=False, indent=2))
            return 0
        summary = run_batch(ROOT, registration, resume=args.resume_batch)
        if summary.get('status') == 'COMPLETED':
            finalize_results(ROOT, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.resume_batch:
        parser.error("resume-batch必须同时指定实际mvp-registration")
    config, raw = read_run_config(args.config)
    identity = dict(status="PREFLIGHT_ONLY", config=asdict(config),
                    code_version=git_code_version(), scientific_stage2_status="NOT_RUN")
    if not args.execute:
        print(json.dumps(identity, ensure_ascii=False, indent=2))
        return 0
    if args.run_kind != config.run_kind or args.transition_budget != config.transition_budget:
        parser.error("执行必须显式给出与配置相符的run-kind和transition-budget")
    if (config.run_kind == "scientific_training"
            or not raw["preparation"].get("execution_authorized")):
        parser.error("本轮只授权工程准备；科研运行安排尚未登记/授权")
    if args.output_dir is None:
        parser.error("执行必须指定独立output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def log_sink(kind: str, record: dict[str, Any]) -> None:
        """每个episode/更新流式落盘，不保存重复场景全集。"""
        with (args.output_dir / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    harness = B0TrainingHarness(
        config, load_project_config(ROOT / raw["project_config"]),
        load_training_scenario_config(ROOT / raw["scenario_config"]),
        code_version=identity["code_version"], log_sink=log_sink,
    )
    if args.resume:
        harness.load_checkpoint(args.resume, trusted_local=args.trusted_local)
    try:
        summary = harness.run(checkpoint_callback=lambda current: current.save_checkpoint(
            args.output_dir / "latest_resume.pt"))
        harness.save_checkpoint(args.output_dir / "latest_resume.pt")
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        (args.output_dir / "failure.json").write_text(json.dumps(dict(
            status="FAILED_ENGINEERING_RUN", traceback=traceback.format_exc(),
            summary=harness.summary(), failure_metadata=harness.failure_metadata,
            previous_checkpoint_only="latest_resume.pt; no partial-step snapshot",
        ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
