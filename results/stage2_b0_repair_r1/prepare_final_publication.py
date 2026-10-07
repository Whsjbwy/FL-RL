"""只整理R1小体积公开证据和真实命令；不运行科学计算或Git操作。"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
PREFIX = TASK.relative_to(ROOT).as_posix()

FINAL_COMMANDS = {
    "control_comparability_final", "r1_targeted_final", "r1_pytest_all", "r1_phase_a",
    "r1_static_final", "r1_compileall", "r1_scientific_batch", "actual_worker_watch",
    "r1_final_evidence_audit_retry", "r1_final_analysis_retry_serialization",
    "r1_analyzer_numpy_serialization_check", "r1_actual_results_plot_png",
    "r1_final_report_generation", "r1_final_report_generation_clarified",
}
FIRST_FAILURES = {
    "repair_goal_diagnostic_roundoff_counterexample", "repair_harness_targeted",
    "r1_ruff", "r1_final_analysis", "r1_final_evidence_audit",
}
REQUIRED_COMMANDS = {
    "r1_pytest_all", "r1_phase_a", "r1_static_final", "r1_compileall",
    "r1_scientific_batch", "r1_final_evidence_audit_retry",
    "r1_final_analysis_retry_serialization", "r1_actual_results_plot_png",
}
RESULT_PATHS = (
    "R1_RESULT.json", "R1_RESULT_REPORT.md", "storage_usage.json",
    "COMMANDS.txt", "build_final_report.py", "prepare_final_publication.py",
    "final_publication_paths.json", "analyze_r1.py", "audit_final_evidence.py",
    "watch_actual_worker.py", "batch_state.json", "actual_training_worker_exit.json",
    "diag_budget.json", "targeted_final.xml", "pytest_all.xml", "pytest_all_counts.json",
    "pytest_phase_a.xml", "pytest_phase_a_counts.json",
    "analysis/endpoint_comparison.csv", "analysis/paired_endpoint_changes.csv",
    "analysis/validation_points.csv", "analysis/training_episode_bins.csv",
    "analysis/update_bins.csv", "analysis/figures/fixed_100k_val300_raw_seeds.svg",
    "analysis/figures/paired_monitor30_learning_curves.svg",
    "analysis/figures/update_diagnostics_comparison.svg",
    "public_evidence/training_episodes.csv.gz",
    "public_evidence/validation_episodes.csv.gz",
    "public_evidence/fixed_100k_index0_trajectories.csv.gz",
    "public_evidence/fixed_diagnostics_summary.csv",
)
LOCAL_ONLY = (
    "analysis/r1_analysis.json", "analysis/final_evidence_audit.json",
    "analysis/final_fixed_diagnostics.json", "analysis/termination_strata.csv",
    "validation_manifest.json", "frozen_replay_result.json", "offline/v1_offline_analysis.json",
    "public_evidence/training_episodes.csv", "public_evidence/validation_episodes.csv",
    "public_evidence/selected_validation_trajectories.csv.gz", "commands.jsonl",
    "r1_scientific_batch.stdout.txt", "protocol_read.stdout.txt",
    "protocol_science_read.stdout.txt", "models/", "local_snapshots/", "diagnostic_replay/",
    "math/critic_rollouts/", "seed_*/latest_resume.pt", "seed_*/segments/*/*.jsonl",
    "development/", "runtime/",
)


def _write_json(path: Path, value: object) -> None:
    """只写此helper的派生结果，不修改原命令和科学数据。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _public_argument(value: str) -> str:
    """公开文本标明已有绘图解释器替代，原始绝对路径记录仍留本机。"""
    normalized = value.replace("\\", "/")
    if "/Users/" in normalized and "/.cache/codex-runtimes/" in normalized:
        return "<EXISTING_BUNDLED_PLOT_PYTHON>"
    return value


def _selected_commands() -> list[dict]:
    """每个指定命令取实际最后一条记录，不创造退出码或运行。"""
    records = [json.loads(line) for line in (TASK / "commands.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    latest = {record["name"]: (index, record) for index, record in enumerate(records)}
    missing = sorted(REQUIRED_COMMANDS - latest.keys())
    if missing:
        raise ValueError(f"缺少必要真实命令记录：{missing}")
    for name in REQUIRED_COMMANDS:
        if latest[name][1].get("exit_code") != 0:
            raise ValueError(f"必要命令没有成功结束：{name}")
    names = FINAL_COMMANDS | FIRST_FAILURES
    names.update(name for name in latest if name.startswith((
        "r1_report_builder", "r1_publication_helper", "final_publication")))
    selected = [latest[name] for name in names if name in latest]
    return [record for _, record in sorted(selected)]


def _write_commands(records: list[dict]) -> None:
    """保留实际命令参数、退出码、Git提交和时刻，注明隐私路径表示法。"""
    lines = [
        "R1 FINAL PUBLIC COMMAND RECORDS",
        "Derived from local commands.jsonl; original records are unchanged.",
        "This helper did not run any scientific training, evaluation, or Git command.",
        "Plot interpreter: <EXISTING_BUNDLED_PLOT_PYTHON> means the existing bundled",
        "Python used in the actual record. Its personal user directory is withheld here.",
        "Arguments are JSON arrays, not shell-reconstructed commands.",
        "Early nonzero exits below are preserved failures, not reported as PASS.", "",
    ]
    for record in records:
        lines.extend([
            f"name={record['name']}", f"cwd={record.get('cwd', 'NOT RECORDED')}",
            "command=" + json.dumps([_public_argument(value) for value in record["command"]],
                                     ensure_ascii=False),
            f"exit_code={record.get('exit_code', 'NOT RECORDED')}",
            f"code_commit={record.get('code_commit', 'NOT RECORDED')}",
            f"start_utc={record.get('start_utc', 'NOT RECORDED')}",
            f"end_utc={record.get('end_utc', 'NOT RECORDED')}",
            f"actual_worker_pid={record.get('actual_worker_pid', 'NOT RECORDED')}", "",
        ])
    (TASK / "COMMANDS.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_fixed_summary() -> None:
    """九条固定诊断只保留身份、事件、误差和标量结果，完整状态留本机。"""
    data = json.loads((TASK / "analysis/final_fixed_diagnostics.json").read_text(
        encoding="utf-8"))
    keys = (
        "trajectory_id", "selection", "training_seed", "model_transition", "task_profile",
        "environment_seed", "scenario_id", "experiment_code_commit", "registration_id",
        "branch", "status", "transitions", "warmup_control_transitions", "failure_type",
        "complete", "reward", "discounted_reward", "initial_goal_distance_m",
        "final_goal_distance_m", "goal_success_rewards", "executed_segment_count",
        "independent_observation_max_abs_error", "independent_command_max_abs_error",
        "reward_component_max_abs_error", "undiscounted_progress_telescoping_error",
        "actor_parameters_unchanged", "goal_event_consistent", "science_training_steps",
        "science_training_updates", "training_replay_writes", "actor_gradient_operations",
    )
    rows = []
    for case in data["cases"]:
        row = {key: case.get(key) for key in keys}
        closest = case.get("closest_approach") or {}
        entry = case.get("first_goal_entry") or {}
        row.update(closest_distance_m=closest.get("distance_m"),
                   closest_timestamp_s=closest.get("timestamp_s"),
                   first_goal_entry_timestamp_s=entry.get("timestamp_s"),
                   event_mismatch_count=len(case["event_mismatches"]),
                   boundary_subtypes=json.dumps(case.get("boundary_subtypes", [])),
                   reward_components=json.dumps(case["reward_components"], sort_keys=True))
        rows.append(row)
    if not rows:
        raise ValueError("没有已完成固定诊断，不能生成空公开总结。")
    destination = TASK / "public_evidence/fixed_diagnostics_summary.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """只在报告和精简CSV.gz已生成后制作显式待stage清单，不暂存或推送。"""
    required = [TASK / name for name in RESULT_PATHS
                if name not in ("COMMANDS.txt", "final_publication_paths.json",
                                "public_evidence/fixed_diagnostics_summary.csv")]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"先生成必要最终报告和精简文件；当前缺少：{missing}")
    records = _selected_commands()
    _write_fixed_summary()
    _write_commands(records)
    paths = [".gitignore", "docs/PROJECT_STATUS.md",
             *[f"{PREFIX}/{name}" for name in RESULT_PATHS]]
    for record in records:
        name = record["name"]
        for suffix in ("worker.json", "stdout.txt", "stderr.txt"):
            path = TASK / f"{name}.{suffix}"
            if path.is_file() and path.stat().st_size <= 100_000:
                paths.append(path.relative_to(ROOT).as_posix())
    paths = sorted(set(paths))
    metadata = []
    for relative in paths:
        path = ROOT / relative
        if relative.endswith("final_publication_paths.json"):
            continue
        size = path.stat().st_size
        if size > 500_000:
            raise ValueError(f"候选文件超过本轮500kB人工审阅界限：{relative}, {size}")
        metadata.append({"path": relative, "bytes": size})
    plan = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        purpose="explicit final R1 public stage candidates; no Git operations performed",
        current_public_code_commit="9029d60a3d88210e9113936b5da4f604bbf8c202",
        stage_paths=paths, file_sizes=metadata,
        measured_candidate_bytes_excluding_this_plan=sum(item["bytes"] for item in metadata),
        local_only=[f"{PREFIX}/{name}" for name in LOCAL_ONLY],
        original_commands_unchanged=True, scientific_files_modified=False,
        extra_file_digests_generated=False, git_add_commit_push_performed=False,
        note="Report claims remain those of actual R1 results; this is publication planning only.",
    )
    _write_json(TASK / "final_publication_paths.json", plan)
    print(json.dumps(dict(status="SMALL_FINAL_PUBLICATION_PLAN_CREATED", files=len(paths),
                          bytes_excluding_plan=plan[
                              "measured_candidate_bytes_excluding_this_plan"])))


if __name__ == "__main__":
    main()
