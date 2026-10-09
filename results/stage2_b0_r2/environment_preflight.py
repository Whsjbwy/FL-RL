"""R2只读环境和资源核对；仅生成本任务结果，不安装、不训练、不计算摘要。"""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import winreg
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]


def write(name: str, value: object) -> None:
    """仅保存本轮现场记录；不修改已有历史证据。"""
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")


def memory_status() -> dict:
    """使用系统原生接口读取RAM，不安装psutil。"""
    class MemoryStatus(ctypes.Structure):
        """Windows MEMORYSTATUSEX的公开结构。"""
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "total", "available", "total_page", "available_page",
                        "total_virtual", "available_virtual", "extended")]]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return dict(total_bytes=status.total, available_bytes=status.available,
                memory_load_percent=status.load, source="Windows GlobalMemoryStatusEx")


def main() -> None:
    """按实际退出码记录版本、GPU小张量和依赖；估算空间而不分配Replay。"""
    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    def run(name: str, command: list[str]) -> subprocess.CompletedProcess:
        """真实执行一个命令，保留分离的stdout/stderr和开始结束时间。"""
        start = datetime.now(UTC).isoformat()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False,
                                env={**os.environ, "PYTHONUTF8": "1"})
        (OUT / f"{name}.stdout.txt").write_bytes(result.stdout)
        (OUT / f"{name}.stderr.txt").write_bytes(result.stderr)
        records.append(dict(name=name, command=command, cwd=str(ROOT), start_utc=start,
                            end_utc=datetime.now(UTC).isoformat(), exit_code=result.returncode))
        write("environment_commands.json", records)
        return result

    git_results = {}
    for name, arguments in (
        ("branch", ["branch", "--show-current"]), ("status", ["status", "--short"]),
        ("head", ["rev-parse", "HEAD"]), ("log", ["log", "-6", "--oneline"]),
        ("remote", ["remote", "-v"]),
        ("r1_local_ref", ["rev-parse", "refs/heads/codex/stage2-b0-repair-r1-public"]),
        ("receipt", ["show", "--stat", "--oneline", "f476b1b"]),
    ):
        result = run(f"git_{name}", GIT + arguments)
        git_results[name] = dict(exit_code=result.returncode,
                                 output=result.stdout.decode("utf-8", errors="replace").strip())
    identity_code = (
        "import sys,platform,json,importlib.metadata as m,torch; "
        "x=torch.ones(8,device='cuda'); y=(x*x).sum(); torch.cuda.synchronize(); "
        "print(json.dumps(dict(executable=sys.executable,python=sys.version,"
        "platform=platform.platform(),pytorch=str(torch.__version__),"
        "torch_file=torch.__file__,cuda_build=torch.version.cuda,"
        "cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0),"
        "device_capability=list(torch.cuda.get_device_capability(0)),"
        "cuda_smoke_result=y.item(),cuda_synchronized=True,"
        "dependencies={k:m.version(k) for k in "
        "['numpy','scipy','PyYAML','pytest','ruff','torch']}),"
        "indent=2))"
    )
    identity_run = run("python_cuda_identity", [sys.executable, "-B", "-c", identity_code])
    if identity_run.returncode:
        raise RuntimeError("现有Python/CUDA小张量检查失败，原始日志已保存。")
    identity = json.loads(identity_run.stdout.decode("utf-8"))
    dependencies = run("pip_check", [sys.executable, "-B", "-m", "pip", "check"])
    gpu = run("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                            "memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"])
    gpu_apps = run("nvidia_compute_processes", ["nvidia-smi", "--query-compute-apps=pid,"
                                              "process_name,used_memory",
                                              "--format=csv,noheader,nounits"])
    cpu_key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cpu_key) as key:
        cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    ram = memory_status()
    disks = {drive: dict(zip(("total_bytes", "used_bytes", "free_bytes"),
                             shutil.disk_usage(drive), strict=True))
             for drive in ("C:/", "D:/", "F:/")}
    # 仅导入已存在的schema常量；不创建Replay、不实例化Agent、不加载checkpoint。
    sys.path.insert(0, str(ROOT / "src"))
    from auv_risk_rl.rl.replay import FIELDS

    field_bytes = {name: math.prod(shape) * np.dtype(dtype).itemsize
                   for name, (shape, dtype) in FIELDS.items()}
    row_bytes = sum(field_bytes.values())
    slots = math.ceil(300_000 / 4096) * 4096
    raw_300k = row_bytes * 300_000
    allocated_300k = row_bytes * slots
    observed = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
                for path in sorted((ROOT / "results/stage2_b0_mvp_v1").glob(
                    "seed_*/latest_resume.pt"))]
    observed_checkpoint_upper = max(item["bytes"] for item in observed)
    six_latest = 6 * observed_checkpoint_upper
    one_atomic_temporary = observed_checkpoint_upper
    # 空间估算分项明确：日志1GiB/run为工程保守预留，不是实测或科学参数。
    logs_reserved = 6 * 1024**3
    small_model_count_estimate = 6 * 12
    # R2保存Actor+四个Q网络；V1 Actor-only快照不可用作本项估算。
    known_small = sorted((ROOT / "results/stage2_b0_repair_r1/models").glob("*.pt"))
    small_model_upper = max(path.stat().st_size for path in known_small)
    models_reserved = small_model_count_estimate * small_model_upper
    planned = six_latest + one_atomic_temporary + logs_reserved + models_reserved
    resources = dict(
        observed_at_utc=datetime.now(UTC).isoformat(), cpu_model=cpu_model,
        logical_cores=os.cpu_count(), ram=ram, disks=disks,
        nvidia_smi=dict(exit_code=gpu.returncode, output=gpu.stdout.decode().strip(),
                        memory_units="MiB as reported by nvidia-smi",
                        gpu_utilization_is_observed_not_assumed_idle=True),
        nvidia_compute_processes=dict(exit_code=gpu_apps.returncode,
                                      output=gpu_apps.stdout.decode(errors="replace").strip()),
        replay_schema=dict(source="src/auv_risk_rl/rl/replay.py:FIELDS",
                           field_bytes=field_bytes, bytes_per_transition=row_bytes,
                           raw_300k_bytes=raw_300k, allocated_slots_at_300k=slots,
                           allocated_300k_array_bytes=allocated_300k,
                           raw_500k_bytes=500_000 * row_bytes,
                           replay_was_allocated=False),
        six_run_storage_estimate=dict(
            method="six independent 300k resumes; sequential run and one atomic temporary",
            historical_300k_checkpoint_file_sizes=observed,
            latest_checkpoint_upper_bytes_per_run=observed_checkpoint_upper,
            six_latest_checkpoints_bytes=six_latest,
            one_atomic_write_temporary_bytes=one_atomic_temporary,
            log_reserve_bytes=logs_reserved, log_reserve_is_estimate=True,
            small_model_count_estimate=small_model_count_estimate,
            observed_small_model_bytes=small_model_upper,
            observed_small_model_schema="R1 Actor plus q1/q2/target_q1/target_q2 and alpha",
            observed_small_model_files=[path.relative_to(ROOT).as_posix() for path in known_small],
            small_models_reserved_bytes=models_reserved,
            estimated_additional_bytes=planned, estimated_additional_gib=planned / 1024**3,
            available_workspace_bytes=disks["D:/"]["free_bytes"],
            workspace_margin_after_estimate_bytes=disks["D:/"]["free_bytes"] - planned,
            sufficient_for_estimate=disks["D:/"]["free_bytes"] > planned + 8 * 1024**3,
            ram_meets_existing_5gib_entry_requirement=ram["available_bytes"] >= 5 * 1024**3,
            checkpoint_pickle_overhead_included_via_actual_historical_size=True,
            concurrent_six_run_ram_not_assumed=True,
            no_runtime_or_performance_claim=True),
        historical_data_deleted=False, virtual_environment_copied=False,
        new_dependencies_installed=False, hashes_computed=False,
        scientific_training_steps=0, scientific_gradient_updates=0,
        initial_probe_limitations=["CIM unavailable to sandbox", "psutil is not installed"],
    )
    identity.update(observed_at_utc=datetime.now(UTC).isoformat(), git=git_results,
                    pip_check_exit_code=dependencies.returncode,
                    pip_check_output=dependencies.stdout.decode().strip(),
                    existing_environment_reused=True, new_dependencies_installed=False,
                    cuda_probe="8 float32 elements, forward only, synchronize; not training",
                    launcher_warning="Existing venv launcher warns about F:/Anaconda3 path; "
                                     "actual interpreter/dependencies/CUDA checks completed.")
    remote_record = OUT / "git_remote_heads.command.json"
    remote_output = OUT / "git_remote_heads.stdout.txt"
    if remote_record.is_file() and remote_output.is_file():
        remote = json.loads(remote_record.read_text(encoding="utf-8-sig"))
        remote_lines = remote_output.read_text(encoding="utf-8-sig").splitlines()
        remote["observed_remote_refs"] = dict(line.split()[::-1] for line in remote_lines
                                             if line.strip())
        remote["local_r1_receipt_not_on_remote_r1"] = (
            git_results["r1_local_ref"]["output"] != remote["observed_remote_refs"].get(
                "refs/heads/codex/stage2-b0-repair-r1-public"))
        identity["remote_ref_query"] = remote
    write("environment_identity.json", identity)
    write("resources_preflight.json", resources)
    print(json.dumps(dict(environment_identity="WRITTEN", resources_preflight="WRITTEN",
                          pip_check_exit_code=dependencies.returncode,
                          disk_sufficient=resources["six_run_storage_estimate"][
                              "sufficient_for_estimate"],
                          ram_available_bytes=ram["available_bytes"])))


if __name__ == "__main__":
    main()
