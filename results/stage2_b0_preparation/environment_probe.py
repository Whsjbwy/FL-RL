"""读取本机项目环境身份与磁盘空间，不修改依赖。"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


def main() -> None:
    """保存安装前或安装后的真实环境。"""
    record = dict(
        time_utc=datetime.now(UTC).isoformat(), execution_location="USER_LOCAL_WINDOWS",
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        project_venv=sys.prefix, torch_version=torch.__version__, torch_file=torch.__file__,
        cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        devices=[dict(name=torch.cuda.get_device_name(i),
                      capability=torch.cuda.get_device_capability(i))
                 for i in range(torch.cuda.device_count())],
        dependencies={name: importlib.metadata.version(name) for name in
                      ("pip", "numpy", "scipy", "PyYAML", "pytest", "ruff")},
        disk_free_bytes={drive: shutil.disk_usage(drive).free for drive in ("C:/", "D:/", "F:/")},
        pip_version=subprocess.run([sys.executable, "-m", "pip", "--version"],
                                   capture_output=True, text=True, check=False).stdout.strip(),
    )
    for label, command in (("pip_check", [sys.executable, "-m", "pip", "check"]),
                           ("nvidia_smi", ["nvidia-smi",
                                           "--query-gpu=name,driver_version,memory.total",
                                           "--format=csv"])):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        record[label] = dict(exit_code=result.returncode, stdout=result.stdout,
                             stderr=result.stderr)
    output = Path(__file__).parent / f"environment_{sys.argv[1]}.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
