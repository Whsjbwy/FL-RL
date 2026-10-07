"""记录本轮命令原始输出、退出码和实际 Git 代码版本，不生成文件摘要。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    """在同一工作区运行命令，保留失败记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(__file__).resolve().parent
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("缺少命令")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPYCACHEPREFIX", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root),
                                        env.get("PYTHONPATH", "")))
    git = "D:/Program Files/Git/cmd/git.exe"
    version = subprocess.run(
        [git, "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root, capture_output=True, text=True, check=False,
    ).stdout.strip()
    start = datetime.now(UTC).isoformat()
    with (output / f"{args.name}.stdout.txt").open("wb") as stdout, (
        output / f"{args.name}.stderr.txt"
    ).open("wb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        identity_path = output / f'{args.name}.worker.json'
        identity = dict(status='RUNNING', worker_pid=process.pid, command=command,
                        start_utc=start, code_commit=version,
                        note='PID is the actual direct child, not the outer launcher.')
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
        print(json.dumps(identity, ensure_ascii=False), flush=True)
        return_code = process.wait()
        identity.update(status='EXITED', actual_worker_exit_code=return_code,
                        end_utc=datetime.now(UTC).isoformat())
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
    record = dict(name=args.name, command=command, cwd=str(root),
                  start_utc=start, end_utc=datetime.now(UTC).isoformat(),
                  exit_code=return_code, actual_worker_pid=process.pid, code_commit=version,
                  recording_interpreter=sys.executable)
    with (output / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    for channel, limit in (("stdout", 5000), ("stderr", 2000)):
        print((output / f"{args.name}.{channel}.txt").read_text(
            encoding="utf-8", errors="replace")[-limit:])
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
