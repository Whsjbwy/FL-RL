"""复用既有环境的简短只读核对；不安装依赖、不运行训练或正式benchmark。"""

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
git = ['git', '-c', f'safe.directory={ROOT.as_posix()}']
remote = subprocess.run([*git, 'ls-remote', 'origin',
                         'refs/heads/codex/stage2-b0-mvp-v1'], cwd=ROOT,
                        capture_output=True, text=True, check=False)
pip = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                     capture_output=True, text=True, check=False)
gpu = subprocess.run(['nvidia-smi',
                      '--query-gpu=name,driver_version,memory.total,memory.used',
                      '--format=csv,noheader'], capture_output=True, text=True, check=False)
x = torch.ones(2, device='cuda')
value = (x + x).cpu().tolist()
torch.cuda.synchronize()
record = dict(recorded_at_utc=datetime.now(UTC).isoformat(), interpreter=sys.executable,
              python=sys.version, torch=str(torch.__version__), torch_file=torch.__file__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
              tensor_result=value, pip_check_exit=pip.returncode, pip_check=pip.stdout.strip(),
              remote_v1_exit=remote.returncode, remote_v1=remote.stdout.strip(),
              remote_error=remote.stderr.strip(), nvidia_smi_exit=gpu.returncode,
              gpu_summary=gpu.stdout.strip(), free_disk_bytes=shutil.disk_usage(ROOT).free,
              environment_reinstalled=False)
path = Path(__file__).parent / 'environment.json'
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, indent=2))
if pip.returncode or remote.returncode or value != [2., 2.]:
    raise SystemExit(1)
