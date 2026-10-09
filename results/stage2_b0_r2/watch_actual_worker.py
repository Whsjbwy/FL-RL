"""只读持有真实计算进程句柄，记录其结束码，不以venv启动器代替。"""

import ctypes
import json
import sys
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    """不会启动、暂停或终止训练，也不改变checkpoint/日志。"""
    pid = int(sys.argv[1])
    output = Path(__file__).parent / 'actual_training_worker_exit.json'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    record = dict(status='WAITING', actual_computation_pid=pid,
                  monitored_at_utc=datetime.now(UTC).isoformat(),
                  mechanism='read-only Windows process handle; actual batch_state PID')
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    try:
        while True:
            status = kernel.WaitForSingleObject(handle, 1000)
            if status == 0:
                break
            if status != 258:
                raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        record.update(status='EXITED', actual_worker_exit_code=code.value,
                      exited_at_utc=datetime.now(UTC).isoformat())
        output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record), flush=True)
        return 0
    finally:
        kernel.CloseHandle(handle)


if __name__ == '__main__':
    raise SystemExit(main())
