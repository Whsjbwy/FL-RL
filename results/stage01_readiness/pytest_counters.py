"""验收期计数插件：区分测试中的环境调用、优化器操作与科研训练。"""

import json
from pathlib import Path

import torch

from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.world import AUVWorld

COUNTS = {
    "world_step_calls": 0,
    "world_step_returns": 0,
    "local_env_step_calls": 0,
    "local_env_step_returns": 0,
    "adam_steps_in_unit_tests": 0,
    "sgd_steps_in_unit_tests": 0,
}
ORIGINALS = []


def pytest_sessionstart(session):
    """原函数原样执行，仅累计调用与成功返回，不影响数值结果。"""

    for cls, method, call_key, return_key in (
        (AUVWorld, "step", "world_step_calls", "world_step_returns"),
        (LocalNavigationEnv, "step", "local_env_step_calls", "local_env_step_returns"),
        (torch.optim.Adam, "step", "adam_steps_in_unit_tests", None),
        (torch.optim.SGD, "step", "sgd_steps_in_unit_tests", None),
    ):
        original = getattr(cls, method)

        def counted(*args, _fn=original, _call=call_key, _return=return_key, **kwargs):
            """保持原函数的异常、返回值和梯度语义。"""

            COUNTS[_call] += 1
            result = _fn(*args, **kwargs)
            if _return is not None:
                COUNTS[_return] += 1
            return result

        ORIGINALS.append((cls, method, original))
        setattr(cls, method, counted)


def pytest_sessionfinish(session, exitstatus):
    """保存实际测试计数；这些优化器操作不是导航科研训练结果。"""

    for cls, method, original in ORIGINALS:
        setattr(cls, method, original)
    output = Path(session.config.rootpath) / "results/stage01_readiness/pytest_counters.json"
    output.write_text(json.dumps({
        "purpose": "unit/integration acceptance only; no scientific training task",
        "pytest_exit_code": int(exitstatus),
        "counts": COUNTS,
        "nested_world_and_local_env_calls_must_not_be_added": True,
        "scientific_rl_training_steps": 0,
        "scientific_gradient_updates": 0,
    }, indent=2), encoding="utf-8")
