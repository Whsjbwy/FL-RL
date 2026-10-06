"""只观察测试中的调用计数，不替代科学断言或改变计算。"""

import json
import os
from pathlib import Path

import torch

from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.local_navigation import LocalNavigationEnv
from auv_risk_rl.env.world import AUVWorld
from auv_risk_rl.rl.agent import OrdinarySACAgent

COUNTS: dict[str, int] = {}
ORIGINALS: list[tuple[type, str, object]] = []
UPDATE_NODES: dict[str, int] = {}
COMPLETE_UPDATE_NODES: dict[str, int] = {}
CURRENT_NODE = ""


def pytest_runtest_setup(item) -> None:
    """保留每个批输入更新的测试来源，避免把真实环境夹具混称合成数据。"""
    global CURRENT_NODE
    CURRENT_NODE = item.nodeid


def pytest_sessionstart(session) -> None:
    """调用和成功返回分别累计；嵌套World与环境计数不可相加。"""
    for cls, method, label in (
        (AUVWorld, "step", "world"), (LocalNavigationEnv, "step", "finite_env"),
        (B0NavigationEnv, "step", "b0_env"), (torch.optim.Adam, "step", "adam"),
        (torch.optim.SGD, "step", "sgd"), (OrdinarySACAgent, "update", "ordinary_update"),
    ):
        original = getattr(cls, method)

        def counted(*args, _fn=original, _label=label, **kwargs):
            """不拦截异常或修改返回值。"""
            COUNTS[_label + "_calls"] = COUNTS.get(_label + "_calls", 0) + 1
            explicit_batch = (_label == "ordinary_update"
                              and (len(args) > 1 and args[1] is not None
                                   or kwargs.get("batch") is not None))
            result = _fn(*args, **kwargs)
            COUNTS[_label + "_returns"] = COUNTS.get(_label + "_returns", 0) + 1
            if _label == "ordinary_update":
                COMPLETE_UPDATE_NODES[CURRENT_NODE] = (
                    COMPLETE_UPDATE_NODES.get(CURRENT_NODE, 0) + 1)
            if explicit_batch:
                key = "explicit_batch_complete_update_returns"
                COUNTS[key] = COUNTS.get(key, 0) + 1
                UPDATE_NODES[CURRENT_NODE] = UPDATE_NODES.get(CURRENT_NODE, 0) + 1
            return result

        ORIGINALS.append((cls, method, original))
        setattr(cls, method, counted)


def pytest_sessionfinish(session, exitstatus) -> None:
    """保存实际计数到独立目录，不覆盖上轮记录。"""
    for cls, method, original in ORIGINALS:
        setattr(cls, method, original)
    name = os.environ.get("AUV_COUNT_LABEL", "pytest_all")
    output = Path(session.config.rootpath) / "results/stage2_b0_mvp_v1"
    (output / f"{name}_counts.json").write_text(json.dumps(dict(
        counts=COUNTS, explicit_batch_update_nodes=UPDATE_NODES,
        complete_update_nodes=COMPLETE_UPDATE_NODES,
        pytest_exit_code=int(exitstatus),
        purpose="unit/integration tests, not scientific training",
        nested_counts_must_not_be_added=True,
        scientific_training_steps=0, scientific_training_updates=0,
    ), indent=2) + "\n", encoding="utf-8")
