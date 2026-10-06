"""仅执行预登记的两条264步B0工程轨迹，持久累计预算并检验完整恢复。"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def compare_state(left: Any, right: Any, path: str = "state") -> list[str]:
    """离散/RNG精确，浮点使用实施前登记容差，不靠摘要作比较。"""
    mismatches = []
    if type(left) is not type(right):
        return [path + ":type"]
    if isinstance(left, torch.Tensor):
        same = (torch.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True)
                if left.is_floating_point() else torch.equal(left, right))
        return [] if same else [path]
    if isinstance(left, np.ndarray):
        same = (np.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True)
                if np.issubdtype(left.dtype, np.floating)
                else np.array_equal(left, right))
        return [] if same else [path]
    if isinstance(left, np.random.Generator):
        return compare_state(left.bit_generator.state, right.bit_generator.state, path + ".rng")
    if isinstance(left, np.random.SeedSequence):
        return compare_state(left.state, right.state, path + ".seed_sequence")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ":keys"]
        for key in left:
            mismatches.extend(compare_state(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, list | tuple):
        if len(left) != len(right):
            return [path + ":length"]
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            mismatches.extend(compare_state(a, b, f"{path}[{i}]"))
    elif hasattr(left, "__dict__"):
        return compare_state(vars(left), vars(right), path)
    elif isinstance(left, float):
        if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7):
            if not (math.isnan(left) and math.isnan(right)):
                return [path]
    elif left != right:
        return [path]
    return mismatches


def main() -> int:
    """预算计入失败尝试；任何失败立即结束，不追加运行直到有利结果。"""
    from scripts.run_b0_training import git_code_version, read_run_config

    from auv_risk_rl.config import load_project_config
    from auv_risk_rl.env.b0_navigation import B0NavigationEnv
    from auv_risk_rl.env.scenario_generator import load_training_scenario_config
    from auv_risk_rl.env.world import AUVWorld
    from auv_risk_rl.rl.agent import OrdinarySACAgent
    from auv_risk_rl.training.harness import B0TrainingHarness, states_equal

    output = Path(__file__).parent / "engineering_smoke"
    output.mkdir(exist_ok=False)
    budget_path = Path(__file__).parent / "engineering_budget.json"
    if budget_path.exists():
        raise RuntimeError("已有工程预算记录，不允许静默重置计算次数")
    budget = dict(env_transition_attempts=0, env_transitions=0, complete_update_attempts=0,
                  complete_updates=0, world_step_returns=0,
                  env_transition_ceiling=2048, complete_update_ceiling=128)

    def save_budget() -> None:
        """每次实际操作前后落盘，故障仍保留已经消耗的预算。"""
        budget_path.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")

    original_step, original_update, original_world = (
        B0NavigationEnv.step, OrdinarySACAgent.update, AUVWorld.step)

    def env_step(*args, **kwargs):
        """工程验证环境也计入保守总上限，不与物理小步混算。"""
        if budget["env_transition_attempts"] >= budget["env_transition_ceiling"]:
            raise RuntimeError("本轮工程真实转移预算已耗尽")
        budget["env_transition_attempts"] += 1
        save_budget()
        result = original_step(*args, **kwargs)
        budget["env_transitions"] += 1
        save_budget()
        return result

    def update(*args, **kwargs):
        """一次完整SAC更新与四次Adam操作分开计数。"""
        if budget["complete_update_attempts"] >= budget["complete_update_ceiling"]:
            raise RuntimeError("本轮工程更新预算已耗尽")
        budget["complete_update_attempts"] += 1
        save_budget()
        result = original_update(*args, **kwargs)
        budget["complete_updates"] += 1
        save_budget()
        return result

    def world_step(*args, **kwargs):
        """包含warm-up的World调用单独累计，不与环境转移相加。"""
        result = original_world(*args, **kwargs)
        budget["world_step_returns"] += 1
        return result

    B0NavigationEnv.step, OrdinarySACAgent.update, AUVWorld.step = env_step, update, world_step
    save_budget()
    try:
        config, raw = read_run_config(ROOT / "configs/stage2_b0_engineering.yaml")
        assert config.sac.device == "cuda" and torch.cuda.is_available()
        code_version = git_code_version()
        project = load_project_config(ROOT / raw["project_config"])
        scenario = load_training_scenario_config(ROOT / raw["scenario_config"])

        def create(name: str) -> B0TrainingHarness:
            """固定同初始化及独立输出，不使用科研seed。"""
            directory = output / name
            directory.mkdir()

            def log(kind: str, record: dict[str, Any]) -> None:
                """日志是实际测量，不将短工程行为解释为学习效果。"""
                with (directory / f"{kind}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

            return B0TrainingHarness(config, project, scenario, code_version=code_version,
                                     log_sink=log)

        continuous = create("continuous")
        initial = continuous.agent.state_dict()
        continuous.run(checkpoint_callback=lambda current: current.save_checkpoint(
            output / "continuous/latest_resume.pt"))
        split = create("split")
        assert states_equal(initial, split.agent.state_dict())
        for _ in range(256):
            split.step()
        pending_reset = [slot["needs_reset"] for slot in split.slots]
        split.save_checkpoint(output / "split/latest_resume.pt")
        resumed = create("resumed_256")
        resumed.load_checkpoint(output / "split/latest_resume.pt", trusted_local=True)
        for _ in range(4):
            resumed.step()
        active_steps = [slot["steps"] for slot in resumed.slots]
        resumed.save_checkpoint(output / "resumed_256/latest_resume.pt")
        final = create("resumed_260")
        final.load_checkpoint(output / "resumed_256/latest_resume.pt", trusted_local=True)
        final.run()
        torch.cuda.synchronize()
        discrepancies = compare_state(continuous.state_dict(), final.state_dict())
        if discrepancies:
            raise AssertionError({"resume_discrepancies": discrepancies[:30]})
        assert continuous.transitions == final.transitions == 264
        assert continuous.summary()["updates"] == final.summary()["updates"] == 9
        assert budget["complete_updates"] == 18
        report = dict(status="PASS", purpose="ENGINEERING_ONLY_NOT_A_SCIENTIFIC_RESULT",
                      code_version=code_version, torch_version=torch.__version__,
                      config=config.to_dict(), budget=budget,
                      continuous=continuous.summary(), resumed=final.summary(),
                      pending_reset_at_256=pending_reset, active_steps_at_260=active_steps,
                      full_state_exact_equal=states_equal(
                          continuous.state_dict(), final.state_dict()),
                      full_state_registered_tolerance_equal=True,
                      evaluation_training_state_preserved=True,
                      scientific_training_steps=0, scientific_training_updates=0,
                      local_stage2="NOT_RUN")
        (output / "smoke_resume.json").write_text(json.dumps(report, ensure_ascii=False,
                                                            indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        (output / "failure.json").write_text(json.dumps(dict(
            status="FAIL_OR_BLOCKED", traceback=traceback.format_exc(), budget=budget,
            scientific_training_steps=0, scientific_training_updates=0,
        ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        B0NavigationEnv.step = original_step
        OrdinarySACAgent.update = original_update
        AUVWorld.step = original_world
        save_budget()


if __name__ == "__main__":
    raise SystemExit(main())
