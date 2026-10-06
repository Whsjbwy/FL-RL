"""目标设备的真实张量与普通SAC有限更新，不是导航科研训练。"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from auv_risk_rl.rl.agent import OrdinarySACAgent
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.training.config import B0HarnessConfig
from auv_risk_rl.training.harness import B0TrainingHarness, states_equal


def test_project_torch_declared_version() -> None:
    """项目声明的稳定Torch版本必须与本次验收环境对齐。"""
    assert torch.__version__.split("+")[0] == "2.11.0"


def test_target_device_tensor_and_complete_sac_update() -> None:
    """显式选择目标设备，验证真实前反向、四优化器和目标更新。"""
    device = os.environ.get("AUV_TARGET_DEVICE", "cpu")
    if device == "cuda":
        assert torch.cuda.is_available(), "目标GPU未就绪；不能静默换CPU"
    agent = OrdinarySACAgent(SACConfig(device=device), source_fingerprint="git:synthetic-device")
    generator = np.random.default_rng(820001)
    batch = dict(
        obs=generator.normal(0, 0.1, (256, 234)).astype(np.float32),
        next_obs=generator.normal(0, 0.1, (256, 234)).astype(np.float32),
        nominal_action=generator.uniform(-0.5, 0.5, (256, 3)).astype(np.float32),
        reward=generator.normal(0, 0.1, (256, 1)).astype(np.float32),
        terminated=np.zeros((256, 1), dtype=np.float32),
    )
    before = {name: [p.detach().clone() for p in getattr(agent, name).parameters()]
              for name in ("actor", "q1", "q2", "target_q1", "target_q2")}
    before_alpha = agent.log_alpha.detach().clone()
    tensor = torch.arange(32, device=device, dtype=torch.float32)
    assert float(torch.dot(tensor, tensor).cpu()) == sum(i * i for i in range(32))
    metrics = agent.update(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    for name, original in before.items():
        parameters = list(getattr(agent, name).parameters())
        assert all(p.dtype == torch.float32 and p.device.type == device for p in parameters)
        assert all(torch.isfinite(p).all() for p in parameters)
        assert any(not torch.equal(a, b) for a, b in zip(original, parameters, strict=True))
    assert not torch.equal(before_alpha, agent.log_alpha)
    assert all(np.isfinite(value) for value in metrics.values())
    assert agent.counters["gradient_updates"] == 1
    assert [agent.counters[name] for name in ("actor", "q1", "q2", "alpha")] == [1] * 4
    output = os.environ.get("AUV_DEVICE_EVIDENCE")
    if output:
        Path(output).write_text(json.dumps(dict(
            device=device, gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
            torch_version=torch.__version__, cuda_build=torch.version.cuda,
            complete_synthetic_updates=1, optimizer_steps=4, metrics=metrics,
            scientific_training_steps=0, scientific_training_updates=0,
        ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_same_device_checkpoint_preserves_adam_step_location(project_config, tmp_path) -> None:
    """非capturable Adam步数保持CPU，网络及一二阶矩保持目标设备。"""
    from auv_risk_rl.env.scenario_generator import load_training_scenario_config

    device = os.environ.get("AUV_TARGET_DEVICE", "cpu")
    scenario = load_training_scenario_config(
        Path(__file__).resolve().parents[1] / "configs/train_scenario_v1.yaml")
    config = B0HarnessConfig(run_kind="engineering_smoke", transition_budget=8,
                             sac=replace(SACConfig(), device=device))
    left = B0TrainingHarness(config, project_config, scenario, code_version="device-fixture")
    batch = dict(obs=np.zeros((256, 234), np.float32),
                 next_obs=np.full((256, 234), 0.1, np.float32),
                 nominal_action=np.full((256, 3), 0.2, np.float32),
                 reward=np.ones((256, 1), np.float32),
                 terminated=np.zeros((256, 1), np.float32))
    left.agent.update(batch)
    path = tmp_path / "trusted_device_checkpoint.pt"
    left.save_checkpoint(path)
    right = B0TrainingHarness(config, project_config, scenario, code_version="device-fixture")
    right.load_checkpoint(path, trusted_local=True)
    for optimizer in right.agent.optimizers.values():
        for state in optimizer.state.values():
            assert state["step"].device.type == "cpu"
            assert state["exp_avg"].device.type == device
            assert state["exp_avg_sq"].device.type == device
    assert states_equal(left.state_dict(), right.state_dict())
    left.agent.update(batch)
    right.agent.update(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    assert states_equal(left.state_dict(), right.state_dict())
