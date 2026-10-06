"""B2动作、任务reward、即时成本和终止协议测试。"""

import numpy as np
import pytest

from auv_risk_rl.env.local_task import (
    LocalTaskConfig,
    command_to_normalized,
    normalized_to_command,
    task_reward,
)


@pytest.mark.parametrize('action', [[-1., -1., -1.], [1., 1., 1.], [.2, -.4, .7]])
def test_action_roundtrip(project_config, action):
    command = normalized_to_command(np.array(action), project_config.dynamics)
    np.testing.assert_allclose(command_to_normalized(command, project_config.dynamics), action,
                               atol=1e-15)
    assert command.surge_speed_command_mps == pytest.approx(.3+.6*(action[0]+1))
    assert command.yaw_rate_command_rad_s == pytest.approx(.35*action[1])
    assert command.pitch_rate_command_rad_s == pytest.approx(.25*action[2])


@pytest.mark.parametrize('action', [[np.nan, 0, 0], [0, 0], [1.01, 0, 0]])
def test_action_invalid(project_config, action):
    with pytest.raises(ValueError):
        normalized_to_command(np.array(action), project_config.dynamics)


def test_env_02_03_reward_native_formula():
    task = LocalTaskConfig()
    reward, parts = task_reward(10., 9.8, True, .05, .2, np.zeros(3),
                               np.array([.5, -.5, 1.]), task)
    assert parts == pytest.approx(dict(progress=.2, goal=100., time=-.0025, smoothness=-.03))
    assert reward == pytest.approx(100.1675)
    other, other_parts = task_reward(10., 9.8, True, .05, .2, np.zeros(3), np.zeros(3), task)
    assert other_parts['smoothness'] == 0
    assert other - reward == pytest.approx(.03)


@pytest.mark.parametrize('rejected', [False, True])
@pytest.mark.parametrize('event', ['none', 'success', 'collision', 'boundary', 'timeout'])
def test_env_04_cost_components(rejected, event):
    from auv_risk_rl.env.local_task import instantaneous_cost

    components = instantaneous_cost(rejected, event)
    assert components['c_risk'] == int(rejected)
    assert components['c_real'] == int(event == 'collision')
    assert components['f_t'] == int(event in ('collision', 'boundary'))
    assert components['c_train'] == max(int(rejected), components['f_t'])
    if event == 'collision':
        assert instantaneous_cost(rejected, event, True)['c_real'] == 0


def test_env_12_independent_budgets(project_config):
    from dataclasses import replace

    task = LocalTaskConfig(d_C=.02)
    risk = replace(project_config.risk, short_horizon_risk_budget=.1)
    assert task.d_C == .02 and risk.short_horizon_risk_budget == .1
    assert LocalTaskConfig().d_C == .05
    assert project_config.risk.short_horizon_risk_budget == .05


@pytest.mark.parametrize('event,name', [
    ('none', 'none'), ('success', 'goal_success'), ('collision', 'collision'),
    ('boundary', 'operational_boundary_failure'), ('timeout', 'task_horizon'),
])
@pytest.mark.parametrize('cutoff', [False, True])
def test_env_05_10_termination_mapping(event, name, cutoff):
    from auv_risk_rl.env.local_task import termination_flags

    terminated, truncated, failure = termination_flags(event, cutoff)
    assert terminated == (event != 'none')
    assert truncated == cutoff
    assert failure == ('external_truncation' if cutoff and event == 'none' else name)
