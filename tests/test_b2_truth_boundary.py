"""B2静态边界和动态毒化对象审计；物理传播允许真值，编码/估计不允许。"""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auv_risk_rl.env.observation import ObservationBuilder
from auv_risk_rl.runtime.local_perception import PerceptionSession
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.sensors.rays import cast_sonar_rays
from auv_risk_rl.types import AUVState


class PoisonObstacle:
    """可读当前几何，但任何真速度或未来真值访问立即失败。"""

    obstacle_id = 1
    position_ned_m = np.array([30., 50., 20.])
    radius_m = 1.

    @property
    def velocity_ned_mps(self):
        raise AssertionError('truth velocity accessed')

    @property
    def future_position_ned_m(self):
        raise AssertionError('future truth accessed')


def test_obs_12_13_warm_02_poison_history(project_config):
    config = replace(project_config, sensor=replace(project_config.sensor, dropout_probability=0.))
    own = AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)
    session = PerceptionSession(config, SeedManager(57), {1: 1.})
    for tick in range(6):
        world = SimpleNamespace(timestamp_s=tick*.2, auv_state=own,
                                obstacle_states=(PoisonObstacle(),))
        rays = session.capture(world, tick)
        tracks = session.tracks(tick*.2)
        if tick == 0:
            np.testing.assert_array_equal(tracks[0].track_state.state_mean_ned[3:], 0)
    output = ObservationBuilder(config).build(own, np.array([80., 50., 20.]),
                                              np.zeros(3), 1000, rays, tracks, 1.)
    assert output.shape == (234,) and np.all(np.isfinite(output))
    assert len(session.generated) == 6
    poisoned_rays = cast_sonar_rays(own, (PoisonObstacle(),), config.sensor)
    np.testing.assert_array_equal(rays.ranges_m, poisoned_rays.ranges_m)


def test_truth_static_forbidden_dependencies():
    root = Path(__file__).resolve().parents[1]/'src/auv_risk_rl'
    paths = [root/'env/observation.py', root/'env/local_task.py']
    paths += [p for directory in ('tracking', 'prediction', 'risk', 'safety')
              for p in (root/directory).glob('*.py')]
    forbidden = {'GroundTruthObstacleState', 'AUVWorld', 'velocity_ned_mps',
                 'future_position_ned_m', 'obstacle_states'}
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not names & forbidden, (path, names & forbidden)


def test_sensor_no_truth_attributes_static():
    root = Path(__file__).resolve().parents[1]/'src/auv_risk_rl/sensors'
    for name in ('sonar.py', 'occlusion.py', 'rays.py'):
        tree = ast.parse((root/name).read_text(encoding='utf-8-sig'))
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not attributes & {'velocity_ned_mps', 'future_position_ned_m'}
