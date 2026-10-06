"""B2射线与中心感知边界的独立几何测试。"""

from dataclasses import replace

import numpy as np
import pytest

from auv_risk_rl.frames import rotation_body_to_ned
from auv_risk_rl.sensors.rays import cast_sonar_rays, ray_directions_body
from auv_risk_rl.sensors.visibility import is_obstacle_geometrically_visible
from auv_risk_rl.types import AUVState, GroundTruthObstacleState


def own():
    """合法自身状态。"""
    return AUVState(np.array([20., 50., 20.]), 0., 0., .3, 0., 0.)


def sphere(center, radius=1., identity=1):
    """纯几何测试的物理球。"""
    return GroundTruthObstacleState(identity, np.asarray(center), np.zeros(3), radius)


def test_ray_01_04_empty(project_config):
    frame = cast_sonar_rays(own(), (), project_config.sensor)
    assert frame.ranges_m.shape == frame.valid_masks.shape == (45,)
    np.testing.assert_array_equal(frame.ranges_m, 25.)
    np.testing.assert_array_equal(frame.valid_masks, 0.)
    directions = ray_directions_body(project_config.sensor)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.)
    np.testing.assert_allclose(directions[22], [1, 0, 0], atol=1e-15)


@pytest.mark.parametrize('offset,visible', [(-1e-8, True), (0., True), (1e-8, False)])
@pytest.mark.parametrize('axis', ['horizontal', 'vertical', 'range'])
def test_ray_02_03_center_edges(project_config, offset, visible, axis):
    bearing = np.deg2rad(60.) + offset if axis == 'horizontal' else 0.
    elevation = np.deg2rad(30.) + offset if axis == 'vertical' else 0.
    distance = 25. + offset if axis == 'range' else 10.
    relative = distance * np.array([np.cos(elevation)*np.cos(bearing),
                                   np.cos(elevation)*np.sin(bearing), -np.sin(elevation)])
    detected, _ = is_obstacle_geometrically_visible(
        own(), sphere(own().position_ned_m + relative), project_config.sensor)
    assert detected == visible


@pytest.mark.parametrize('offset,mask', [(-1e-8, 1), (0., 1), (1e-8, 0)])
def test_ray_03_surface_range_edge(project_config, offset, mask):
    obstacle = sphere(own().position_ned_m + [26. + offset, 0., 0.])
    frame = cast_sonar_rays(own(), (obstacle,), project_config.sensor)
    assert frame.valid_masks[22] == mask
    assert frame.ranges_m[22] == pytest.approx(min(25., 25. + offset))


def test_ray_05_07_nearest_rotation(project_config):
    state = replace(own(), yaw_rad=.7, pitch_rad=.2)
    rotation = rotation_body_to_ned(.7, .2)
    obstacles = tuple(sphere(state.position_ned_m + rotation @ np.array([d, 0., 0.]),
                             identity=i) for i, d in enumerate([15., 10.]))
    frame = cast_sonar_rays(state, obstacles, project_config.sensor)
    assert frame.ranges_m[22] == pytest.approx(9.)
    assert frame.valid_masks[22] == 1


def test_ray_06_occlusion(project_config):
    from auv_risk_rl.sensors.occlusion import unoccluded_centers

    state = own()
    near = sphere(state.position_ned_m + [6., 0., 0.], identity=1)
    far = sphere(state.position_ned_m + [12., 0., 0.], identity=2)
    partial = sphere(state.position_ned_m + [12., 2.5, 0.], radius=2., identity=3)
    assert [o.obstacle_id for o in unoccluded_centers(state, (far,))] == [2]
    assert [o.obstacle_id for o in unoccluded_centers(state, (near, far))] == [1]
    assert [o.obstacle_id for o in unoccluded_centers(state, (near, partial))] == [1, 3]
    frame = cast_sonar_rays(state, (near, partial), project_config.sensor)
    assert frame.ranges_m[22] == pytest.approx(5.)
    assert frame.valid_masks[23] == 1
    assert frame.ranges_m[23] > 9.


def test_sensor_rng_namespaces():
    from auv_risk_rl.seeding import SeedManager

    first, second = SeedManager(41), SeedManager(41)
    first.get_rng('sensor_noise').normal(size=87)
    first.get_rng('dropout').random(17)
    for name in ('scenario', 'environment'):
        np.testing.assert_array_equal(first.get_rng(name).random(20),
                                      second.get_rng(name).random(20))
