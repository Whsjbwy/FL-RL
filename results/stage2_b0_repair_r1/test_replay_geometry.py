"""R1外部观察器的独立手算参考；不运行环境、不做学习更新。"""

import math

import numpy as np
import pytest
from replay_diagnostics import independent_rotation, nearest_goal, scalar_cross, sphere_entry


@pytest.mark.parametrize('start,end,expected', [
    (-3., 3., 1/6), (3., -3., 1/6), (-1., 3., 0.), (3., 4., None),
])
def test_sphere_entry_scalar(start: float, end: float, expected: float | None) -> None:
    """2m球的左右线段首根按解析数值验证，合法失败也有参考。"""
    actual = sphere_entry(np.array([start, 0., 0.]), np.array([end, 0., 0.]),
                          np.zeros(3), 2.)
    assert actual == expected


def test_tangent() -> None:
    """切线恰在中点触及闭球，不能漏报。"""
    assert sphere_entry(np.array([-3., 2., 0.]), np.array([3., 2., 0.]),
                        np.zeros(3), 2.) == .5


def test_boundary_scalar() -> None:
    """终点恰在闭界合法；未提交提议越界才定位fraction。"""
    assert scalar_cross(.4, .6, -.5, .5) == (.5, 'upper')
    assert scalar_cross(-.4, -.6, -.5, .5) == (.5, 'lower')
    assert scalar_cross(.4, .5, -.5, .5) is None


def test_projection() -> None:
    """独立线段投影最小距离，不仅检查控制节点。"""
    distance, fraction = nearest_goal(np.array([-3., 4., 0.]), np.array([3., 4., 0.]),
                                      np.zeros(3))
    assert distance == 4.
    assert fraction == .5


def test_ned_pitch_sign() -> None:
    """正pitch将Body前向速度转成负Down，按手算三角式。"""
    velocity = independent_rotation(0., math.pi/6) @ np.array([1., 0., 0.])
    np.testing.assert_allclose(velocity, [math.sqrt(3)/2, 0., -.5], atol=1e-10, rtol=0.)
