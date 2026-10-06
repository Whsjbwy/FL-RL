"""
简化声呐检测生成模块。

功能：
1. 对几何可见目标应用独立 dropout；
2. 在 Body 系笛卡尔相对位置上加入冻结高斯测量噪声；
3. 显式写入测量发生时刻与到达时刻；
4. 不向检测对象写入障碍真实速度或未来状态。
"""

from __future__ import annotations

import numpy as np

from auv_risk_rl.config import DynamicsConfig, SensorConfig
from auv_risk_rl.sensors.visibility import is_obstacle_geometrically_visible
from auv_risk_rl.types import AUVState, GroundTruthObstacleState, SensorDetection


def generate_sonar_detections(
    auv_state: AUVState,
    obstacle_states: tuple[GroundTruthObstacleState, ...],
    measurement_timestamp_s: float,
    sensor_config: SensorConfig,
    dynamics_config: DynamicsConfig,
    sensor_rng: np.random.Generator,
    *,
    measurement_control_tick: int | None = None,
    dropout_rng: np.random.Generator | None = None,
) -> tuple[SensorDetection, ...]:
    """
    生成一个控制时刻的简化声呐检测。

    对应技术协议：
        Eq. (16)–(18) 与第 9 章 Dropout/Delay 规则。

    参数：
        auv_state:
            测量发生时刻 AUV 状态；位置 NED。
        obstacle_states:
            同时刻障碍 Ground Truth；仅本传感仿真前端允许访问。
        measurement_timestamp_s:
            采样发生世界时刻，单位 s。
        sensor_config:
            量程、FOV、Body 噪声和延迟步数。
        dynamics_config:
            用 control_dt_s 将延迟控制步数换算成到达时间。
        sensor_rng:
            SeedManager 提供的显式 sensor 随机流。
        dropout_rng:
            B2可提供独立漏检流；省略时保留Stage0原有随机序列兼容性。
        measurement_control_tick:
            主运行链传入的整数采样控制步；到达步等于它加固定延迟步数。
            旧调用省略时由队列的秒制兼容入口转换，不用于新主链。

    返回：
        SensorDetection 元组；相对位置 shape=(3,) m，协方差 shape=(3,3) m^2，Body。

    关键假设：
        A1、A5、A6；噪声为零均值笛卡尔高斯，dropout 与测量噪声独立。

    重要限制：
        该函数输入真值仅用于生成测量；输出不包含真实速度、未来位置或 Oracle 信息。
    """

    if not np.isfinite(measurement_timestamp_s):
        raise ValueError("measurement_timestamp_s 必须有限。")
    if measurement_control_tick is not None and (
        isinstance(measurement_control_tick, bool)
        or not isinstance(measurement_control_tick, int | np.integer)
        or measurement_control_tick < 0
    ):
        raise ValueError("measurement_control_tick 必须是非负整数。")
    arrival_control_tick = (
        None if measurement_control_tick is None
        else int(measurement_control_tick) + sensor_config.measurement_delay_control_steps
    )
    measurement_std_body_m = np.asarray(sensor_config.measurement_std_body_m, dtype=np.float64)
    measurement_covariance_body_m2 = np.diag(measurement_std_body_m**2)
    arrival_timestamp_s = (
        measurement_timestamp_s
        + sensor_config.measurement_delay_control_steps * dynamics_config.control_dt_s
    )
    detections: list[SensorDetection] = []
    for obstacle_state in sorted(obstacle_states, key=lambda item: item.obstacle_id):
        is_visible, relative_position_body_m = is_obstacle_geometrically_visible(
            auv_state,
            obstacle_state,
            sensor_config,
        )
        if not is_visible:
            continue
        dropout_source = sensor_rng if dropout_rng is None else dropout_rng
        if dropout_source.random() < sensor_config.dropout_probability:
            continue
        measurement_noise_body_m = sensor_rng.normal(
            loc=0.0,
            scale=measurement_std_body_m,
            size=3,
        )
        detections.append(
            SensorDetection(
                obstacle_id=obstacle_state.obstacle_id,
                relative_position_body_m=(
                    relative_position_body_m + measurement_noise_body_m
                ).astype(np.float64),
                measurement_covariance_body_m2=measurement_covariance_body_m2.copy(),
                measurement_timestamp_s=float(measurement_timestamp_s),
                arrival_timestamp_s=float(arrival_timestamp_s),
                measurement_control_tick=measurement_control_tick,
                arrival_control_tick=arrival_control_tick,
            )
        )
    return tuple(detections)
