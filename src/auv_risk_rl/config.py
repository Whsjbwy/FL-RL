"""
Stage 0 配置读取与强类型数据结构。

功能：
1. 从 YAML 加载环境、动力学、传感、KF、风险、验证器与有限时域成本配置；
2. 统一单位后缀，禁止核心模块内部散落 Magic Number；
3. 为运行快照与版本追踪提供稳定序列化入口。

说明：
本模块只负责配置语义与基础一致性校验，不执行物理、滤波或风险计算。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    开放三维仿真环境与任务终止配置。

    对应技术协议：
        第 7、9、15、25.1–25.2 章；无独立编号公式。

    参数：
        position_lower_bound_ned_m / position_upper_bound_ned_m:
            AUV 物理活动空间边界，shape=(3,)，单位 m，坐标系 NED。
        goal_radius_m:
            到达判定中心距离阈值，单位 m。
        max_episode_control_steps:
            一个计划任务最多控制周期数，无量纲步数。

    关键假设：
        A1、A8；边界按 AUV 完整球包络检查，而不是仅检查中心点。

    重要限制：
        本配置不包含复杂静态迷宫、海床地形或实机作业区约束。
    """

    position_lower_bound_ned_m: tuple[float, float, float]
    position_upper_bound_ned_m: tuple[float, float, float]
    goal_radius_m: float
    max_episode_control_steps: int


@dataclass(frozen=True)
class DynamicsConfig:
    """AUV 降阶动力学与积分配置，单位均由字段名明确。"""

    control_dt_s: float
    integration_dt_s: float
    min_surge_speed_mps: float
    max_surge_speed_mps: float
    max_yaw_rate_rad_s: float
    max_pitch_rate_rad_s: float
    max_pitch_rad: float
    surge_time_constant_s: float
    yaw_rate_time_constant_s: float
    pitch_rate_time_constant_s: float
    max_surge_accel_mps2: float
    max_yaw_accel_rad_s2: float
    max_pitch_accel_rad_s2: float


@dataclass(frozen=True)
class KFConfig:
    """CV-KF 配置。"""

    acceleration_spectral_density_m2_s3: tuple[float, float, float]
    initial_velocity_std_mps: float
    covariance_symmetry_tolerance: float
    covariance_psd_tolerance: float


@dataclass(frozen=True)
class SensorConfig:
    """
    简化声呐配置。

    对应技术协议：
        Eq. (16)–(18) 与第 9 章 Delay 规则。

    说明：
        measurement_delay_control_steps 使用控制周期步数而不是秒，避免把“步”和“秒”混用。
    """

    range_m: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    measurement_std_body_m: tuple[float, float, float]
    dropout_probability: float
    measurement_delay_control_steps: int


@dataclass(frozen=True)
class RiskConfig:
    """风险上界与几何安全半径配置。"""

    auv_radius_m: float
    extra_margin_m: float
    short_horizon_risk_budget: float
    mean_norm_tolerance_m: float
    variance_tolerance_m2: float


@dataclass(frozen=True)
class ValidatorConfig:
    """在线验证器有限候选与滚动时域配置。"""

    validation_horizon_s: float
    speed_grid_mps: tuple[float, ...]
    yaw_rate_grid_rad_s: tuple[float, ...]
    pitch_rate_grid_rad_s: tuple[float, ...]


@dataclass(frozen=True)
class CostConfig:
    """
    有限任务时域归一化成本配置。

    对应技术协议：
        Eq. (45)–(46)、Eq. (50)–(51)。

    说明：
        当前 Stage 0 只实现数学账本，不实现 cost critic 或拉格朗日优化器。
    """

    discount_gamma: float
    planned_horizon_control_steps: int


@dataclass(frozen=True)
class ProjectConfig:
    """Stage 0 顶层配置。"""

    protocol_version: str
    config_version: str
    stage: str
    environment: EnvironmentConfig
    dynamics: DynamicsConfig
    kf: KFConfig
    sensor: SensorConfig
    risk: RiskConfig
    validator: ValidatorConfig
    cost: CostConfig

    def to_dict(self) -> dict[str, Any]:
        """
        将配置转换为可序列化字典。

        对应技术协议：
            第 30 章工程代码结构与附录 D.3 运行记录要求；无独立编号公式。

        输入：
            无。

        返回：
            完整配置字典；不涉及 shape、单位或坐标系。

        关键假设：
            YAML 已通过本模块的强类型构造与一致性校验。

        重要限制：
            本函数不校验实验 Train/Validation/Test split 或 Git 工作区清洁状态。
        """

        return asdict(self)


def _tuple_of_floats(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """将 YAML 数值序列稳定转换为 float tuple，便于 frozen dataclass 使用。"""

    return tuple(float(value) for value in values)


def _validate_project_config(config: ProjectConfig) -> None:
    """
    对跨模块配置执行最小一致性检查。

    对应技术协议：
        第 7、9、13、14 章参数冻结要求；无独立编号公式。

    参数：
        config:
            完整项目配置。

    返回：
        无；发现违反协议的配置组合时抛出 ValueError。

    shape/单位/坐标系：
        环境边界为 shape=(3,) 的 NED 米制向量；其他字段单位见各 dataclass。

    关键假设：
        Stage 0 主设置要求 control_dt_s 是 integration_dt_s 的整数倍。

    重要限制：
        这里只检查会破坏接口语义的关系，不替代后续场景可达性检查。
    """

    if config.dynamics.control_dt_s <= 0.0 or config.dynamics.integration_dt_s <= 0.0:
        raise ValueError("控制周期与积分步长必须为正。")
    integration_steps_float = config.dynamics.control_dt_s / config.dynamics.integration_dt_s
    if abs(integration_steps_float - round(integration_steps_float)) > 1.0e-12:
        raise ValueError(
            "control_dt_s 必须是 integration_dt_s 的整数倍："
            f"control_dt_s={config.dynamics.control_dt_s}, "
            f"integration_dt_s={config.dynamics.integration_dt_s}。"
        )
    lower_bound_ned_m = config.environment.position_lower_bound_ned_m
    upper_bound_ned_m = config.environment.position_upper_bound_ned_m
    bound_pairs = zip(lower_bound_ned_m, upper_bound_ned_m, strict=True)
    if any(lower >= upper for lower, upper in bound_pairs):
        raise ValueError(
            "NED 空间下界必须逐轴小于上界："
            f"lower={lower_bound_ned_m}, upper={upper_bound_ned_m}。"
        )
    if config.risk.auv_radius_m <= 0.0:
        raise ValueError("auv_radius_m 必须为正。")
    if config.environment.goal_radius_m <= 0.0:
        raise ValueError("goal_radius_m 必须为正。")
    if config.environment.max_episode_control_steps <= 0:
        raise ValueError("max_episode_control_steps 必须为正整数。")
    if not 0.0 <= config.sensor.dropout_probability <= 1.0:
        raise ValueError("dropout_probability 必须位于 [0,1]。")
    if config.sensor.measurement_delay_control_steps < 0:
        raise ValueError("measurement_delay_control_steps 不得为负。")
    if not 0.0 < config.cost.discount_gamma <= 1.0:
        raise ValueError("discount_gamma 必须位于 (0,1]。")
    if config.cost.planned_horizon_control_steps != config.environment.max_episode_control_steps:
        raise ValueError(
            "有限时域成本的 planned_horizon_control_steps 必须与环境计划时域一致，"
            "避免终止尾项使用不同任务长度。"
        )


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """
    加载并构造 Stage 0 项目配置。

    对应技术协议：
        第 30 章工程代码结构与附录 B 参数表；无独立编号公式。

    参数：
        config_path:
            YAML 文件路径；无物理单位和坐标系。

    返回：
        ProjectConfig：强类型配置对象。

    shape/单位/坐标系：
        不适用；具体物理字段的单位已经编码在字段名中。

    关键假设：
        配置字段遵循仓库提供的 stage0.yaml 结构。

    重要限制：
        本函数不会自动“修复”非法值，缺字段、类型错误或跨模块不一致会显式报错。
    """

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    environment_raw = dict(raw["environment"])
    environment_raw["position_lower_bound_ned_m"] = _tuple_of_floats(
        environment_raw["position_lower_bound_ned_m"]
    )
    environment_raw["position_upper_bound_ned_m"] = _tuple_of_floats(
        environment_raw["position_upper_bound_ned_m"]
    )
    environment = EnvironmentConfig(**environment_raw)

    dynamics = DynamicsConfig(**raw["dynamics"])

    kf_raw = dict(raw["kf"])
    kf_raw["acceleration_spectral_density_m2_s3"] = _tuple_of_floats(
        kf_raw["acceleration_spectral_density_m2_s3"]
    )
    kf = KFConfig(**kf_raw)

    sensor_raw = dict(raw["sensor"])
    sensor_raw["measurement_std_body_m"] = _tuple_of_floats(
        sensor_raw["measurement_std_body_m"]
    )
    sensor = SensorConfig(**sensor_raw)

    risk = RiskConfig(**raw["risk"])

    validator_raw = dict(raw["validator"])
    for field_name in (
        "speed_grid_mps",
        "yaw_rate_grid_rad_s",
        "pitch_rate_grid_rad_s",
    ):
        validator_raw[field_name] = _tuple_of_floats(validator_raw[field_name])
    validator = ValidatorConfig(**validator_raw)

    cost = CostConfig(**raw["cost"])

    config = ProjectConfig(
        protocol_version=str(raw["protocol_version"]),
        config_version=str(raw["config_version"]),
        stage=str(raw["stage"]),
        environment=environment,
        dynamics=dynamics,
        kf=kf,
        sensor=sensor,
        risk=risk,
        validator=validator,
        cost=cost,
    )
    _validate_project_config(config)
    return config
