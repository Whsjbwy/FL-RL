"""
AUV 风险约束强化学习项目的 Stage 0 数学与接口内核。

本包当前只覆盖技术协议 v2.0 的 Stage 0 范围：坐标、降阶运动学、CV-KF、
预测分布、高斯模型条件风险上界、延迟测量重放与有限候选安全验证。
强化学习训练代码不属于 Stage 0 交付范围。
"""

from .types import (
    AUVState,
    ControlCommand,
    GroundTruthObstacleState,
    KFTrackState,
    PredictionResult,
    RiskResult,
    SensorDetection,
    TrackedObstacle,
    ValidationDecision,
)

__all__ = [
    "AUVState",
    "ControlCommand",
    "GroundTruthObstacleState",
    "KFTrackState",
    "PredictionResult",
    "RiskResult",
    "SensorDetection",
    "TrackedObstacle",
    "ValidationDecision",
]
