"""
项目统一异常类型。

说明：
Stage 0 禁止使用裸 except 或在数值异常后静默继续。所有核心异常均通过明确类型
向上层传播，以便日志中记录 stage_id、run_id、seed、scenario_id 与 component。
"""


class AUVResearchError(RuntimeError):
    """科研代码基础异常。"""


class InvalidCovarianceError(AUVResearchError):
    """协方差矩阵存在明显非对称、非有限值或非半正定问题。"""


class CoordinateFrameError(AUVResearchError):
    """坐标系输入不一致或旋转矩阵不满足约束。"""


class NumericalRiskError(AUVResearchError):
    """风险计算遇到无法解释的数值异常。"""


class InvalidActionError(AUVResearchError):
    """控制动作超出协议定义域或包含非有限值。"""


class TimestampOrderError(AUVResearchError):
    """测量时间戳顺序或延迟重放历史不满足要求。"""


class InvalidEnvironmentStateError(AUVResearchError):
    """环境真值状态、边界或事件几何输入不满足协议约束。"""


class SensorSimulationError(AUVResearchError):
    """简化声呐仿真输入或延迟队列状态非法。"""
