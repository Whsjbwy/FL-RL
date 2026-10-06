"""
按整数控制 tick 调度检测到达，保留独立的 measurement_timestamp_s 给 KF。

主链比较 current_control_tick >= arrival_control_tick，不累加浮点小步决定延迟。
旧秒制 API 仅用于兼容已有外置测试/调用：控制节点附近只纠正舍入级差异，
非节点时刻向下取整，不把一般未来测量提前释放。新运行链不走该适配路径。
"""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Integral

from auv_risk_rl.exceptions import TimestampOrderError
from auv_risk_rl.types import SensorDetection

# 只服务旧秒制接口；主整数 tick 路径不使用浮点时间容差。
_LEGACY_ROUNDOFF_ULPS = 32
_LEGACY_CONTROL_DT_S = 0.2


def _control_tick(value: int, name: str) -> int:
    """核验非负整数tick，排除被Python当作整数的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TimestampOrderError(f"{name} 必须为非负整数，actual={value}。")
    return int(value)


class DetectionDelayQueue:
    """冻结固定控制步延迟队列；只负责到达，不改写测量发生时刻或执行KF。"""

    def __init__(
        self,
        control_dt_s: float = _LEGACY_CONTROL_DT_S,
        clock_origin_s: float = 0.0,
    ) -> None:
        """创建空队列；主链显式提供配置周期和世界时钟原点。"""

        if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
            raise TimestampOrderError("control_dt_s 必须为有限正数。")
        if not math.isfinite(clock_origin_s):
            raise TimestampOrderError("clock_origin_s 必须有限。")
        self._control_dt_s = control_dt_s
        self._clock_origin_s = clock_origin_s
        self._pending: list[SensorDetection] = []
        self._last_release_tick = -1

    def _legacy_timestamp_tick(self, timestamp_s: float, require_node: bool) -> int:
        """旧秒制转换：只修正节点附近ULP误差，真正节点之前的非节点查询向下取整。"""

        if not math.isfinite(timestamp_s):
            raise TimestampOrderError("旧秒制时间戳必须有限。")
        tick_value = (timestamp_s - self._clock_origin_s) / self._control_dt_s
        if not math.isfinite(tick_value):
            raise TimestampOrderError("秒制时间转换溢出。")
        nearest = round(tick_value)
        roundoff = _LEGACY_ROUNDOFF_ULPS * math.ulp(max(1.0, abs(tick_value)))
        if abs(tick_value - nearest) <= roundoff:
            return _control_tick(nearest, "converted_control_tick")
        if require_node:
            raise TimestampOrderError("旧检测时间不在控制节点；请显式提供测量和到达tick。")
        return _control_tick(math.floor(tick_value), "current_control_tick")

    def _scheduled_detection(self, detection: SensorDetection) -> SensorDetection:
        """校验或迁移单条检测的tick；所有原始测量字段保持不变。"""

        if not math.isfinite(detection.measurement_timestamp_s) or not math.isfinite(
            detection.arrival_timestamp_s
        ):
            raise TimestampOrderError("检测时间戳必须有限。")
        if detection.arrival_timestamp_s < detection.measurement_timestamp_s:
            raise TimestampOrderError("检测名义到达时间不能早于测量时间。")
        measurement_tick = detection.measurement_control_tick
        arrival_tick = detection.arrival_control_tick
        if measurement_tick is None and arrival_tick is None:
            measurement_tick = self._legacy_timestamp_tick(detection.measurement_timestamp_s, True)
            arrival_tick = self._legacy_timestamp_tick(detection.arrival_timestamp_s, True)
        elif measurement_tick is None or arrival_tick is None:
            raise TimestampOrderError("measurement/arrival_control_tick 必须同时给出。")
        measurement_tick = _control_tick(measurement_tick, "measurement_control_tick")
        arrival_tick = _control_tick(arrival_tick, "arrival_control_tick")
        if arrival_tick < measurement_tick:
            raise TimestampOrderError("到达tick不得早于测量tick。")
        return replace(
            detection,
            measurement_control_tick=measurement_tick,
            arrival_control_tick=arrival_tick,
        )

    def enqueue_many(self, detections: tuple[SensorDetection, ...]) -> None:
        """原子加入并按到达tick、测量时间、目标ID排序，非法批次不部分写入。"""

        scheduled = [self._scheduled_detection(detection) for detection in detections]
        self._pending.extend(scheduled)
        self._pending.sort(
            key=lambda item: (
                item.arrival_control_tick,
                item.measurement_timestamp_s,
                item.obstacle_id,
            )
        )

    def pop_arrived(
        self,
        current_timestamp_s: float | None = None,
        *,
        current_control_tick: int | None = None,
    ) -> tuple[SensorDetection, ...]:
        """
        主接口按整数tick释放；旧位置参数仅兼容秒制查询，两者禁止同时指定。

        主释放比较完全是整数；没有epsilon提前释放或额外整步延迟。
        旧接口在极窄ULP窗口将同一物理控制节点的浮点表示视作同一tick，
        不能用于区分该窗口内真实不同的亚步时刻；需要这种区分时必须传入tick。
        """

        if current_control_tick is None:
            if current_timestamp_s is None:
                raise TimestampOrderError("必须提供 current_control_tick。")
            tick = self._legacy_timestamp_tick(current_timestamp_s, False)
        else:
            if current_timestamp_s is not None:
                raise TimestampOrderError("不得同时提供秒时间和控制tick。")
            tick = _control_tick(current_control_tick, "current_control_tick")
        if tick < self._last_release_tick:
            raise TimestampOrderError("到达查询控制tick不得倒退。")
        self._last_release_tick = tick
        split_index = 0
        while split_index < len(self._pending):
            arrival_tick = self._pending[split_index].arrival_control_tick
            if arrival_tick is None:
                raise TimestampOrderError("内部错误：队列内检测缺少到达tick。")
            if arrival_tick > tick:
                break
            split_index += 1
        arrived = tuple(self._pending[:split_index])
        self._pending = self._pending[split_index:]
        return arrived

    @property
    def pending_count(self) -> int:
        """返回尚未释放检测数量。"""

        return len(self._pending)
