"""
科研运行日志辅助模块。

所有正式日志统一携带 stage_id、run_id、seed、scenario_id 与 component 字段，
避免在多 seed、多场景实验中出现无法追溯的孤立日志。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class LogContext:
    """日志上下文字段；不涉及物理单位、shape 或坐标系。"""

    stage_id: str
    run_id: str
    seed: int
    scenario_id: str
    component: str


class ResearchLoggerAdapter(logging.LoggerAdapter):
    """
    为标准 logging 自动附加科研追溯字段。

    对应技术协议：
        第 15、30 章日志与运行记录要求；无独立编号公式。

    关键假设：
        上层在每个运行/场景开始时创建正确的 LogContext。

    重要限制：
        本类不负责日志文件轮转和集中式采集。
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """将上下文字段注入日志 extra，避免业务代码重复拼接字符串。"""

        context = dict(self.extra or {})
        supplied = dict(kwargs.get("extra") or {})
        for fields in (context, supplied):
            # 仅迁移旧调用字典：绝不把保留键 module 传入 LogRecord。
            legacy_component = fields.pop("module", None)
            if legacy_component is not None:
                fields.setdefault("component", legacy_component)
        context.update(supplied)
        updated_kwargs = dict(kwargs)
        updated_kwargs["extra"] = context
        return msg, updated_kwargs


def configure_logging(level: int = logging.INFO) -> None:
    """
    配置科研项目默认日志格式。

    对应技术协议：
        第 30 章工程记录要求；无独立编号公式。

    参数：
        level:
            logging 级别，默认 INFO；无单位与坐标系。

    返回：
        无。

    shape/单位/坐标系：
        不适用。

    关键假设：
        由进程入口统一调用一次。

    重要限制：
        正式实验如需 JSON 日志，应在上层增加 handler，而不是在核心数学模块 print。
    """

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s stage=%(stage_id)s run=%(run_id)s seed=%(seed)s "
            "scenario=%(scenario_id)s component=%(component)s %(message)s",
            defaults={
                "stage_id": "unset", "run_id": "unset", "seed": "unset",
                "scenario_id": "unset", "component": "unset",
            },
        )
    )
    # 对未附科研上下文的标准库日志使用显式 unset，不能伪造运行元数据。
    logging.basicConfig(level=level, handlers=[handler])
