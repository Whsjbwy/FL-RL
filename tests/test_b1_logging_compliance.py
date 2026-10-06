"""CM07：真实日志调用、保留字段保护以及标准 formatter 回归。"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import asdict

from auv_risk_rl.logging_utils import LogContext, ResearchLoggerAdapter


def test_logger_accepts_research_context(caplog) -> None:
    """PA07：component 为应用字段，标准 LogRecord.module 保持原语义。"""

    logger = logging.getLogger("b1.context")
    context = asdict(LogContext("S0", "b1", 11, "unit", "risk"))
    adapter = ResearchLoggerAdapter(logger, context)
    supplied = {"candidate_id": 7, "reason": "unit"}
    with caplog.at_level(logging.INFO, logger=logger.name):
        adapter.info("真实日志调用", extra=supplied)
    record = caplog.records[-1]
    assert record.component == "risk"
    assert record.module == "test_b1_logging_compliance"
    assert record.candidate_id == 7
    assert record.stage_id == "S0"
    assert supplied == {"candidate_id": 7, "reason": "unit"}


def test_legacy_module_dictionary_is_migrated_not_forwarded(caplog) -> None:
    """兼容外置Phase A旧字典，但不保留保留键覆盖错误。"""

    logger = logging.getLogger("b1.legacy")
    old_context = {"module": "risk"}
    with caplog.at_level(logging.INFO, logger=logger.name):
        ResearchLoggerAdapter(logger, old_context).info("旧字典迁移", extra={"module": "validator"})
    record = caplog.records[-1]
    assert record.component == "validator"
    assert record.module == "test_b1_logging_compliance"
    assert old_context == {"module": "risk"}


def test_configured_formatter_accepts_plain_and_context_logs() -> None:
    """在独立进程实际使用formatter，不污染pytest自己的root handler。"""

    source = (
        "import logging; from auv_risk_rl.logging_utils import configure_logging; "
        "configure_logging(); logging.getLogger('b1').info('plain'); "
        "logging.getLogger('b1').info('ctx', extra={'component':'sensor'})"
    )
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, check=False)
    assert completed.returncode == 0
    output = completed.stderr.decode("utf-8")
    assert "Logging error" not in output
    assert "component=unset" in output
    assert "component=sensor" in output
