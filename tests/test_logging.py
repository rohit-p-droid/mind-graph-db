"""Tests for logging infrastructure."""

import io
from mind_graph_db.logging.logger import configure_logging, get_logger


def test_get_logger_prefix() -> None:
    logger = get_logger("test_module")
    assert logger.name == "mind_graph_db.test_module"

    logger2 = get_logger("mind_graph_db.custom")
    assert logger2.name == "mind_graph_db.custom"


def test_configure_logging_custom_stream() -> None:
    stream = io.StringIO()
    configure_logging(level="DEBUG", stream=stream)

    logger = get_logger("logging_test")
    logger.debug("Test debug message")

    log_output = stream.getvalue()
    assert "DEBUG" in log_output
    assert "Test debug message" in log_output
