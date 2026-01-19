import pandas as pd
from SKALD.utils import format_time, find_max_decimal_places, log_performance
import time
import psutil
import os
from unittest.mock import MagicMock, patch


def test_format_time_basic():
    assert format_time(0) == (0, 0, 0)
    assert format_time(59) == (0, 0, 59)
    assert format_time(60) == (0, 1, 0)
    assert format_time(3661) == (1, 1, 1)


def test_find_max_decimal_places():
    s = pd.Series(["1.234", "2.1", "3"])
    assert find_max_decimal_places(s) == 3

    s2 = pd.Series(["10", "2", "300"])
    assert find_max_decimal_places(s2) == 0


def test_log_performance(tmp_path):
    """
    Verify that log_performance calls logger.info with time + memory.
    """

    logger = MagicMock()

    # Fake memory info
    class FakeProcess:
        def memory_info(self):
            class X:
                rss = 123 * 1024 * 1024  # 123MB
            return X()

    with patch("psutil.Process", return_value=FakeProcess()):
        start_time = time.time() - 1.5  # pretend 1.5 sec elapsed
        log_performance(logger, "TestStep", start_time)

    # Validate logger was called correctly
    assert logger.info.call_count == 1
    logged_msg = logger.info.call_args[0][0]

    assert "TestStep" in logged_msg
    assert "Time taken" in logged_msg
    assert "Memory" in logged_msg
