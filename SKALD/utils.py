import logging
import os
import json
import psutil
import time
from datetime import datetime
from tqdm import tqdm

import logging
logger = logging.getLogger("SKALD")


def log_performance(step_name: str, start_time: float):
    """Logs elapsed time and memory usage for a step."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    elapsed = time.time() - start_time
    logger.info(f"[{step_name}] Time taken: {elapsed:.2f} sec | Memory: {mem_mb:.2f} MB")


def format_time(seconds):
    """
    Convert elapsed time in seconds into hours, minutes, and seconds.

    Args:
        seconds (float): Elapsed time.

    Returns:
        tuple: (hours, minutes, seconds)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return hours, minutes, secs

def find_max_decimal_places(series):
    decimals = series.dropna().map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
    return decimals.max()
