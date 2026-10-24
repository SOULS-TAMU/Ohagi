"""Narrow compatibility for older Orbax throughput logging on Windows."""

from functools import wraps
import importlib
import math
import os
import sys


def _patch_handler(handler):
    original = handler._log_io_metrics
    if getattr(original, "_ohagi_nonzero_clock_interval", False):
        return

    @wraps(original)
    def log_io_metrics(size, start_time, *args, **kwargs):
        now = handler.time.time()
        if now == start_time:
            # Python <3.13 on Windows can report the same coarse wall-clock
            # tick for a small checkpoint. Orbax 0.11 converts that zero
            # duration to NaN and humanize raises while formatting it.
            # Adjust only the diagnostic interval; checkpoint data is untouched.
            start_time = math.nextafter(start_time, -math.inf)
        return original(size, start_time, *args, **kwargs)

    log_io_metrics._ohagi_nonzero_clock_interval = True
    handler._log_io_metrics = log_io_metrics


def prepare_orbax(ocp):
    """Load Orbax and prepare compatibility support for the installed JAX version."""
    if os.name != "nt" or sys.version_info >= (3, 13):
        return
    if not str(getattr(ocp, "__version__", "")).startswith("0.11."):
        return
    handler = importlib.import_module(
        "orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler"
    )
    if hasattr(handler, "_log_io_metrics"):
        _patch_handler(handler)
