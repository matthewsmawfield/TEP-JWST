"""
Centralised logging utility for the TEP-JWST pipeline.

Provides:
  - TEPLogger: per-step logger with dual console + file output, DEBUG support
  - print_status: formatted status messages with severity-level prefixes
  - log_section / log_subsection: visual section headers for structured output
  - log_data: tabular data dump helper for verbose diagnostic output
  - log_timing: context manager for measuring and reporting elapsed time
  - log_dict: structured key-value dump for dictionaries and result objects

Usage pattern (standard in all pipeline steps):

    from scripts.utils.logger import TEPLogger, set_step_logger, print_status

    logger = TEPLogger(f"step_{STEP_NUM}", log_file_path=LOGS_PATH / f"step_{STEP_NUM}_{STEP_NAME}.log")
    set_step_logger(logger)

    print_status("Step 001: Load UNCOVER DR4 Data", "TITLE")
    print_status("Loading catalog...", "PROCESS")
    print_status(f"Loaded N = {len(df)} galaxies", "INFO")
    print_status("Step 001 complete.", "SUCCESS")

Verbosity control:
  - Default level is INFO (standard pipeline runs).
  - Set environment variable TEP_VERBOSE=1 to enable DEBUG-level output
    (detailed per-galaxy diagnostics, intermediate computations, etc.).
  - DEBUG messages are always written to log files regardless of console level,
    ensuring full traceability for post-hoc debugging.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Verbosity control: TEP_VERBOSE=1 enables DEBUG-level console output
# ---------------------------------------------------------------------------
_VERBOSE = os.environ.get("TEP_VERBOSE", "0") == "1"

# Module-level logger used by print_status when set via set_step_logger
_active_logger: TEPLogger | None = None


class TEPLogger:
    """Per-step logger with dual console + file output.

    Parameters
    ----------
    name : str
        Logger name (typically f"step_{STEP_NUM}").
    log_file_path : Path or None
        If provided, log messages are written to this file in addition to
        the console. The file always receives DEBUG-level messages for
        full traceability, regardless of console verbosity.
    """

    def __init__(self, name, log_file_path=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Always capture everything at file level

        # Remove existing handlers to avoid duplicates on re-instantiation
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler — full timestamp format, DEBUG level (captures everything)
        if log_file_path:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-7s %(message)s', datefmt='%H:%M:%S'))
            self.logger.addHandler(fh)

        # Console handler — plain format (no timestamp on console), level depends on verbosity
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if _VERBOSE else logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def exception(self, msg):
        self.logger.exception(msg)


def set_step_logger(logger: TEPLogger):
    """Register a TEPLogger so that print_status routes through it."""
    global _active_logger
    _active_logger = logger


# ---------------------------------------------------------------------------
# Severity-level prefixes for console output
# ---------------------------------------------------------------------------
_LEVEL_PREFIXES = {
    "TITLE":   "═══",
    "PROCESS": ">>>",
    "SUCCESS": "✓  ",
    "ERROR":   "✗  ",
    "WARNING": "⚠  ",
    "INFO":    "   ",
    "DEBUG":   " · ",
    None:      "   ",
}


def print_status(msg: str, level: str | None = None):
    """Print a formatted status message with severity-level prefix.

    If a step logger has been registered via set_step_logger(), the message
    is routed through it (both console and log file). Otherwise falls back
    to a plain print.

    Parameters
    ----------
    msg : str
        The message to display.
    level : str or None
        One of "TITLE", "PROCESS", "SUCCESS", "ERROR", "WARNING", "INFO",
        "DEBUG", or None. Controls the prefix and routing severity.
    """
    prefix = _LEVEL_PREFIXES.get(level, "   ")
    formatted = f"{prefix} {msg}" if msg else ""

    if _active_logger is not None:
        if level == "ERROR":
            _active_logger.error(formatted)
        elif level == "WARNING":
            _active_logger.warning(formatted)
        elif level == "DEBUG":
            _active_logger.debug(formatted)
        else:
            _active_logger.info(formatted)
    else:
        print(formatted)


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log_section(title: str, width: int = 70):
    """Log a major section header with visual separator.

    Example output:
        ┌──────────────────────────────────────────────────────────────────────┐
        │ SECTION TITLE                                                        │
        └──────────────────────────────────────────────────────────────────────┘
    """
    bar = "─" * width
    print_status(f"┌{bar}┐", "INFO")
    print_status(f"│ {title:<{width - 2}}│", "INFO")
    print_status(f"└{bar}┘", "INFO")


def log_subsection(title: str, width: int = 60):
    """Log a subsection header with a lighter visual separator.

    Example output:
        ── Subsection Title ──────────────────────────────────────────
    """
    dash_count = max(4, width - len(title) - 4)
    print_status(f"── {title} {'─' * dash_count}", "INFO")


def log_data(label: str, value, fmt: str = "auto", indent: int = 2):
    """Log a labelled data value with consistent formatting.

    Parameters
    ----------
    label : str
        Human-readable description of the value.
    value : any
        The value to display. Numbers are formatted according to `fmt`.
    fmt : str
        Format specifier: "auto" (default), ".2f", ".4e", ".1%", "d", "s", etc.
    indent : int
        Number of leading spaces for alignment.
    """
    prefix = " " * indent
    if fmt == "auto":
        if isinstance(value, float):
            if abs(value) < 0.001 and value != 0:
                s = f"{value:.4e}"
            elif abs(value) >= 1e6:
                s = f"{value:.4e}"
            else:
                s = f"{value:.4f}"
        elif isinstance(value, int):
            s = f"{value:,}"
        else:
            s = str(value)
    else:
        try:
            s = format(value, fmt)
        except (ValueError, TypeError):
            s = str(value)
    print_status(f"{prefix}{label}: {s}", "INFO")


def log_dict(d: dict, indent: int = 2, max_items: int = 20, level: str = "INFO"):
    """Log a dictionary as a structured key-value listing.

    Parameters
    ----------
    d : dict
        Dictionary to display.
    indent : int
        Leading spaces for alignment.
    max_items : int
        Maximum number of items to display; remaining are summarised as a count.
    level : str
        Severity level for routing (default "INFO").
    """
    prefix = " " * indent
    items = list(d.items())
    for k, v in items[:max_items]:
        if isinstance(v, float):
            if abs(v) < 0.001 and v != 0:
                s = f"{v:.4e}"
            else:
                s = f"{v:.4f}"
        elif isinstance(v, int):
            s = f"{v:,}"
        else:
            s = str(v)
        print_status(f"{prefix}{k}: {s}", level)
    if len(items) > max_items:
        remaining = len(items) - max_items
        print_status(f"{prefix}... and {remaining} more fields", level)


@contextmanager
def log_timing(label: str, level: str = "INFO"):
    """Context manager that logs elapsed time for a code block.

    Usage:
        with log_timing("SFRD computation"):
            compute_sfrd(...)

    Output:
        >>> SFRD computation...
        ✓  SFRD computation completed in 1.23s
    """
    print_status(f"{label}...", "PROCESS")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if elapsed < 1:
            time_str = f"{elapsed * 1000:.0f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            time_str = f"{elapsed / 60:.1f}min"
        print_status(f"{label} completed in {time_str}", "SUCCESS")
