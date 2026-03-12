import logging
import sys
from pathlib import Path

# Module-level logger used by print_status when set via set_step_logger
_active_logger = None


class TEPLogger:
    def __init__(self, name, log_file_path=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

        # Console handler — plain, no timestamp (timestamps go to file only)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

        # File handler — full timestamp format
        if log_file_path:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), mode='w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

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


_LEVEL_PREFIXES = {
    "TITLE":   "═══",
    "PROCESS": ">>>",
    "SUCCESS": "✓  ",
    "ERROR":   "✗  ",
    "WARNING": "⚠  ",
    "INFO":    "   ",
    None:      "   ",
}


def print_status(msg: str, level: str | None = None):
    """
    Print a formatted status message.

    If a step logger has been registered via set_step_logger(), the message
    is routed through it (→ both console and log file). Otherwise falls back
    to a plain print.
    """
    prefix = _LEVEL_PREFIXES.get(level, "   ")
    formatted = f"{prefix} {msg}" if msg else ""

    if _active_logger is not None:
        if level == "ERROR":
            _active_logger.error(formatted)
        elif level == "WARNING":
            _active_logger.warning(formatted)
        else:
            _active_logger.info(formatted)
    else:
        print(formatted)
