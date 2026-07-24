from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False
_default_handler: logging.Handler | None = None


def _ensure_default() -> None:
    """Install a minimal stderr handler so INFO messages are visible."""
    global _default_handler
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FMT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _default_handler = handler


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    *,
    quiet_third_party: bool = True,
) -> None:
    """Configure root logger with console and optional file output.

    Call once at application startup (e.g. from CLI entry point).
    """
    global _configured, _default_handler
    root = logging.getLogger()

    if _default_handler is not None:
        root.removeHandler(_default_handler)
        _default_handler = None
    root.handlers.clear()

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FMT))
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FMT))
        root.addHandler(file_handler)

    if quiet_third_party:
        for name in ("matplotlib", "PIL", "nibabel", "ants", "numexpr", "scipy"):
            logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, installing a minimal default config if needed."""
    if not logging.getLogger().handlers:
        _ensure_default()
    return logging.getLogger(name)
