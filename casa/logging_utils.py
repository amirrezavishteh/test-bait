"""Structured, sanitising logging for CASA.

Two requirements from the spec drive this module:

* Logs must be parseable by a log-aggregation system → optional JSON-lines
  formatter (:class:`_JsonFormatter`).
* No log line may contain raw model output → :func:`sanitize` strips control
  characters and truncates, and is applied by callers before logging any text
  that originated from a model.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_LOGGER_NAME = "casa"


def sanitize(text: Any, max_chars: int = 200) -> str:
    """Make arbitrary (possibly model-produced) text safe to log.

    Control characters are replaced with ``.``, newlines/tabs collapsed to a
    single space, and the result truncated with an explicit ``…(+N)`` marker.

    Args:
        text: Any object; coerced via :func:`str`.
        max_chars: Maximum characters retained before truncation (>= 1).

    Returns:
        A single-line, control-char-free, length-bounded string.
    """
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _CONTROL_RE.sub(".", s)
    if max_chars >= 1 and len(s) > max_chars:
        hidden = len(s) - max_chars
        s = s[:max_chars] + f"…(+{hidden})"
    return s


class _JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects for aggregation."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Promote structured 'extra' fields (anything not standard).
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_STANDARD_ATTRS = set(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime"}


def configure_logging(level: str = "INFO", json_output: bool = False) -> logging.Logger:
    """Configure and return the shared ``casa`` logger.

    Idempotent: repeated calls reset handlers rather than stacking them.

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_output: If true use the JSON-lines formatter.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(stream=sys.stderr)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | casa:%(funcName)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the shared ``casa`` logger (configuring defaults if needed)."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        configure_logging()
    return logger
