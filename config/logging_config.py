from __future__ import annotations

import logging
from typing import Optional

from config.settings import SETTINGS


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """
    Configure application-wide logging.

    Parameters
    ----------
    level:
        Optional logging level. When omitted, the value configured in
        application settings is used.
    log_format:
        Format applied to log messages.
    """
    resolved_level = (level or SETTINGS.log_level).upper()

    numeric_level = getattr(logging, resolved_level, None)

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid logging level: {resolved_level!r}"
        )

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        force=True,
    )


def get_logger(name: str = "seo_growth_intelligence") -> logging.Logger:
    """
    Return a named application logger.
    """
    return logging.getLogger(name)


configure_logging()

logger = get_logger()