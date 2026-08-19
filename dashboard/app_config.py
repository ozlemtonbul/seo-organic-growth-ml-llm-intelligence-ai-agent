from __future__ import annotations

from pathlib import Path

from config.settings import (
    OUTPUT_DIR as SETTINGS_OUTPUT_DIR,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

_CONFIGURED_OUTPUT_DIR = Path(
    SETTINGS_OUTPUT_DIR
).expanduser()

OUTPUT_DIR = (
    _CONFIGURED_OUTPUT_DIR
    if _CONFIGURED_OUTPUT_DIR.is_absolute()
    else PROJECT_ROOT
    / _CONFIGURED_OUTPUT_DIR
).resolve()


# ============================================================
# APPLICATION
# ============================================================

APP_TITLE = (
    "SEO Organic Growth Intelligence AI Agent"
)

APP_SUBTITLE = (
    "AI-Powered SEO, GEO and Organic Growth "
    "Decision Intelligence Platform"
)


# ============================================================
# SEO TARGET DEFAULTS
# ============================================================

DEFAULT_TARGET_POSITION = 10.0

DEFAULT_TARGET_CTR = 0.05