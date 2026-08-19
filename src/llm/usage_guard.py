from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from threading import Lock

from config.settings import (
    LLM_DAILY_REQUEST_LIMIT,
    LLM_USAGE_FILE,
)


_USAGE_LOCK = Lock()


def _usage_path() -> Path:
    path = Path(LLM_USAGE_FILE)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _empty_usage() -> dict[str, str | int]:
    return {
        "date": date.today().isoformat(),
        "requests": 0,
    }


def _load_usage() -> dict[str, str | int]:
    path = _usage_path()

    if not path.exists():
        return _empty_usage()

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return _empty_usage()

    today = date.today().isoformat()

    if data.get("date") != today:
        return _empty_usage()

    return {
        "date": today,
        "requests": int(data.get("requests", 0)),
    }


def _save_usage(
    usage: dict[str, str | int],
) -> None:
    path = _usage_path()

    path.write_text(
        json.dumps(
            usage,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def get_daily_usage() -> dict[str, str | int | bool]:
    """
    Return safe local LLM usage information.
    """

    with _USAGE_LOCK:
        usage = _load_usage()

        requests = int(usage["requests"])

        remaining = max(
            LLM_DAILY_REQUEST_LIMIT - requests,
            0,
        )

        return {
            "date": str(usage["date"]),
            "requests": requests,
            "limit": LLM_DAILY_REQUEST_LIMIT,
            "remaining": remaining,
            "limit_reached": (
                requests >= LLM_DAILY_REQUEST_LIMIT
            ),
        }


def llm_request_allowed() -> bool:
    """
    Return whether another live LLM API call is allowed.
    """

    usage = get_daily_usage()

    return not bool(usage["limit_reached"])


def register_llm_request() -> bool:
    """
    Reserve one LLM API request.

    Returns False when the daily limit has already
    been reached.
    """

    with _USAGE_LOCK:
        usage = _load_usage()

        requests = int(usage["requests"])

        if requests >= LLM_DAILY_REQUEST_LIMIT:
            return False

        usage["requests"] = requests + 1

        _save_usage(usage)

        return True