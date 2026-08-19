from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "streamlit.testing.v1",
    reason="Streamlit AppTest is not available.",
)
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOME_PAGE = PROJECT_ROOT / "dashboard" / "pages" / "0_Home.py"


def _exceptions(at: AppTest) -> list[str]:
    result = []
    try:
        for item in at.exception:
            value = getattr(item, "value", None)
            result.append(str(value if value is not None else item))
    except Exception:
        pass
    return result


def test_language_selector_records_new_navigation_intent():
    at = AppTest.from_file(str(HOME_PAGE))
    at.session_state["dashboard_language"] = "tr"
    at.run(timeout=45)

    assert not _exceptions(at)

    language_widget = at.selectbox(key="dashboard_language")
    language_widget.select("en")
    at.run(timeout=45)

    assert not _exceptions(at)
    assert at.session_state["dashboard_language"] == "en"
    assert (
        at.session_state["_dashboard_language_requested"]
        == "en"
    )
