from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import src.utils.date_utils as date_utils


def test_parse_iso_date() -> None:
    parsed_date = date_utils.parse_iso_date(
        "2026-07-16",
        "TEST_DATE",
    )

    assert parsed_date.year == 2026
    assert parsed_date.month == 7
    assert parsed_date.day == 16


def test_parse_iso_date_rejects_invalid_format() -> None:
    with pytest.raises(
        ValueError,
        match="YYYY-MM-DD",
    ):
        date_utils.parse_iso_date(
            "16-07-2026",
            "TEST_DATE",
        )


def test_resolve_custom_date_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings = SimpleNamespace(
        date_mode="custom",
        date_from="2026-06-01",
        date_to="2026-06-30",
        api_data_delay_days=2,
    )

    monkeypatch.setattr(
        date_utils,
        "SETTINGS",
        test_settings,
    )

    assert date_utils.resolve_date_range() == (
        "2026-06-01",
        "2026-06-30",
    )


def test_custom_date_range_rejects_reverse_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings = SimpleNamespace(
        date_mode="custom",
        date_from="2026-07-31",
        date_to="2026-07-01",
        api_data_delay_days=2,
    )

    monkeypatch.setattr(
        date_utils,
        "SETTINGS",
        test_settings,
    )

    with pytest.raises(
        ValueError,
        match="DATE_FROM cannot be later",
    ):
        date_utils.resolve_date_range()


def test_last_60_days_contains_sixty_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings = SimpleNamespace(
        date_mode="last_60_days",
        date_from="",
        date_to="",
        api_data_delay_days=2,
    )

    monkeypatch.setattr(
        date_utils,
        "SETTINGS",
        test_settings,
    )

    start_text, end_text = (
        date_utils.resolve_date_range()
    )

    start_date = datetime.strptime(
        start_text,
        "%Y-%m-%d",
    ).date()

    end_date = datetime.strptime(
        end_text,
        "%Y-%m-%d",
    ).date()

    assert (
        end_date - start_date
    ).days == 59


def test_unsupported_date_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings = SimpleNamespace(
        date_mode="unsupported",
        date_from="",
        date_to="",
        api_data_delay_days=2,
    )

    monkeypatch.setattr(
        date_utils,
        "SETTINGS",
        test_settings,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported DATE_MODE",
    ):
        date_utils.resolve_date_range()