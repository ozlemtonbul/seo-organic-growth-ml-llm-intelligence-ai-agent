from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.utils import (
    format_currency,
    format_integer,
    format_number,
    format_percent,
    format_position,
)


def render_metric_card(
    label: str,
    value: Any,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    """
    Render a generic Streamlit metric card.
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text,
    )


def render_integer_metric(
    label: str,
    value: Any,
    delta: str | None = None,
) -> None:
    render_metric_card(
        label=label,
        value=format_integer(value),
        delta=delta,
    )


def render_number_metric(
    label: str,
    value: Any,
    decimals: int = 2,
    delta: str | None = None,
) -> None:
    render_metric_card(
        label=label,
        value=format_number(
            value,
            decimals=decimals,
        ),
        delta=delta,
    )


def render_percent_metric(
    label: str,
    value: Any,
    decimals: int = 1,
    delta: str | None = None,
    value_is_ratio: bool = True,
) -> None:
    render_metric_card(
        label=label,
        value=format_percent(
            value,
            decimals=decimals,
            value_is_ratio=value_is_ratio,
        ),
        delta=delta,
    )


def render_position_metric(
    label: str,
    value: Any,
    decimals: int = 2,
    delta: str | None = None,
) -> None:
    render_metric_card(
        label=label,
        value=format_position(
            value,
            decimals=decimals,
        ),
        delta=delta,
    )


def render_currency_metric(
    label: str,
    value: Any,
    symbol: str = "₺",
    decimals: int = 2,
    delta: str | None = None,
) -> None:
    render_metric_card(
        label=label,
        value=format_currency(
            value,
            symbol=symbol,
            decimals=decimals,
        ),
        delta=delta,
    )


def render_kpi_row(
    metrics: list[dict[str, Any]],
) -> None:
    """
    Render a row of KPI cards.

    Each metric dict can contain:
    label, value, delta, help.
    """
    if not metrics:
        return

    columns = st.columns(
        len(metrics)
    )

    for column, metric in zip(
        columns,
        metrics,
    ):
        with column:
            st.metric(
                label=str(
                    metric.get(
                        "label",
                        "",
                    )
                ),
                value=str(
                    metric.get(
                        "value",
                        "-",
                    )
                ),
                delta=metric.get(
                    "delta"
                ),
                help=metric.get(
                    "help"
                ),
            )
