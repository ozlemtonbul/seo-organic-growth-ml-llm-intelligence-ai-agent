from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.localization import (
    active_language,
    localize_column_name,
    localize_dataframe,
)


def _no_data() -> None:
    language = active_language()
    st.info(
        "Veri bulunamadı."
        if language == "tr"
        else "No data available."
    )


def render_line_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str | list[str],
    title: str,
    height: int = 420,
) -> None:
    if dataframe.empty or x not in dataframe.columns:
        _no_data()
        return

    y_columns = [y] if isinstance(y, str) else y
    valid_y = [
        column
        for column in y_columns
        if column in dataframe.columns
    ]

    if not valid_y:
        _no_data()
        return

    language = active_language()
    display = localize_dataframe(
        dataframe,
        language,
    )

    labels = {
        x: localize_column_name(x, language),
        "value": (
            "Değer"
            if language == "tr"
            else "Value"
        ),
        "variable": (
            "Metrik"
            if language == "tr"
            else "Metric"
        ),
    }

    for column in valid_y:
        labels[column] = localize_column_name(
            column,
            language,
        )

    figure = px.line(
        display,
        x=x,
        y=(
            valid_y[0]
            if len(valid_y) == 1
            else valid_y
        ),
        title=title,
        markers=False,
        labels=labels,
    )

    for trace in figure.data:
        raw_name = str(
            getattr(trace, "name", "")
        )
        if raw_name in valid_y:
            trace.name = localize_column_name(
                raw_name,
                language,
            )

    figure.update_layout(
        height=height,
        legend_title_text="",
        hovermode="x unified",
        showlegend=(len(valid_y) > 1),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_bar_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    orientation: str = "v",
    height: int = 420,
) -> None:
    if (
        dataframe.empty
        or x not in dataframe.columns
        or y not in dataframe.columns
    ):
        _no_data()
        return

    language = active_language()
    display = localize_dataframe(
        dataframe,
        language,
    )

    figure = px.bar(
        display,
        x=x,
        y=y,
        title=title,
        orientation=orientation,
        labels={
            x: localize_column_name(x, language),
            y: localize_column_name(y, language),
        },
    )

    figure.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_scatter_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    size: str | None = None,
    color: str | None = None,
    hover_name: str | None = None,
    height: int = 450,
) -> None:
    if (
        dataframe.empty
        or x not in dataframe.columns
        or y not in dataframe.columns
    ):
        _no_data()
        return

    language = active_language()
    display = localize_dataframe(
        dataframe,
        language,
    )

    actual_size = (
        size
        if size in dataframe.columns
        else None
    )
    actual_color = (
        color
        if color in dataframe.columns
        else None
    )
    actual_hover = (
        hover_name
        if hover_name in dataframe.columns
        else None
    )

    labels = {
        x: localize_column_name(x, language),
        y: localize_column_name(y, language),
    }

    for column in (
        actual_size,
        actual_color,
        actual_hover,
    ):
        if column:
            labels[column] = localize_column_name(
                column,
                language,
            )

    figure = px.scatter(
        display,
        x=x,
        y=y,
        size=actual_size,
        color=actual_color,
        hover_name=actual_hover,
        title=title,
        labels=labels,
    )

    figure.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_donut_chart(
    dataframe: pd.DataFrame,
    names: str,
    values: str,
    title: str,
    height: int = 420,
) -> None:
    if (
        dataframe.empty
        or names not in dataframe.columns
        or values not in dataframe.columns
    ):
        _no_data()
        return

    language = active_language()
    display = localize_dataframe(
        dataframe,
        language,
    )

    figure = px.pie(
        display,
        names=names,
        values=values,
        title=title,
        hole=0.55,
        labels={
            names: localize_column_name(
                names,
                language,
            ),
            values: localize_column_name(
                values,
                language,
            ),
        },
    )

    figure.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_forecast_vs_actual(
    dataframe: pd.DataFrame,
    date_column: str,
    actual_column: str,
    predicted_column: str,
    title: str,
    height: int = 420,
) -> None:
    required = {
        date_column,
        actual_column,
        predicted_column,
    }

    if (
        dataframe.empty
        or not required.issubset(
            dataframe.columns
        )
    ):
        _no_data()
        return

    language = active_language()
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=dataframe[date_column],
            y=dataframe[actual_column],
            mode="lines",
            name=(
                "Gerçek"
                if language == "tr"
                else "Actual"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=dataframe[date_column],
            y=dataframe[predicted_column],
            mode="lines",
            name=(
                "Tahmin"
                if language == "tr"
                else "Predicted"
            ),
        )
    )

    figure.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        xaxis_title=localize_column_name(
            date_column,
            language,
        ),
        yaxis_title=(
            "Değer"
            if language == "tr"
            else "Value"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )
