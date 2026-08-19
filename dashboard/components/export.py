from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert DataFrame to UTF-8 CSV bytes.
    """
    return dataframe.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


def dataframe_to_excel_bytes(
    dataframe: pd.DataFrame,
    sheet_name: str = "Data",
) -> bytes:
    """
    Convert DataFrame to Excel bytes.

    Requires an installed pandas-compatible Excel writer.
    """
    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31],
        )

    buffer.seek(0)

    return buffer.getvalue()


def render_csv_download(
    dataframe: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    key: str | None = None,
) -> None:
    """
    Render CSV download button.
    """
    if dataframe.empty:
        return

    st.download_button(
        label=label,
        data=dataframe_to_csv_bytes(
            dataframe
        ),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_excel_download(
    dataframe: pd.DataFrame,
    filename: str,
    label: str = "Download Excel",
    sheet_name: str = "Data",
    key: str | None = None,
) -> None:
    """
    Render Excel download button.
    """
    if dataframe.empty:
        return

    try:
        excel_data = (
            dataframe_to_excel_bytes(
                dataframe=dataframe,
                sheet_name=sheet_name,
            )
        )

    except ImportError:
        st.caption(
            (
                "Excel dışa aktarımı için openpyxl gereklidir."
                if st.session_state.get("dashboard_language", "tr") == "tr"
                else "Excel export requires openpyxl."
            )
        )
        return

    st.download_button(
        label=label,
        data=excel_data,
        file_name=filename,
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        key=key,
    )


def render_export_buttons(
    dataframe: pd.DataFrame,
    basename: str,
    csv_label: str = "CSV",
    excel_label: str = "Excel",
) -> None:
    """
    Render CSV and Excel export buttons side by side.
    """
    if dataframe.empty:
        return

    first, second = st.columns(
        2
    )

    with first:
        render_csv_download(
            dataframe=dataframe,
            filename=f"{basename}.csv",
            label=csv_label,
            key=f"{basename}_csv",
        )

    with second:
        render_excel_download(
            dataframe=dataframe,
            filename=f"{basename}.xlsx",
            label=excel_label,
            key=f"{basename}_excel",
        )
