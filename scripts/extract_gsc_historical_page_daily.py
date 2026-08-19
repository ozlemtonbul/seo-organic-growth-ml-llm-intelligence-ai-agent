from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS


GSC_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
ROW_LIMIT = 25_000
DEFAULT_START_DATE = date(2023, 8, 2)
WINDOW_DAYS = 7


@dataclass(frozen=True)
class ExtractionResult:
    output_path: Path
    requested_start: date
    requested_end: date
    actual_start: Optional[pd.Timestamp]
    actual_end: Optional[pd.Timestamp]
    rows: int
    pages: int
    observed_days: int
    calendar_days: int


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_end_date() -> date:
    integrated = PROJECT_ROOT / "outputs" / "seo_integrated_data.csv"

    if integrated.exists():
        try:
            dates = pd.read_csv(
                integrated,
                usecols=["date"],
                low_memory=False,
            )["date"]
            parsed = pd.to_datetime(dates, errors="coerce").dropna()
            if not parsed.empty:
                return parsed.max().date()
        except Exception:
            pass

    return date.today() - timedelta(days=2)


def _build_service(credentials_path: Path):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google API kutuphaneleri bulunamadi. Mevcut proje GSC extractor "
            "ortaminda google-api-python-client ve google-auth kurulu olmali."
        ) from exc

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=[GSC_SCOPE],
    )

    return build(
        "searchconsole",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


def _retry_execute(request, max_attempts: int = 6):
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        HttpError = Exception

    for attempt in range(1, max_attempts + 1):
        try:
            return request.execute()
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            retryable = status in {429, 500, 502, 503, 504}
            if not retryable or attempt >= max_attempts:
                raise
            wait_seconds = min(2 ** attempt, 30)
            print(
                f"[WARN] GSC API gecici hata status={status}; "
                f"{wait_seconds}s sonra tekrar denenecek."
            )
            time.sleep(wait_seconds)


def _query_window(
    service,
    site_url: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    rows = []
    start_row = 0

    while True:
        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["date", "page"],
            "type": "web",
            "dataState": "final",
            "rowLimit": ROW_LIMIT,
            "startRow": start_row,
        }

        response = _retry_execute(
            service.searchanalytics().query(
                siteUrl=site_url,
                body=body,
            )
        )

        batch = response.get("rows", []) or []

        for item in batch:
            keys = item.get("keys", [])
            if len(keys) < 2:
                continue

            rows.append(
                {
                    "date": keys[0],
                    "page": keys[1],
                    "clicks": float(item.get("clicks", 0.0) or 0.0),
                    "impressions": float(item.get("impressions", 0.0) or 0.0),
                    "ctr": float(item.get("ctr", 0.0) or 0.0),
                    "position": float(item.get("position", 0.0) or 0.0),
                }
            )

        if len(batch) < ROW_LIMIT:
            break

        start_row += ROW_LIMIT

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "page",
                "clicks",
                "impressions",
                "ctr",
                "position",
            ]
        )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    ).dt.date.astype(str)

    return result


def _date_windows(
    start_date: date,
    end_date: date,
    window_days: int = WINDOW_DAYS,
) -> Iterable[tuple[date, date]]:
    current = start_date

    while current <= end_date:
        window_end = min(
            current + timedelta(days=window_days - 1),
            end_date,
        )
        yield current, window_end
        current = window_end + timedelta(days=1)


def _part_path(parts_dir: Path, start_date: date, end_date: date) -> Path:
    return parts_dir / (
        f"gsc_page_daily_{start_date.isoformat()}_{end_date.isoformat()}.csv"
    )


def _validate_combined(dataframe: pd.DataFrame) -> dict[str, object]:
    if dataframe.empty:
        raise RuntimeError("GSC API sifir satir dondurdu.")

    required = {
        "date",
        "page",
        "clicks",
        "impressions",
        "ctr",
        "position",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise RuntimeError(f"Historical GSC output eksik kolonlar: {missing}")

    result = dataframe.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date", "page"]).copy()

    if result.duplicated(["date", "page"]).any():
        duplicate_count = int(result.duplicated(["date", "page"]).sum())
        raise RuntimeError(
            f"Historical GSC output date+page duplicate iceriyor: {duplicate_count}"
        )

    for column in ("clicks", "impressions", "position"):
        values = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        if (values < 0).any():
            raise RuntimeError(f"Negatif deger bulundu: {column}")

    min_date = result["date"].min()
    max_date = result["date"].max()
    calendar_days = int((max_date - min_date).days) + 1

    return {
        "rows": int(len(result)),
        "pages": int(result["page"].nunique()),
        "observed_days": int(result["date"].nunique()),
        "calendar_days": calendar_days,
        "min_date": min_date,
        "max_date": max_date,
    }


def extract_history(
    start_date: date,
    end_date: date,
    output_path: Path,
    reset_parts: bool = False,
) -> ExtractionResult:
    site_url = str(getattr(SETTINGS, "gsc_site_url", "") or "").strip()
    credential_value = str(
        getattr(SETTINGS, "gsc_service_account_file", "") or ""
    ).strip()

    if not site_url:
        raise RuntimeError(
            "SETTINGS.gsc_site_url bos. Mevcut .env/config GSC ayarini kontrol et."
        )

    if not credential_value:
        raise RuntimeError(
            "SETTINGS.gsc_service_account_file bos. Mevcut .env/config credential ayarini kontrol et."
        )

    credentials_path = _resolve_project_path(credential_value)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"GSC service-account dosyasi bulunamadi: {credentials_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output_path.parent / "gsc_history_parts"

    if reset_parts and parts_dir.exists():
        import shutil
        shutil.rmtree(parts_dir)

    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] GSC property: {site_url}")
    print(f"[INFO] Requested range: {start_date} -> {end_date}")
    print(f"[INFO] Window size: {WINDOW_DAYS} days | rowLimit={ROW_LIMIT}")
    print("[INFO] Existing completed window files will be reused.")

    service = _build_service(credentials_path)
    windows = list(_date_windows(start_date, end_date))

    for index, (window_start, window_end) in enumerate(windows, start=1):
        part = _part_path(parts_dir, window_start, window_end)

        if part.exists():
            print(
                f"[{index}/{len(windows)}] SKIP {window_start} -> {window_end} "
                f"(checkpoint exists)"
            )
            continue

        frame = _query_window(
            service=service,
            site_url=site_url,
            start_date=window_start,
            end_date=window_end,
        )

        frame.to_csv(part, index=False)
        print(
            f"[{index}/{len(windows)}] OK {window_start} -> {window_end} | "
            f"rows={len(frame):,}"
        )

    part_files = sorted(parts_dir.glob("gsc_page_daily_*.csv"))
    frames = []

    for part in part_files:
        try:
            frame = pd.read_csv(part, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("Historical GSC checkpoint dosyalari veri icermiyor.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(
        combined["date"],
        errors="coerce",
    ).dt.date.astype(str)

    combined = (
        combined
        .dropna(subset=["date", "page"])
        .drop_duplicates(["date", "page"], keep="last")
        .sort_values(["date", "page"])
        .reset_index(drop=True)
    )

    metrics = _validate_combined(combined)
    combined.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "actual_start": str(metrics["min_date"].date()),
        "actual_end": str(metrics["max_date"].date()),
        "rows": metrics["rows"],
        "pages": metrics["pages"],
        "observed_days": metrics["observed_days"],
        "calendar_days": metrics["calendar_days"],
        "dimensions": ["date", "page"],
        "data_state": "final",
        "search_type": "web",
        "method": "SearchConsoleAPI-page-date",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("HISTORICAL GSC EXTRACTION COMPLETE")
    print("=" * 84)
    print(f"[OUTPUT] {output_path}")
    print(f"[INFO] Actual range: {metadata['actual_start']} -> {metadata['actual_end']}")
    print(f"[INFO] Calendar coverage: {metadata['calendar_days']:,} days")
    print(f"[INFO] Observed days: {metadata['observed_days']:,}")
    print(f"[INFO] Rows: {metadata['rows']:,} | Pages: {metadata['pages']:,}")

    return ExtractionResult(
        output_path=output_path,
        requested_start=start_date,
        requested_end=end_date,
        actual_start=metrics["min_date"],
        actual_end=metrics["max_date"],
        rows=int(metrics["rows"]),
        pages=int(metrics["pages"]),
        observed_days=int(metrics["observed_days"]),
        calendar_days=int(metrics["calendar_days"]),
    )


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="Default: latest date in outputs/seo_integrated_data.csv, otherwise today-2.",
    )
    parser.add_argument(
        "--output",
        default="",
    )
    parser.add_argument(
        "--reset-parts",
        action="store_true",
    )
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = (
        _parse_date(args.end_date)
        if args.end_date.strip()
        else _resolve_end_date()
    )

    if start_date > end_date:
        print("[FAIL] start-date end-date'den buyuk olamaz.")
        return 1

    default_output = (
        PROJECT_ROOT
        / "data"
        / "historical"
        / (
            f"gsc_page_daily_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
        )
    )

    output_path = (
        _resolve_project_path(args.output)
        if args.output.strip()
        else default_output
    )

    try:
        result = extract_history(
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            reset_parts=bool(args.reset_parts),
        )
    except Exception as exc:
        print()
        print(f"[FAIL] Historical GSC extraction: {exc}")
        return 1

    if result.calendar_days >= 730:
        print("[PASS] 365-day strategic backtest icin 730+ gun coverage var.")
    elif result.calendar_days >= 360:
        print("[PASS] 90 ve 180 gun strategic backtest icin coverage var.")
        print("[WARN] 365 gun backtest icin 730+ gun gerektigi icin bu source yetmiyor.")
    elif result.calendar_days >= 180:
        print("[PASS] 90 gun strategic backtest icin coverage var.")
        print("[WARN] 180/365 gun backtest icin tarihsel coverage yetersiz.")
    else:
        print("[WARN] 90/180/365 strategic backtest icin tarihsel coverage yetersiz.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
