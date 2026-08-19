from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config.settings import SETTINGS
from dashboard.app_config import (
    OUTPUT_DIR,
    PROJECT_ROOT,
)


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class AnalysisRunResult:
    success: bool
    message: str
    run_directory: Path | None = None
    log_tail: str = ""


# ============================================================
# SOURCE READINESS
# ============================================================

def _credential_exists(
    path_value: str | None,
) -> bool:
    """
    Check whether a configured credential file exists.
    """

    if not path_value:
        return False

    try:
        return Path(path_value).exists()
    except Exception:
        return False


def gsc_ready() -> bool:
    """
    Return whether Google Search Console API configuration
    appears ready without exposing credentials.
    """

    if SETTINGS.data_source_mode not in {
        "api",
        "hybrid",
    }:
        return False

    return bool(
        SETTINGS.gsc_site_url
        and SETTINGS.gsc_service_account_file
        and _credential_exists(
            SETTINGS.gsc_service_account_file
        )
    )


def ga4_ready() -> bool:
    """
    Return whether GA4 API configuration appears ready
    without exposing credentials.
    """

    if SETTINGS.data_source_mode not in {
        "api",
        "hybrid",
    }:
        return False

    return bool(
        SETTINGS.ga4_property_id
        and SETTINGS.ga4_service_account_file
        and _credential_exists(
            SETTINGS.ga4_service_account_file
        )
    )


def get_source_status() -> dict[str, bool]:
    """
    Return configuration readiness without exposing
    API keys or credentials.
    """

    return {
        "gsc": gsc_ready(),
        "ga4": ga4_ready(),
    }


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def _read_csv_safe(
    path: Path,
) -> pd.DataFrame:
    """
    Safely read a CSV file.
    """

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def _validate_daily_output(
    staging_directory: Path,
    start_date: date,
    end_date: date,
) -> tuple[bool, str]:
    """
    Validate the generated daily SEO output.

    The output must:
    - exist
    - contain data
    - contain a valid date column
    - stay inside the requested analysis period
    """

    daily_path = (
        staging_directory
        / "seo_daily_performance.csv"
    )

    if not daily_path.exists():
        return (
            False,
            "SEO günlük performans çıktısı oluşturulamadı.",
        )

    daily = _read_csv_safe(
        daily_path
    )

    if daily.empty:
        return (
            False,
            "Seçilen dönem için SEO günlük performans verisi bulunamadı.",
        )

    date_column = None

    for candidate in (
        "Date",
        "date",
    ):
        if candidate in daily.columns:
            date_column = candidate
            break

    if date_column is None:
        return (
            False,
            "SEO günlük performans çıktısında tarih alanı bulunamadı.",
        )

    daily_dates = pd.to_datetime(
        daily[date_column],
        errors="coerce",
    ).dropna()

    if daily_dates.empty:
        return (
            False,
            "SEO günlük performans çıktısında geçerli tarih bulunamadı.",
        )

    minimum_date = (
        daily_dates.min().date()
    )

    maximum_date = (
        daily_dates.max().date()
    )

    if (
        minimum_date < start_date
        or maximum_date > end_date
    ):
        return (
            False,
            (
                "GSC/GA4 çıktısındaki tarihler "
                "seçilen analiz dönemiyle uyuşmuyor."
            ),
        )

    return True, ""


def _validate_required_outputs(
    staging_directory: Path,
) -> tuple[bool, str]:
    """
    Validate core SEO decision-intelligence outputs.
    """

    required_outputs = {
        "seo_integrated_data.csv":
            "GSC + GA4 entegre veri çıktısı",
        "seo_scenario_simulation.csv":
            "SEO senaryo simülasyonu",
        "seo_recommendations.csv":
            "SEO öneri çıktısı",
        "seo_model_metrics.csv":
            "SEO model metrikleri",
    }

    for file_name, label in required_outputs.items():

        path = (
            staging_directory
            / file_name
        )

        if not path.exists():
            return (
                False,
                f"{label} oluşturulamadı.",
            )

        dataframe = _read_csv_safe(
            path
        )

        if dataframe.empty:
            return (
                False,
                f"{label} boş oluşturuldu.",
            )

    return True, ""


def _validate_staged_outputs(
    staging_directory: Path,
    start_date: date,
    end_date: date,
) -> tuple[bool, str]:
    """
    Validate all staged pipeline outputs before promotion.
    """

    valid, message = _validate_daily_output(
        staging_directory,
        start_date,
        end_date,
    )

    if not valid:
        return valid, message

    return _validate_required_outputs(
        staging_directory
    )


# ============================================================
# MANIFEST
# ============================================================

def _write_manifest(
    directory: Path,
    start_date: date,
    end_date: date,
) -> None:
    """
    Write metadata describing the successful analysis run.
    """

    manifest = {
        "analysis_start_date":
            start_date.isoformat(),

        "analysis_end_date":
            end_date.isoformat(),

        "completed_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "source":
            "dashboard",

        "pipeline":
            "seo-organic-growth-intelligence",

        "data_sources": [
            "google_search_console",
            "ga4",
        ],
    }

    (
        directory
        / "analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# OUTPUT PROMOTION
# ============================================================

def _promote_outputs(
    staging_directory: Path,
    start_date: date,
    end_date: date,
) -> Path:
    """
    Promote validated staging outputs into:
    - output/runs/<period>
    - output/
    """

    period_key = (
        f"{start_date.isoformat()}_"
        f"{end_date.isoformat()}"
    )

    run_directory = (
        Path(OUTPUT_DIR)
        / "runs"
        / period_key
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_directory = Path(
        OUTPUT_DIR
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for source in staging_directory.iterdir():

        if not source.is_file():
            continue

        shutil.copy2(
            source,
            run_directory / source.name,
        )

        shutil.copy2(
            source,
            output_directory / source.name,
        )

    _write_manifest(
        run_directory,
        start_date,
        end_date,
    )

    _write_manifest(
        output_directory,
        start_date,
        end_date,
    )

    return run_directory


# ============================================================
# RUN PIPELINE
# ============================================================

def run_analysis_for_period(
    start_date: date,
    end_date: date,
    timeout_seconds: int = 1800,
) -> AnalysisRunResult:
    """
    Run the SEO pipeline for a selected dashboard period.

    Pipeline outputs are first generated in a temporary
    staging directory.

    Existing dashboard outputs are replaced only after
    successful validation.
    """

    original_start_date = start_date
    original_end_date = end_date

    start_date = min(
        original_start_date,
        original_end_date,
    )

    end_date = max(
        original_start_date,
        original_end_date,
    )

    # --------------------------------------------------------
    # SOURCE VALIDATION
    # --------------------------------------------------------

    source_status = get_source_status()

    if not source_status["gsc"]:
        return AnalysisRunResult(
            success=False,
            message=(
                "Google Search Console API yapılandırması "
                "eksik veya service-account dosyasına "
                "erişilemiyor."
            ),
        )

    if not source_status["ga4"]:
        return AnalysisRunResult(
            success=False,
            message=(
                "GA4 API yapılandırması eksik veya "
                "service-account dosyasına erişilemiyor."
            ),
        )

    # --------------------------------------------------------
    # STAGING
    # --------------------------------------------------------

    output_directory = Path(
        OUTPUT_DIR
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = (
        output_directory
        / ".staging"
    )

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_directory = (
        staging_root
        / uuid.uuid4().hex
    )

    staging_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # PIPELINE ENVIRONMENT
    # --------------------------------------------------------

    environment = os.environ.copy()

    environment.update(
        {
            "DATE_MODE": "custom",

            "DATE_FROM":
                start_date.isoformat(),

            "DATE_TO":
                end_date.isoformat(),

            "SEO_OUTPUT_DIR":
                str(staging_directory),

            "PYTHONUNBUFFERED": "1",
        }
    )

    command = [
        sys.executable,
        str(
            Path(PROJECT_ROOT)
            / "main.py"
        ),
    ]

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    try:

        completed = subprocess.run(
            command,
            cwd=Path(
                PROJECT_ROOT
            ),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )

    except subprocess.TimeoutExpired:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        return AnalysisRunResult(
            success=False,
            message=(
                "SEO analizi zaman aşımına uğradı."
            ),
        )

    except Exception as exc:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        return AnalysisRunResult(
            success=False,
            message=(
                "SEO pipeline başlatılamadı: "
                f"{exc}"
            ),
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    combined_log = "\n".join(
        part
        for part in (
            completed.stdout,
            completed.stderr,
        )
        if part
    )

    log_tail = "\n".join(
        combined_log.splitlines()[-40:]
    )

    # --------------------------------------------------------
    # PROCESS FAILURE
    # --------------------------------------------------------

    if completed.returncode != 0:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        return AnalysisRunResult(
            success=False,
            message=(
                "SEO pipeline hata ile tamamlandı. "
                "Ayrıntılar için çalışma kaydını "
                "kontrol edin."
            ),
            log_tail=log_tail,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid, validation_message = (
        _validate_staged_outputs(
            staging_directory,
            start_date,
            end_date,
        )
    )

    if not valid:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        return AnalysisRunResult(
            success=False,
            message=validation_message,
            log_tail=log_tail,
        )

    # --------------------------------------------------------
    # PROMOTE OUTPUTS
    # --------------------------------------------------------

    try:

        run_directory = _promote_outputs(
            staging_directory,
            start_date,
            end_date,
        )

    except Exception as exc:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

        return AnalysisRunResult(
            success=False,
            message=(
                "Analiz çıktıları kaydedilirken "
                f"hata oluştu: {exc}"
            ),
            log_tail=log_tail,
        )

    finally:

        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return AnalysisRunResult(
        success=True,
        message=(
            f"{start_date:%d.%m.%Y}"
            "–"
            f"{end_date:%d.%m.%Y} "
            "dönemi başarıyla analiz edildi."
        ),
        run_directory=run_directory,
        log_tail=log_tail,
    )