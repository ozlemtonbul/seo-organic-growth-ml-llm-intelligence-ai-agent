from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_output_dir() -> Path:
    try:
        from config.settings import OUTPUT_DIR
        path = Path(OUTPUT_DIR).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    except Exception:
        return (PROJECT_ROOT / "outputs").resolve()


OUTPUT_DIR = _resolve_output_dir()


def _csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        pytest.fail(f"Required pipeline output is missing: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def output_dir() -> Path:
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def integrated() -> pd.DataFrame:
    return _csv("seo_integrated_data.csv")


@pytest.fixture(scope="session")
def scenarios() -> pd.DataFrame:
    return _csv("seo_scenario_simulation.csv")


@pytest.fixture(scope="session")
def recommendations() -> pd.DataFrame:
    return _csv("seo_recommendations.csv")


@pytest.fixture(scope="session")
def model_metrics() -> pd.DataFrame:
    return _csv("seo_model_metrics.csv")


def pytest_report_header(config):
    return [
        f"SEO Final QA project root: {PROJECT_ROOT}",
        f"SEO Final QA output dir: {OUTPUT_DIR}",
    ]
