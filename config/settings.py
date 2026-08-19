from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables from the project root
load_dotenv(BASE_DIR / ".env")


def get_env(name: str, default: str = "") -> str:
    """
    Return an environment variable as a stripped string.
    """
    value = os.getenv(name, default)

    if value is None:
        return default

    return str(value).strip()


def get_env_float(name: str, default: float) -> float:
    """
    Return an environment variable as a float.
    """
    value = os.getenv(name)

    if value is None or str(value).strip() == "":
        return float(default)

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain a numeric value. "
            f"Received: {value!r}"
        ) from exc


def get_env_int(name: str, default: int) -> int:
    """
    Return an environment variable as an integer.
    """
    value = os.getenv(name)

    if value is None or str(value).strip() == "":
        return int(default)

    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain an integer-compatible value. "
            f"Received: {value!r}"
        ) from exc


def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Return an environment variable as a boolean.
    """
    value = os.getenv(name)

    if value is None:
        return default

    normalized = str(value).strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"{name} must contain a boolean-compatible value. "
        f"Received: {value!r}"
    )


def resolve_project_path(value: str) -> str:
    """
    Resolve a relative path from the project root directory.
    """
    text = str(value).strip()

    if not text:
        return ""

    path = Path(text)

    if path.is_absolute():
        return str(path)

    return str((BASE_DIR / path).resolve())


@dataclass(frozen=True)
class Settings:
    # ========================================================
    # APPLICATION
    # ========================================================

    project_name: str = get_env(
        "PROJECT_NAME",
        "SEO Organic Growth Intelligence",
    )

    app_environment: str = get_env(
        "APP_ENV",
        "development",
    )

    log_level: str = get_env(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    # ========================================================
    # WEBSITE
    # ========================================================

    site_name: str = get_env(
        "SITE_NAME",
        "Example Site",
    )

    site_domain: str = get_env(
        "SITE_DOMAIN",
        "example.com",
    )

    default_brand: str = get_env(
        "DEFAULT_BRAND",
        "Example Brand",
    )

    # ========================================================
    # DATA SOURCE
    # Supported values: csv, api, hybrid
    # ========================================================

    data_source_mode: str = get_env(
        "DATA_SOURCE_MODE",
        "csv",
    ).lower()

    input_file: str = resolve_project_path(
        get_env(
            "SEO_INPUT_FILE",
            "./data/raw/seo_data.csv",
        )
    )

    output_dir: str = resolve_project_path(
        get_env(
            "SEO_OUTPUT_DIR",
            "./outputs",
        )
    )

    crawl_input_file: str = resolve_project_path(
        get_env(
            "CRAWL_INPUT_FILE",
            "",
        )
    )

    product_feed_file: str = resolve_project_path(
        get_env(
            "PRODUCT_FEED_FILE",
            "",
        )
    )

    # ========================================================
    # TECHNICAL CRAWL
    # ========================================================

    technical_crawl_enabled: bool = get_env_bool(
        "TECHNICAL_CRAWL_ENABLED",
        False,
    )

    crawl_start_url: str = get_env(
        "CRAWL_START_URL",
        "",
    )

    crawl_max_pages: int = get_env_int(
        "CRAWL_MAX_PAGES",
        500,
    )

    crawl_request_timeout: int = get_env_int(
        "CRAWL_REQUEST_TIMEOUT",
        15,
    )

    crawl_delay_seconds: float = get_env_float(
        "CRAWL_DELAY_SECONDS",
        0.25,
    )

    crawl_user_agent: str = get_env(
        "CRAWL_USER_AGENT",
        "SEOOrganicGrowthIntelligenceBot/1.0",
    )

    crawl_respect_robots: bool = get_env_bool(
        "CRAWL_RESPECT_ROBOTS",
        True,
    )

    # ========================================================
    # FULL-SITE URL INVENTORY / CRAWL ROTATION
    # ========================================================

    full_site_inventory_enabled: bool = get_env_bool(
        "FULL_SITE_INVENTORY_ENABLED",
        True,
    )

    sitemap_url: str = get_env(
        "SITEMAP_URL",
        "",
    )

    sitemap_max_urls: int = get_env_int(
        "SITEMAP_MAX_URLS",
        50000,
    )

    crawl_rotation_state_file: str = resolve_project_path(
        get_env(
            "CRAWL_ROTATION_STATE_FILE",
            "./outputs/crawl_rotation_state.json",
        )
    )

    # ========================================================
    # PAGESPEED INSIGHTS
    # ========================================================

    pagespeed_enabled: bool = get_env_bool(
        "PAGESPEED_ENABLED",
        False,
    )

    pagespeed_api_key: str = get_env(
        "PAGESPEED_API_KEY",
        "",
    )

    pagespeed_strategy: str = get_env(
        "PAGESPEED_STRATEGY",
        "mobile",
    ).lower()

    pagespeed_max_urls: int = get_env_int(
        "PAGESPEED_MAX_URLS",
        30,
    )

    pagespeed_request_timeout: int = get_env_int(
        "PAGESPEED_REQUEST_TIMEOUT",
        60,
    )

    pagespeed_delay_seconds: float = get_env_float(
        "PAGESPEED_DELAY_SECONDS",
        0.20,
    )

    pagespeed_freshness_days: int = get_env_int(
        "PAGESPEED_FRESHNESS_DAYS",
        14,
    )

    pagespeed_rotation_state_file: str = resolve_project_path(
        get_env(
            "PAGESPEED_ROTATION_STATE_FILE",
            "./outputs/pagespeed_rotation_state.json",
        )
    )

    # ========================================================
    # DATE RANGE
    #
    # Supported DATE_MODE values:
    # custom
    # last_30_days
    # last_60_days
    # last_90_days
    # last_180_days
    # last_365_days
    # ========================================================

    date_mode: str = get_env(
        "DATE_MODE",
        "last_60_days",
    ).lower()

    date_from: str = get_env(
        "DATE_FROM",
        "",
    )

    date_to: str = get_env(
        "DATE_TO",
        "",
    )

    api_data_delay_days: int = get_env_int(
        "API_DATA_DELAY_DAYS",
        2,
    )

    default_lookback_days: int = get_env_int(
        "DEFAULT_LOOKBACK_DAYS",
        90,
    )

    # ========================================================
    # GOOGLE SEARCH CONSOLE
    # ========================================================

    gsc_site_url: str = get_env(
        "GSC_SITE_URL",
        "",
    )

    gsc_service_account_file: str = resolve_project_path(
        get_env(
            "GSC_SERVICE_ACCOUNT_FILE",
            "./credentials/google_service_account.json",
        )
    )

    gsc_row_limit: int = get_env_int(
        "GSC_ROW_LIMIT",
        25000,
    )

    # ========================================================
    # GOOGLE ANALYTICS 4
    # ========================================================

    ga4_property_id: str = get_env(
        "GA4_PROPERTY_ID",
        "",
    )

    ga4_service_account_file: str = resolve_project_path(
        get_env(
            "GA4_SERVICE_ACCOUNT_FILE",
            "./credentials/google_service_account.json",
        )
    )

    ga4_row_limit: int = get_env_int(
        "GA4_ROW_LIMIT",
        250000,
    )

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    min_ml_rows: int = get_env_int(
        "MIN_ML_ROWS",
        20,
    )

    random_state: int = get_env_int(
        "ML_RANDOM_STATE",
        42,
    )

    test_size: float = get_env_float(
        "ML_TEST_SIZE",
        0.20,
    )

    n_estimators: int = get_env_int(
        "ML_N_ESTIMATORS",
        250,
    )

    max_depth: int = get_env_int(
        "ML_MAX_DEPTH",
        8,
    )

    min_samples_leaf: int = get_env_int(
        "ML_MIN_SAMPLES_LEAF",
        2,
    )

    value_per_click: float = get_env_float(
        "SEO_VALUE_PER_CLICK",
        0.50,
    )

    # ========================================================
    # LARGE LANGUAGE MODEL
    # Multi-provider: OpenAI / Anthropic / Gemini
    # ========================================================

    llm_enabled: bool = get_env_bool(
        "LLM_ENABLED",
        False,
    )

    llm_provider: str = get_env(
        "LLM_PROVIDER",
        "auto",
    ).lower()

    llm_model: str = get_env(
        "LLM_MODEL",
        "",
    )

    llm_language: str = get_env(
        "LLM_LANGUAGE",
        "tr",
    ).lower()

    llm_max_pages: int = get_env_int(
        "LLM_MAX_PAGES",
        20,
    )

    llm_max_tokens: int = get_env_int(
        "LLM_MAX_TOKENS",
        800,
    )

    llm_daily_request_limit: int = get_env_int(
        "LLM_DAILY_REQUEST_LIMIT",
        20,
    )

    llm_usage_file: str = resolve_project_path(
        get_env(
            "LLM_USAGE_FILE",
            "./outputs/llm_usage.json",
        )
    )

    llm_temperature: float = get_env_float(
        "LLM_TEMPERATURE",
        0.2,
    )

    anthropic_api_key: str = get_env(
        "ANTHROPIC_API_KEY",
        "",
    )

    openai_api_key: str = get_env(
        "OPENAI_API_KEY",
        "",
    )

    gemini_api_key: str = get_env(
        "GEMINI_API_KEY",
        "",
    )

    # ========================================================
    # RAG / VECTOR KNOWLEDGE MEMORY
    # Embeddings are independent from the chat LLM provider.
    # ========================================================

    rag_enabled: bool = get_env_bool(
        "RAG_ENABLED",
        False,
    )

    rag_embedding_provider: str = get_env(
        "RAG_EMBEDDING_PROVIDER",
        "local",
    ).lower()

    rag_embedding_model: str = get_env(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    rag_embedding_dimensions: int = get_env_int(
        "RAG_EMBEDDING_DIMENSIONS",
        384,
    )

    rag_chunk_size: int = get_env_int(
        "RAG_CHUNK_SIZE",
        800,
    )

    rag_chunk_overlap: int = get_env_int(
        "RAG_CHUNK_OVERLAP",
        150,
    )

    rag_top_k: int = get_env_int(
        "RAG_TOP_K",
        8,
    )

    rag_min_similarity: float = get_env_float(
        "RAG_MIN_SIMILARITY",
        0.25,
    )

    rag_batch_size: int = get_env_int(
        "RAG_BATCH_SIZE",
        50,
    )

    # ========================================================
    # SEO COST ASSUMPTIONS
    # ========================================================

    cost_maintain: float = get_env_float(
        "SEO_COST_MAINTAIN",
        0,
    )

    cost_title_meta: float = get_env_float(
        "SEO_COST_TITLE_META",
        40,
    )

    cost_content_refresh: float = get_env_float(
        "SEO_COST_CONTENT_REFRESH",
        120,
    )

    cost_internal_linking: float = get_env_float(
        "SEO_COST_INTERNAL_LINKING",
        60,
    )

    cost_default: float = get_env_float(
        "SEO_COST_DEFAULT_FALLBACK",
        80,
    )
    # ========================================================
    # SCHEDULED PIPELINE
    # ========================================================

    scheduler_enabled: bool = get_env_bool(
        "SCHEDULER_ENABLED",
        False,
    )

    scheduler_timezone: str = get_env(
        "SCHEDULER_TIMEZONE",
        "Europe/Istanbul",
    )

    scheduler_hour: int = get_env_int(
        "SCHEDULER_HOUR",
        6,
    )

    scheduler_minute: int = get_env_int(
        "SCHEDULER_MINUTE",
        0,
    )

    scheduler_poll_seconds: int = get_env_int(
        "SCHEDULER_POLL_SECONDS",
        30,
    )

    scheduler_run_on_start: bool = get_env_bool(
        "SCHEDULER_RUN_ON_START",
        False,
    )

    scheduler_stale_lock_minutes: int = get_env_int(
        "SCHEDULER_STALE_LOCK_MINUTES",
        240,
    )

    scheduler_lock_file: str = resolve_project_path(
        get_env(
            "SCHEDULER_LOCK_FILE",
            "./outputs/scheduler.lock",
        )
    )

    scheduler_status_file: str = resolve_project_path(
        get_env(
            "SCHEDULER_STATUS_FILE",
            "./outputs/scheduler_status.json",
        )
    )
    
    # ========================================================
    # POSTGRESQL
    # ========================================================

    postgres_enabled: bool = get_env_bool(
        "POSTGRES_ENABLED",
        False,
    )

    postgres_if_exists: str = get_env(
        "POSTGRES_IF_EXISTS",
        "replace",
    )

    postgres_host: str = get_env(
        "POSTGRES_HOST",
        "localhost",
    )

    postgres_port: int = get_env_int(
        "POSTGRES_PORT",
        5432,
    )

    postgres_database: str = get_env(
        "POSTGRES_DATABASE",
        "seo_intelligence",
    )

    postgres_user: str = get_env(
        "POSTGRES_USER",
        "postgres",
    )

    postgres_password: str = get_env(
        "POSTGRES_PASSWORD",
        "",
    )

    def validate(self) -> None:
        """
        Validate the main application settings.
        """
        allowed_data_source_modes = {
            "csv",
            "api",
            "hybrid",
        }

        if self.data_source_mode not in allowed_data_source_modes:
            raise ValueError(
                "DATA_SOURCE_MODE must be one of: "
                f"{sorted(allowed_data_source_modes)}. "
                f"Received: {self.data_source_mode!r}"
            )

        allowed_date_modes = {
            "custom",
            "last_30_days",
            "last_60_days",
            "last_90_days",
            "last_180_days",
            "last_365_days",
        }

        if self.date_mode not in allowed_date_modes:
            raise ValueError(
                "DATE_MODE must be one of: "
                f"{sorted(allowed_date_modes)}. "
                f"Received: {self.date_mode!r}"
            )

        if self.date_mode == "custom":
            if not self.date_from or not self.date_to:
                raise ValueError(
                    "DATE_FROM and DATE_TO are required "
                    "when DATE_MODE is custom."
                )

        if self.api_data_delay_days < 0:
            raise ValueError(
                "API_DATA_DELAY_DAYS cannot be negative."
            )

        if self.default_lookback_days < 1:
            raise ValueError(
                "DEFAULT_LOOKBACK_DAYS must be at least 1."
            )

        if not 0 < self.test_size < 1:
            raise ValueError(
                "ML_TEST_SIZE must be greater than 0 and lower than 1."
            )

        if self.min_ml_rows < 2:
            raise ValueError(
                "MIN_ML_ROWS must be at least 2."
            )

        if self.n_estimators < 1:
            raise ValueError(
                "ML_N_ESTIMATORS must be at least 1."
            )

        if self.max_depth < 1:
            raise ValueError(
                "ML_MAX_DEPTH must be at least 1."
            )

        if self.min_samples_leaf < 1:
            raise ValueError(
                "ML_MIN_SAMPLES_LEAF must be at least 1."
            )

        if self.gsc_row_limit < 1:
            raise ValueError(
                "GSC_ROW_LIMIT must be at least 1."
            )

        if self.ga4_row_limit < 1:
            raise ValueError(
                "GA4_ROW_LIMIT must be at least 1."
            )

        if self.crawl_max_pages < 1:
            raise ValueError(
                "CRAWL_MAX_PAGES must be at least 1."
            )

        if self.crawl_request_timeout < 1:
            raise ValueError(
                "CRAWL_REQUEST_TIMEOUT must be at least 1."
            )

        if self.crawl_delay_seconds < 0:
            raise ValueError(
                "CRAWL_DELAY_SECONDS cannot be negative."
            )

        if self.sitemap_max_urls < 1:
            raise ValueError(
                "SITEMAP_MAX_URLS must be at least 1."
            )

        if self.pagespeed_strategy not in {"mobile", "desktop"}:
            raise ValueError(
                "PAGESPEED_STRATEGY must be mobile or desktop."
            )

        if self.pagespeed_max_urls < 1:
            raise ValueError(
                "PAGESPEED_MAX_URLS must be at least 1."
            )

        if self.pagespeed_request_timeout < 1:
            raise ValueError(
                "PAGESPEED_REQUEST_TIMEOUT must be at least 1."
            )

        if self.pagespeed_delay_seconds < 0:
            raise ValueError(
                "PAGESPEED_DELAY_SECONDS cannot be negative."
            )

        if self.pagespeed_freshness_days < 1:
            raise ValueError(
                "PAGESPEED_FRESHNESS_DAYS must be at least 1."
            )

        allowed_llm_providers = {
            "auto",
            "anthropic",
            "openai",
            "gemini",
        }

        if self.llm_provider not in allowed_llm_providers:
            raise ValueError(
                "LLM_PROVIDER must be one of: "
                f"{sorted(allowed_llm_providers)}. "
                f"Received: {self.llm_provider!r}"
            )

        if self.llm_max_pages < 1:
            raise ValueError(
                "LLM_MAX_PAGES must be at least 1."
            )

        if self.llm_max_tokens < 1:
            raise ValueError(
                "LLM_MAX_TOKENS must be at least 1."
            )

        if self.llm_daily_request_limit < 1:
            raise ValueError(
                "LLM_DAILY_REQUEST_LIMIT must be at least 1."
            )

        if not 0 <= self.llm_temperature <= 2:
            raise ValueError(
                "LLM_TEMPERATURE must be between 0 and 2."
            )

        allowed_rag_embedding_providers = {
            "local",
        }

        if (
            self.rag_embedding_provider
            not in allowed_rag_embedding_providers
        ):
            raise ValueError(
                "RAG_EMBEDDING_PROVIDER must be one of: "
                f"{sorted(allowed_rag_embedding_providers)}. "
                f"Received: {self.rag_embedding_provider!r}"
            )

        if self.rag_embedding_dimensions < 1:
            raise ValueError(
                "RAG_EMBEDDING_DIMENSIONS must be at least 1."
            )

        if self.rag_chunk_size < 1:
            raise ValueError(
                "RAG_CHUNK_SIZE must be at least 1."
            )

        if self.rag_chunk_overlap < 0:
            raise ValueError(
                "RAG_CHUNK_OVERLAP cannot be negative."
            )

        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be smaller than "
                "RAG_CHUNK_SIZE."
            )

        if self.rag_top_k < 1:
            raise ValueError(
                "RAG_TOP_K must be at least 1."
            )

        if not 0 <= self.rag_min_similarity <= 1:
            raise ValueError(
                "RAG_MIN_SIMILARITY must be between 0 and 1."
            )

        if self.rag_batch_size < 1:
            raise ValueError(
                "RAG_BATCH_SIZE must be at least 1."
            )

        if self.rag_enabled and not self.postgres_enabled:
            raise ValueError(
                "POSTGRES_ENABLED must be true when "
                "RAG_ENABLED is true."
            )

        if self.data_source_mode in {"api", "hybrid"}:
            if not self.gsc_site_url:
                raise ValueError(
                    "GSC_SITE_URL is required when "
                    "DATA_SOURCE_MODE is api or hybrid."
                )

            if not self.ga4_property_id:
                raise ValueError(
                    "GA4_PROPERTY_ID is required when "
                    "DATA_SOURCE_MODE is api or hybrid."
                )

            if not self.gsc_service_account_file:
                raise ValueError(
                    "GSC_SERVICE_ACCOUNT_FILE is required when "
                    "DATA_SOURCE_MODE is api or hybrid."
                )

            if not self.ga4_service_account_file:
                raise ValueError(
                    "GA4_SERVICE_ACCOUNT_FILE is required when "
                    "DATA_SOURCE_MODE is api or hybrid."
                )

        if not self.scheduler_timezone:
            raise ValueError(
                "SCHEDULER_TIMEZONE cannot be empty."
            )

        if not 0 <= self.scheduler_hour <= 23:
            raise ValueError(
                "SCHEDULER_HOUR must be between 0 and 23."
            )

        if not 0 <= self.scheduler_minute <= 59:
            raise ValueError(
                "SCHEDULER_MINUTE must be between 0 and 59."
            )

        if self.scheduler_poll_seconds < 1:
            raise ValueError(
                "SCHEDULER_POLL_SECONDS must be at least 1."
            )

        if self.scheduler_stale_lock_minutes < 1:
            raise ValueError(
                "SCHEDULER_STALE_LOCK_MINUTES must be at least 1."
            )

        if not self.scheduler_lock_file:
            raise ValueError(
                "SCHEDULER_LOCK_FILE cannot be empty."
            )

        if not self.scheduler_status_file:
            raise ValueError(
                "SCHEDULER_STATUS_FILE cannot be empty."
            )

        if self.postgres_if_exists not in {
            "fail",
            "replace",
            "append",
        }:
            raise ValueError(
                "POSTGRES_IF_EXISTS must be fail, replace, or append."
            )

    @property
    def postgres_url(self) -> str:
        """
        Build the SQLAlchemy PostgreSQL connection URL.
        """
        return (
            "postgresql+psycopg2://"
            f"{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_database}"
        )


SETTINGS = Settings()


# ============================================================
# DASHBOARD COMPATIBILITY EXPORTS
# ============================================================
# Dashboard modules use the same module-level configuration
# pattern as the Ads Budget Intelligence project.
# Canonical value still lives in the immutable SETTINGS object.

OUTPUT_DIR = SETTINGS.output_dir


# ============================================================
# LLM COMPATIBILITY EXPORTS
# ============================================================
# Ads-style src.llm modules import module-level constants.
# Canonical values still live in the immutable SETTINGS object.

ANTHROPIC_API_KEY = SETTINGS.anthropic_api_key
OPENAI_API_KEY = SETTINGS.openai_api_key
GEMINI_API_KEY = SETTINGS.gemini_api_key

LLM_ENABLED = SETTINGS.llm_enabled
LLM_LANGUAGE = SETTINGS.llm_language
LLM_MAX_PAGES = SETTINGS.llm_max_pages
LLM_MAX_TOKENS = SETTINGS.llm_max_tokens
LLM_DAILY_REQUEST_LIMIT = SETTINGS.llm_daily_request_limit
LLM_USAGE_FILE = SETTINGS.llm_usage_file
LLM_TEMPERATURE = SETTINGS.llm_temperature


def resolve_llm_provider() -> str:
    """
    Resolve the active LLM provider.

    When LLM_PROVIDER=auto, select the provider based on
    the available API key.
    """
    configured_provider = (
        SETTINGS.llm_provider
        .strip()
        .lower()
    )

    if configured_provider != "auto":
        return configured_provider

    if OPENAI_API_KEY:
        return "openai"

    if ANTHROPIC_API_KEY:
        return "anthropic"

    if GEMINI_API_KEY:
        return "gemini"

    return ""


# manager.py expects a concrete provider name,
# not the literal "auto".
LLM_PROVIDER = resolve_llm_provider()


def resolve_llm_model(
    provider: str | None = None,
) -> str:
    """
    Resolve the configured model or a provider-specific default.
    """
    configured_model = (
        SETTINGS.llm_model
        .strip()
    )

    if configured_model:
        return configured_model

    active_provider = (
        provider
        or LLM_PROVIDER
        or resolve_llm_provider()
    )

    default_models = {
        "openai": "gpt-5-mini",
        "anthropic": "claude-sonnet-4-6",
        "gemini": "gemini-2.5-flash",
    }

    return default_models.get(
        active_provider,
        "",
    )


LLM_MODEL = resolve_llm_model()

RAG_ENABLED = SETTINGS.rag_enabled
RAG_EMBEDDING_PROVIDER = SETTINGS.rag_embedding_provider
RAG_EMBEDDING_MODEL = SETTINGS.rag_embedding_model
RAG_EMBEDDING_DIMENSIONS = SETTINGS.rag_embedding_dimensions
RAG_CHUNK_SIZE = SETTINGS.rag_chunk_size
RAG_CHUNK_OVERLAP = SETTINGS.rag_chunk_overlap
RAG_TOP_K = SETTINGS.rag_top_k
RAG_MIN_SIMILARITY = SETTINGS.rag_min_similarity
RAG_BATCH_SIZE = SETTINGS.rag_batch_size

# ============================================================
# SCHEDULER COMPATIBILITY EXPORTS
# ============================================================

SCHEDULER_ENABLED = SETTINGS.scheduler_enabled
SCHEDULER_TIMEZONE = SETTINGS.scheduler_timezone
SCHEDULER_HOUR = SETTINGS.scheduler_hour
SCHEDULER_MINUTE = SETTINGS.scheduler_minute
SCHEDULER_POLL_SECONDS = SETTINGS.scheduler_poll_seconds
SCHEDULER_RUN_ON_START = SETTINGS.scheduler_run_on_start
SCHEDULER_STALE_LOCK_MINUTES = SETTINGS.scheduler_stale_lock_minutes
SCHEDULER_LOCK_FILE = SETTINGS.scheduler_lock_file
SCHEDULER_STATUS_FILE = SETTINGS.scheduler_status_file


def llm_ready() -> bool:
    """
    Return whether the configured LLM can make a real API call.
    """
    if not LLM_ENABLED:
        return False

    if not LLM_PROVIDER:
        return False

    if not LLM_MODEL:
        return False

    if LLM_PROVIDER == "openai":
        return bool(
            OPENAI_API_KEY
        )

    if LLM_PROVIDER == "anthropic":
        return bool(
            ANTHROPIC_API_KEY
        )

    if LLM_PROVIDER == "gemini":
        return bool(
            GEMINI_API_KEY
        )

    return False
