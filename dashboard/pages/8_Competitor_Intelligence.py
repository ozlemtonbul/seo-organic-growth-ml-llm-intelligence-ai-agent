from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.app_config import APP_TITLE
from dashboard.i18n import t
from dashboard.layout import initialize_dashboard, localized_text
from dashboard.services.analysis_service import (
    get_available_date_bounds,
    load_analysis_data,
)

base = load_analysis_data()
_, available_end = get_available_date_bounds(base.daily)
initial_language = st.session_state.get("dashboard_language", "tr")

context = initialize_dashboard(
    page_title=(
        f"{t('app_title', initial_language)} - "
        f"{'Rakip Zekâsı' if initial_language == 'tr' else 'Competitor Intelligence'}"
    ),
    page_icon="🔭",
    title="Rakip Zekâsı" if initial_language == "tr" else "Competitor Intelligence",
    subtitle=(
        "Rakip ve SERP verisini canlı veri kaynağı bağlandığında analiz edecek entegrasyon katmanıdır. Veri kaynağı yokken sonuç uydurmaz."
        if initial_language == "tr"
        else "Integration layer for live competitor and SERP intelligence. It never fabricates competitor findings when no source is connected."
    ),
    eyebrow="SEO & GEO KARAR ZEKÂSI" if initial_language == "tr" else "SEO & GEO DECISION INTELLIGENCE",
    default_preset="last_30_days",
    default_comparison="no_comparison",
    reference_date=available_end,
)

language = context.language

enabled = os.getenv("DATAFORSEO_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
login = bool(os.getenv("DATAFORSEO_LOGIN", "").strip())
password = bool(os.getenv("DATAFORSEO_PASSWORD", "").strip())
connected = enabled and login and password

st.subheader(localized_text(language, "Entegrasyon Durumu", "Integration Status"))

status_icon = "🟢" if connected else "🔴"
connection_text = localized_text(
    language,
    "Bağlı / Hazır" if connected else "Bağlı Değil",
    "Connected / Ready" if connected else "Not Connected",
)
data_text = localized_text(
    language,
    "Gerçek Rakip Verisi Kullanılabilir" if connected else "Canlı Rakip Verisi Yok",
    "Live Competitor Data Available" if connected else "No Live Competitor Data",
)

with st.container(border=True):
    st.markdown(f"**{localized_text(language, 'Veri Sağlayıcı', 'Data Provider')}:** DataForSEO")
    st.markdown(
        f"**{localized_text(language, 'Bağlantı Durumu', 'Connection Status')}:** "
        f"{status_icon} {connection_text}"
    )
    st.markdown(
        f"**{localized_text(language, 'Veri Durumu', 'Data Status')}:** {data_text}"
    )

if not connected:
    st.warning(
        localized_text(
            language,
            "DataForSEO production erişimi henüz yapılandırılmadı. Bu nedenle rakip domain, keyword gap, SERP pozisyonu veya görünürlük sonucu gösterilmiyor. Sistem bu verileri uydurmaz.",
            "DataForSEO production access is not configured yet. Therefore competitor domains, keyword gaps, SERP positions and visibility findings are not shown. The system does not fabricate them.",
        )
    )

    st.info(
        localized_text(
            language,
            "Canlı rakip analizi için gerekli sonraki adım: Firma DATAFORSEO_LOGIN ve DATAFORSEO_PASSWORD bilgilerini sağladıktan sonra entegrasyonu scheduler, geçmiş veri saklama ve Decision Engine'e bağlamak.",
            "Next step for live competitor intelligence: once the company provides DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD, connect the integration to the scheduler, historical storage and Decision Engine.",
        )
    )
else:
    st.success(
        localized_text(
            language,
            "DataForSEO credentials algılandı. Canlı entegrasyon katmanı etkinleştirilmeye hazır.",
            "DataForSEO credentials were detected. The live integration layer is ready to be activated.",
        )
    )
