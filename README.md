## 📊 Live Interactive Dashboard

| Experience | Link |
|---|---|
| Historical SEO Performance Dashboard | [View Dashboard →](https://ozlemtonbul.com/dashboards/seo_dashboard.html) |
| AI / ML / LLM SEO & GEO Decision Intelligence AI Agent Demo | [Launch Public Demo →](https://seo-geo-ml-llm-intelligence.streamlit.app/) |

# SEO & GEO Decision Intelligence AI Agent

> **Enterprise AI-Powered Organic Growth Decision Intelligence Platform**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Google Search Console](https://img.shields.io/badge/Google_Search_Console-Integrated-success)
![GA4](https://img.shields.io/badge/GA4-Integrated-success)
![Machine Learning](https://img.shields.io/badge/ML-RF%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![LLM](https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI%20%7C%20Gemini-blueviolet)
![RAG](https://img.shields.io/badge/RAG-pgvector-informational)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supported-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Pytest](https://img.shields.io/badge/Final_QA-123%20passed-success)

---

# Overview

SEO & GEO Decision Intelligence is a production-oriented AI and Machine Learning platform for organic-growth decision support.

The system combines Google Search Console, Google Analytics 4, technical SEO signals, on-page content analysis, forecasting, opportunity intelligence, recommendation logic, LLMs and RAG in a single modular architecture.

Unlike traditional SEO dashboards that primarily explain historical performance, this platform adds forward-looking decision support through multi-horizon ML forecasting, strategic model guardrails, scenario analysis and AI-assisted executive interpretation.

---

# Project Highlights

- Google Search Console integration
- Google Analytics 4 integration
- Technical SEO and on-page intelligence
- Advanced feature engineering
- Multi-model ML benchmarking
- Random Forest, XGBoost and LightGBM support
- Genuine recursive daily ML forecasting
- **7 / 14 / 30 / 90 / 180 / 365-day ML horizons**
- Operational and strategic forecast separation
- Leakage-safe chronological backtesting
- Pure-ML 90-day CTR ensemble
- Strategic ML guardrails and fallback routing
- Scenario simulation
- Page and keyword opportunity intelligence
- Product / category opportunity intelligence
- Content-to-commerce intelligence
- GEO / AI visibility intelligence
- Recommendation engine
- SHAP explainability support
- Multi-LLM architecture
- Anthropic Claude, OpenAI GPT and Google Gemini support
- RAG ingestion, chunking, embeddings and retrieval
- Deterministic fallback logic
- Turkish / English Streamlit dashboard
- PostgreSQL support
- pgvector-ready knowledge layer
- Docker support
- Sanitized public demo data
- **123 final automated QA tests passing**

---

# Business Problem

SEO teams often work with fragmented reporting from Search Console, GA4, crawlers, spreadsheets and third-party tools. This creates several operational challenges:

- Decisions are reactive instead of predictive.
- High-opportunity pages are difficult to prioritize consistently.
- Short-term and long-term traffic expectations are mixed together.
- Technical, content and performance signals are analyzed separately.
- Long-horizon forecasts can be presented with more confidence than the data supports.
- Manual reporting consumes time that could be used for execution.
- Executive stakeholders often receive metrics without decision context.

---

# Business Value

The platform transforms SEO analysis into a predictive, automated decision-support process.

Key outcomes include:

- Forward-looking organic traffic planning
- Opportunity prioritization
- Page-level decision support
- Technical SEO prioritization
- Content gap identification
- GEO / AI visibility analysis
- Strategic forecasting
- Scenario comparison
- Recommendation generation
- Executive-level AI commentary
- Reproducible QA and validation
- Privacy-aware public demonstration

---

# End-to-End Architecture

```text
                Google Search Console
                         │
                         ▼
                    GSC Extractor
                         │
                Google Analytics 4
                         │
                         ▼
                    GA4 Extractor
                         │
                         ▼
                Data Integration Layer
                         │
                         ▼
                  Feature Engineering
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Lag / Trend       SEO Features     Calendar Features
        └────────────────┼────────────────┘
                         ▼
                ML Training Dataset
                         │
                         ▼
              Multi-Model Benchmarking
        ┌──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼
 Random Forest      XGBoost        LightGBM
        └──────────────┼──────────────┘
                       ▼
              Best Model Selection
                       │
                       ▼
               Recursive Daily ML
                       │
        ┌──────────────┼───────────────────────────┐
        ▼              ▼                           ▼
   7 / 14 / 30      90 / 180                  365 Days
   Operational      Strategic                  Strategic
        │              │                           │
        │        Guarded ML Routing                │
        │       ┌──────┴──────────┐                │
        │       ▼                 ▼                │
        │   Direct ML        Pure-ML CTR           │
        │   Candidate         Ensemble             │
        └───────┴──────────┬──────┴────────────────┘
                           ▼
                  Scenario Simulation
                           │
                           ▼
                 Opportunity Intelligence
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       Technical SEO   Content + GEO   Competitor
             └─────────────┼─────────────┘
                           ▼
                 Recommendation Engine
                           │
                           ▼
                     LLM / RAG Layer
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Anthropic      OpenAI      Gemini
              └────────────┼────────────┘
                           ▼
                 Streamlit Dashboard
                    Turkish / English
```

---

# Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainable AI | SHAP |
| Search Data | Google Search Console API |
| Analytics | Google Analytics Data API |
| AI | Anthropic Claude, OpenAI GPT, Google Gemini |
| RAG | Embeddings, retrieval, pgvector-ready storage |
| Database | PostgreSQL |
| Containerization | Docker |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Testing | Pytest |
| Version Control | Git + GitHub |

---

# Machine Learning Pipeline

The forecasting system uses genuine calendar-day ML forecasting rather than multiplying a one-step prediction by the requested number of days.

The pipeline benchmarks Random Forest, XGBoost and LightGBM and evaluates models using metrics such as MAE, RMSE and R².

---

# Multi-Horizon ML Forecasting

Production forecast horizons:

- **7 days**
- **14 days**
- **30 days**
- **90 days / 3 months**
- **180 days / 6 months**
- **365 days / 1 year**

## Operational Forecasting

7 / 14 / 30-day horizons use a recursive daily ML path. Each predicted day updates the state used by the next forecast step.

## Strategic Forecasting

90 / 180 / 365-day horizons use guarded ML routing. Direct strategic models are not automatically promoted; they must pass production safety checks. If a direct target requires unsafe tail scaling, the route falls back to the stable recursive ML path.

---

# Leakage-Safe Backtesting

Historical Search Console research coverage:

- **499 calendar days**
- **2025-04-05 → 2026-08-16**
- **742,200 rows**
- **8,437 pages**

Backtesting is chronological. Future observations are not used to fit, select or calibrate models that are later evaluated on earlier forecast origins.

## Operational Backtest Results

| Horizon | Click Total Error | Click WAPE | Impression Total Error | Impression WAPE |
|---:|---:|---:|---:|---:|
| 7 days | 1.21% | 5.51% | 0.87% | 3.88% |
| 14 days | 1.89% | 5.59% | 2.14% | 3.87% |
| 30 days | 4.77% | 9.94% | 2.50% | 7.01% |

## Strategic Backtest Results

### 90 Days

- Click total error: **13.77%**
- Click WAPE: **25.44%**
- Original impression total error: **68.68%**
- Original impression WAPE: **68.90%**

The original 90-day impression route was not strong enough for production promotion.

### 180 Days

- Click total error: **25.28%**
- Click WAPE: **29.64%**
- Impression total error: **9.59%**
- Impression WAPE: **24.82%**

---

# Pure-ML 90-Day CTR Ensemble

Models:

- CTR Ridge
- CTR Gradient Boosting
- CTR Random Forest
- CTR Histogram Gradient Boosting

Pre-cutoff out-of-fold selection and bias correction were used before the final holdout evaluation.

Final result:

- **90-day impression total error: 28.58%**
- **90-day impression WAPE: 34.14%**
- Production promotion gate: **PASS**

---

# Strategic ML Guardrails

Accepted production behavior:

- 90-day click direct ML target hit the guardrail → `RecursiveDailyML`
- 90-day impressions → `MLOnlyCTREnsembleImpressions`
- 180-day click direct ML target hit the guardrail → `RecursiveDailyML`
- 180-day impressions → `RecursiveDailyML`
- 365-day route → ML hybrid with explicit validation limitation

All six primary forecast horizons remain ML-based.

The one-year forecast is ML-based but is not presented as historically validated because current history is insufficient for a 365-day training period plus a 365-day holdout.

---

# Explainable AI (SHAP)

The project includes SHAP support for model explainability. Explainability can identify important prediction drivers, support recommendation evidence and improve executive interpretation.

---

# Opportunity Intelligence

The system creates decision-support outputs for:

- page opportunities
- keyword opportunities
- content gaps
- product / category opportunities
- technical SEO issues
- GEO / AI visibility
- blog content-to-commerce opportunities

---

# Recommendation Engine

Recommendations combine ML forecast signals, scenario analysis, opportunity scoring, deterministic business rules, technical/content evidence and optional LLM commentary.

The LLM layer is not required for the deterministic decision engine to operate.

---

# Multi-LLM Architecture

Supported providers:

- Anthropic Claude
- OpenAI GPT
- Google Gemini

A centralized provider manager separates business logic from provider-specific integrations. If live LLM generation is disabled or unavailable, deterministic fallback logic remains available.

---

# RAG Architecture

The project contains document ingestion, chunking, embeddings, vector retrieval, an agent layer and PostgreSQL / pgvector-ready knowledge storage.

---

# Interactive Streamlit Dashboard

Dashboard modules include:

- Ana Panel / Home
- Executive Overview
- Page Analysis
- SEO Opportunity Optimizer
- AI Insights
- Ask AI
- Technical SEO
- Content + GEO Intelligence
- Competitor Intelligence

Dashboard capabilities include Turkish / English UI, date and comparison filters, 7 / 14 / 30 / 90 / 180 / 365 ML horizon selection, operational vs strategic forecast views, KPI cards, forecast charts and recommendation tables.

---

# Historical SEO Performance Dashboard

A separate portfolio dashboard presents historical organic-growth and e-commerce performance analysis.

**Dashboard:** https://ozlemtonbul.com/dashboards/seo_dashboard.html

The historical dashboard provides performance evidence, while this repository focuses on the AI/ML engineering architecture, forecasting, testing and decision-intelligence platform.

---

# Public Demo

The repository includes sanitized demo data for public deployment.

Current demo dataset:

- **114 sanitized CSV files**
- **17,790 rows**
- **15 anonymized page identities**
- Secret/privacy scan: **PASS**

The public demo does not require production GSC or GA4 credentials and does not expose private client URLs or exact production business values.

**Live Streamlit Demo:** [Launch Public Demo →](https://seo-geo-ml-llm-intelligence.streamlit.app/)

---

# Automated Testing

Final automated QA:

- ✅ Python compile PASS
- ✅ Pipeline data validation PASS
- ✅ **123 final QA tests passing**
- ✅ Business-rule tests
- ✅ ML backtesting tests
- ✅ Multi-horizon forecasting tests
- ✅ Strategic ML router tests
- ✅ Pure-ML CTR ensemble tests
- ✅ Localization tests
- ✅ Streamlit UI execution tests
- ✅ Dashboard source-contract tests
- ✅ Historical extraction contract tests

Run final QA:

```bash
python scripts/run_final_qa.py
```

Run multi-horizon smoke test:

```bash
python scripts/validate_ml_horizon_behavior.py
```

Run pytest:

```bash
python -m pytest -q
```

---

# Multi-Horizon Automatic Smoke Check

The dedicated smoke test verifies exactly six forecast horizons, daily forecast coverage from day 1 through day 365, daily-to-portfolio reconciliation, horizon changes, non-negative forecasts, ML-only primary routing and dashboard data contracts.

Final result:

```text
[PASS] All ML horizon automatic smoke checks passed.
```

---

# Output Files

| Output | Description |
|---|---|
| seo_integrated_data.csv | Integrated SEO + analytics dataset |
| seo_daily_performance.csv | Daily SEO performance |
| seo_weekly_performance.csv | Weekly performance summary |
| seo_monthly_performance.csv | Monthly performance summary |
| seo_ml_forecast_daily.csv | Daily ML forecast path |
| seo_ml_forecast_horizons.csv | Page-level horizon forecasts |
| seo_ml_forecast_portfolio.csv | Portfolio forecast by horizon |
| seo_ml_forecast_metrics.csv | Model metrics |
| seo_ml_forecast_feature_importance.csv | Feature importance |
| seo_ml_backtest_summary.csv | Backtest summary |
| seo_scenario_simulation.csv | Scenario simulation |
| seo_page_opportunity_intelligence.csv | Page opportunities |
| seo_keyword_opportunity_intelligence.csv | Keyword opportunities |
| seo_product_category_opportunities.csv | Product/category opportunities |
| seo_technical_seo_intelligence.csv | Technical SEO intelligence |
| seo_geo_ai_visibility_intelligence.csv | GEO / AI visibility intelligence |
| seo_recommendations.csv | Recommendation output |
| seo_shap_summary.csv | SHAP summary |
| seo_rag_ingestion_summary.csv | RAG ingestion summary |

---

# Docker

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

---

# Installation

```bash
git clone https://github.com/ozlemtonbul/seo-organic-growth-ml-llm-intelligence-ai-agent.git
cd seo-organic-growth-ml-llm-intelligence-ai-agent
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

---

# Security & Privacy

The public repository intentionally excludes production secrets and private source data, including `.env`, API keys, tokens, service-account JSON, credentials, production database dumps, private raw data, private historical extracts, local logs, backups, virtual environments, caches and patch archives.

The included `outputs/` folder contains sanitized demo data rather than exact private production data.

---

# Engineering Decision History

Not every experimental model was promoted.

Examples:

- damped strategic ensemble → rejected
- statistical champion approach → research benchmark only
- 90-day lifecycle calibrator → improved baseline but failed absolute quality target
- statistical + ML CTR ensemble → replaced by pure-ML architecture
- unsafe direct strategic click targets → blocked by guardrails
- 365-day validation claim → intentionally withheld due insufficient historical coverage

Detailed evidence is available under `docs/engineering/`.

---

# Current Status

| Capability | Status |
|---|---|
| Google Search Console integration | ✅ Implemented |
| GA4 integration | ✅ Implemented |
| Technical SEO extraction | ✅ Implemented |
| Feature engineering | ✅ Implemented |
| Random Forest | ✅ Implemented |
| XGBoost | ✅ Implemented |
| LightGBM | ✅ Implemented |
| Multi-model benchmarking | ✅ Implemented |
| Multi-horizon ML forecasting | ✅ Implemented |
| 7 / 14 / 30-day forecasting | ✅ Backtested |
| 90 / 180-day strategic forecasting | ✅ Backtested |
| 365-day ML forecast | ⚠️ Implemented; historical validation pending |
| Pure-ML CTR ensemble | ✅ Implemented |
| Strategic ML guardrails | ✅ Implemented |
| Scenario simulation | ✅ Implemented |
| SHAP explainability | ✅ Implemented |
| Recommendation engine | ✅ Implemented |
| Technical SEO intelligence | ✅ Implemented |
| Content + GEO intelligence | ✅ Implemented |
| Competitor intelligence | ✅ Implemented |
| Multi-LLM architecture | ✅ Implemented |
| RAG architecture | ✅ Implemented |
| Streamlit dashboard | ✅ Implemented |
| Turkish / English UI | ✅ Implemented |
| PostgreSQL support | ✅ Implemented |
| Docker support | ✅ Implemented |
| Sanitized public demo data | ✅ Implemented |
| Final QA | ✅ 123 passed |

---

# Future Enhancements

- Scheduled orchestration
- Online model retraining
- Model registry
- Drift monitoring
- REST API deployment
- Cloud-native monitoring
- Kubernetes deployment
- Enterprise authentication
- Role-based access
- Additional long-horizon validation as historical coverage grows

---

# Author

## Özlem Tonbul

**AI & Data Intelligence • AI Agents • LLMs • Machine Learning • Growth Analytics • SEO Intelligence**

🌐 Website: https://ozlemtonbul.com

💻 GitHub: https://github.com/ozlemtonbul

💼 LinkedIn: https://www.linkedin.com/in/ozlemtonbul/

---

# License

This repository is provided as a professional engineering and portfolio project.

Production credentials, private datasets, API keys and confidential business information are intentionally excluded from version control.

© 2026 Özlem Tonbul. All rights reserved.
