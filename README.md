# SEO & GEO Decision Intelligence AI Agent

A production-oriented **SEO, GEO, Machine Learning and LLM decision-intelligence platform** that combines Google Search Console, GA4, technical SEO signals, forecasting, scenario simulation, recommendation logic, RAG and a bilingual Streamlit interface.

The system is designed to move beyond descriptive SEO reporting and support **forward-looking decisions**: what is likely to happen, where the largest organic-growth opportunities are, which pages deserve action first, and how those decisions can be explained to business stakeholders.

---

## ğŸ“Š Historical SEO Performance Dashboard

A separate interactive dashboard presents the historical organic-growth and e-commerce performance analysis behind the project, including Google Search Console, GA4 and Semrush-based SEO intelligence.

[**View Historical SEO Dashboard**](https://ozlemtonbul.com/dashboards/seo_dashboard.html)

> The historical dashboard presents business-performance evidence, while this repository focuses on the AI/ML decision-intelligence architecture, multi-horizon forecasting, automation, testing and LLM/RAG capabilities.

---

## Highlights

- **Multi-horizon ML forecasting:** 7, 14, 30, 90, 180 and 365 days
- **Operational + strategic forecasting architecture**
- **Leakage-safe chronological backtesting**
- **Pure-ML 90-day impression ensemble**
- **Guarded strategic ML routing**
- **Scenario simulation and opportunity intelligence**
- **Page, keyword, category and content intelligence**
- **Technical SEO and GEO intelligence**
- **LLM + RAG decision support**
- **Turkish / English Streamlit dashboard**
- **PostgreSQL-ready persistence**
- **Docker support**
- **Automated final QA**
- **Sanitized public demo dataset**

---

## Problem

Traditional SEO dashboards mostly explain what already happened.

This project was built to answer more decision-oriented questions:

- Which pages are most likely to gain or lose organic visibility?
- What do 7-day, 30-day, 3-month, 6-month and 1-year trajectories look like?
- Which opportunities deserve attention first?
- How reliable is a forecast at each horizon?
- Can technical, content, GEO and performance signals be evaluated together?
- Can model output be translated into understandable executive recommendations?

---

## System Architecture

```mermaid
flowchart TD
    A[Google Search Console] --> C[Data Integration]
    B[GA4] --> C
    T[Technical / On-page / PageSpeed Signals] --> C

    C --> D[Feature Engineering]
    D --> E[ML Forecasting]

    E --> E1[Recursive Daily ML]
    E --> E2[Strategic Direct ML Candidates]
    E --> E3[Pure-ML CTR Ensemble]

    E1 --> F[7 / 14 / 30 / 90 / 180 / 365]
    E2 --> F
    E3 --> F

    F --> G[Scenario Simulation]
    F --> H[Opportunity Intelligence]

    G --> I[Recommendation Engine]
    H --> I

    I --> J[Technical SEO Intelligence]
    I --> K[Content + GEO Intelligence]
    I --> L[Competitor Intelligence]

    J --> M[LLM / RAG Decision Support]
    K --> M
    L --> M

    M --> N[Streamlit Decision Dashboard]
    N --> O[Turkish / English]
```

---

## ML Forecasting Design

The production forecasting architecture intentionally does **not** treat all horizons as identical.

### Operational horizons

**7 / 14 / 30 days**

These horizons use a recursive daily ML path. A genuine next-calendar-day model generates the next day, then the new state is recursively fed into later steps.

### Strategic horizons

**90 / 180 / 365 days**

Strategic horizons use guarded ML routing.

- Direct strategic models may be evaluated.
- Unsafe tail-scaling is rejected by production guardrails.
- When a direct strategic target is not safe, the route falls back to the recursive ML path.
- 90-day impressions can use a **pure-ML CTR ensemble**.
- 365 days remain ML-based but are explicitly marked as historically unvalidated because available Search Console history is insufficient for a leakage-safe 365-day train + 365-day holdout.

This avoids presenting long-horizon forecasts with more certainty than the data supports.

---

## Backtesting

Backtesting is chronological and designed to avoid target leakage.

### Historical Search Console research dataset

- **499 calendar days**
- **2025-04-05 â†’ 2026-08-16**
- **742,200 page-day rows**
- **8,437 pages**

The available Search Console history was sufficient for operational and strategic 90/180-day research, but not for an unbiased 365-day train + 365-day holdout.

### Operational forecast performance

| Horizon | Click Total Error | Click WAPE | Impression Total Error | Impression WAPE |
|---:|---:|---:|---:|---:|
| 7 days | 1.21% | 5.51% | 0.87% | 3.88% |
| 14 days | 1.89% | 5.59% | 2.14% | 3.87% |
| 30 days | 4.77% | 9.94% | 2.50% | 7.01% |

### Strategic baseline

| Horizon | Click Total Error | Click WAPE | Impression Total Error | Impression WAPE |
|---:|---:|---:|---:|---:|
| 90 days | 13.77% | 25.44% | 68.68% | 68.90% |
| 180 days | 25.28% | 29.64% | 9.59% | 24.82% |

The original 90-day impression path was not accepted as strong enough. A separate pure-ML CTR architecture was then developed.

### Pure-ML 90-day impression architecture

Models evaluated:

- Ridge
- Gradient Boosting
- Random Forest
- Histogram Gradient Boosting

Pre-cutoff out-of-fold selection and bias correction were used before the final holdout.

Final 90-day impression result:

- **Total error: 28.58%**
- **WAPE: 34.14%**
- Production promotion gate: **passed**

More detail is available in [`docs/engineering/ML_BACKTESTING.md`](docs/engineering/ML_BACKTESTING.md).

---

## Production Guardrails

A strategic forecast is not promoted only because a research model produces a numerically attractive target.

The production router checks whether the requested adjustment can be applied without destabilizing the already validated short-term path.

In the final accepted production design:

- 90-day click direct ML target hit the guardrail â†’ `RecursiveDailyML`
- 90-day impressions â†’ `MLOnlyCTREnsembleImpressions`
- 180-day click direct ML target hit the guardrail â†’ `RecursiveDailyML`
- 180-day impressions â†’ `RecursiveDailyML`
- 365-day route â†’ ML hybrid with explicit validation limitation

All six primary forecast horizons remain **ML-based**.

---

## Final Automated QA

Final release validation:

- Python compile: **PASS**
- Pipeline data validation: **PASS**
- Final automated tests: **123 passed**
- ML daily horizon coverage: **1 â†’ 365**
- Portfolio horizons: **7 / 14 / 30 / 90 / 180 / 365**
- Negative ML forecasts: **none**
- Daily page + forecast-date key: **unique**
- CTR consistency validation: **PASS**
- Engagement-rate validation: **PASS**
- Multi-horizon dashboard contract: **PASS**

Final pipeline snapshot:

- Integrated rows: **36,735**
- Scenario rows: **14,187**
- Recommendation rows: **1,578**
- ML daily forecast rows: **688,025**
- Next-day target validation RÂ² range: **0.9693 â†’ 0.9870**

> The RÂ² values above refer to next-day target validation and are not presented as 90/180/365-day forecast accuracy.

See [`docs/engineering/FINAL_QA.md`](docs/engineering/FINAL_QA.md).

---

## Automated Multi-Horizon Smoke Test

A dedicated automated check verifies that:

- exactly six horizon rows exist
- daily ML output covers day 1 through day 365
- each portfolio horizon reconciles with the cumulative daily forecast
- 30 â†’ 90 â†’ 180 â†’ 365 values actually change
- cumulative clicks and impressions grow with horizon length
- all primary routes are marked as ML
- 365 remains explicitly `Unvalidated-HistoryTooShort`
- the dashboard exposes 7/14/30/90/180/365 options
- AI Insights reads multi-horizon ML outputs

Final result:

```text
[PASS] All ML horizon automatic smoke checks passed.
```

---

## Decision Intelligence Modules

### Executive Overview
Business-level KPI summaries, trends and performance signals.

### Page Analysis
Page-level performance diagnostics and decision support.

### SEO Opportunity Optimizer
Opportunity prioritization using performance, forecasting and business logic.

### AI Insights
Multi-horizon ML forecast center with operational and strategic forecast views.

### Ask AI
LLM / RAG-powered analytical interface with deterministic fallback logic.

### Technical SEO
Crawl, technical signals and prioritization.

### Content + GEO Intelligence
Content-to-commerce, keyword/content gaps and generative-engine visibility signals.

### Competitor Intelligence
Competitive performance context and opportunity analysis.

---

## LLM + RAG Layer

The project includes a provider-independent architecture for:

- Anthropic
- OpenAI
- Google Gemini

It also includes:

- usage guard logic
- deterministic fallback
- RAG ingestion
- chunking
- embeddings
- retrieval
- PostgreSQL / pgvector-ready knowledge storage

The public demo does not require production API credentials.

---

## Public Demo Data

The repository includes a separate sanitized demo layer.

Demo release snapshot:

- **114 sanitized CSV files**
- **17,790 demo rows**
- **15 anonymized page identities**
- real client URLs replaced with `demo.example.com`
- search queries anonymized
- products/categories/organization identifiers anonymized
- numerical business values transformed

The demo is intended to demonstrate application behavior and architecture. Demo numbers are **not exact production business results**.

Security details: [`docs/engineering/PUBLIC_DEMO_SECURITY.md`](docs/engineering/PUBLIC_DEMO_SECURITY.md).

---

## Project Structure

```text
.
â”œâ”€â”€ config/
â”œâ”€â”€ dashboard/
â”‚   â”œâ”€â”€ components/
â”‚   â”œâ”€â”€ pages/
â”‚   â””â”€â”€ services/
â”œâ”€â”€ docker/
â”œâ”€â”€ outputs/                  # sanitized public demo outputs
â”œâ”€â”€ scripts/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ extract/
â”‚   â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ llm/
â”‚   â”œâ”€â”€ memory/
â”‚   â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ rag/
â”‚   â”œâ”€â”€ recommendations/
â”‚   â”œâ”€â”€ reporting/
â”‚   â”œâ”€â”€ utils/
â”‚   â””â”€â”€ warehouse/
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ final_qa/
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ main.py
â””â”€â”€ requirements.txt
```

---

## Run Locally

### 1. Clone

```bash
git clone https://github.com/ozlemtonbul/seo-organic-growth-ml-llm-intelligence-ai-agent.git
cd seo-organic-growth-ml-llm-intelligence-ai-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run dashboard

```bash
python -m streamlit run dashboard/app.py
```

---

## Run Tests

Final QA:

```bash
python scripts/run_final_qa.py
```

Multi-horizon automatic smoke check:

```bash
python scripts/validate_ml_horizon_behavior.py
```

Full pytest suite:

```bash
python -m pytest -q
```

---

## Docker

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

---

## Security & Privacy

The public repository intentionally excludes production secrets and confidential source data.

Excluded or sanitized items include:

- `.env`
- API keys and tokens
- service-account JSON
- production database dumps
- private raw datasets
- private historical extracts
- local logs
- local backups
- virtual environments
- caches
- local patch archives

The included `outputs/` directory contains **sanitized demo data**, not the production dataset.

---

## Engineering Principles Demonstrated

This project emphasizes:

- production-oriented ML rather than notebook-only modeling
- chronological validation
- leakage prevention
- explicit model limitations
- guarded production routing
- reproducible automated QA
- multilingual product design
- explainable business-oriented outputs
- privacy-aware public demo engineering
- separation of deterministic logic and LLM commentary

---

## Current Status

| Capability | Status |
|---|---|
| GSC integration | âœ… Implemented |
| GA4 integration | âœ… Implemented |
| Feature engineering | âœ… Implemented |
| ML forecasting | âœ… Implemented |
| 7/14/30-day operational forecasting | âœ… Backtested |
| 90/180-day strategic forecasting | âœ… Backtested |
| 365-day ML forecasting | âš ï¸ Implemented; historical validation pending |
| Pure-ML CTR ensemble | âœ… Implemented |
| Strategic ML guardrails | âœ… Implemented |
| Scenario simulation | âœ… Implemented |
| Recommendation engine | âœ… Implemented |
| Technical SEO intelligence | âœ… Implemented |
| Content + GEO intelligence | âœ… Implemented |
| RAG architecture | âœ… Implemented |
| Multi-LLM architecture | âœ… Implemented |
| Streamlit dashboard | âœ… Implemented |
| Turkish / English UI | âœ… Implemented |
| PostgreSQL support | âœ… Implemented |
| Docker support | âœ… Implemented |
| Sanitized public demo | âœ… Implemented |
| Final automated QA | âœ… 123 passed |

---

## Author

### Ã–zlem Tonbul

**AI & Data Intelligence Â· AI Agents Â· LLMs Â· Machine Learning Â· Growth Analytics Â· SEO Intelligence**

Portfolio: `ozlemtonbul.com`  
GitHub: `github.com/ozlemtonbul`  
LinkedIn: `linkedin.com/in/ozlemtonbul`

---

## License

This repository is provided as a professional engineering and portfolio project.

Production credentials, private datasets and confidential business information are intentionally excluded.


