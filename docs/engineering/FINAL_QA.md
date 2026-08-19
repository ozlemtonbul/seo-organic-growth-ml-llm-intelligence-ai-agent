# Final QA

## Final release result

```text
PYTHON COMPILE: PASS
PIPELINE DATA VALIDATION: PASS
FINAL PYTEST: PASS
123 passed
```

## Pipeline validation snapshot

| Check | Result |
|---|---|
| Integrated rows | 36,735 |
| Integrated date range | 2026-06-18 → 2026-08-16 |
| Date + page key unique | PASS |
| Negative clicks/impressions/sessions/conversions/revenue | None |
| CTR formula validation | PASS |
| Impression-weighted average position | 4.4160 |
| Engagement rate bounds | PASS |
| Scenario rows | 14,187 |
| Scenario page + scenario key | Unique |
| Recommendation rows | 1,578 |
| Recommendation row per page | PASS |
| ML forecast daily rows | 688,025 |
| ML daily page + date key | Unique |
| ML daily horizon | 1 → 365 |
| ML negative forecasts | None |
| Page-level horizons | 7/14/30/90/180/365 |
| Portfolio horizons | 7/14/30/90/180/365 |

## Test scope

The final automated suite includes coverage for:

- data and business rules
- localization
- Streamlit page execution
- ML backtesting
- multi-horizon forecasting
- strategic direct calibration
- strategic guardrails
- pure-ML CTR ensemble
- historical GSC extraction contracts
- forecast application logic
- dashboard-source contracts

The local `outputs/final_qa_report.txt` is intentionally not published because it contains machine-specific paths.
