# ML Backtesting & Model Selection

## Validation principle

Model selection is performed chronologically. Final holdout observations are not used to fit, select or calibrate models that are later evaluated on the same holdout.

This is particularly important for strategic horizons because the number of independent 90-day and 180-day windows is much smaller than the number of page-day rows.

---

## Historical research coverage

Search Console returned:

- 499 calendar days
- 2025-04-05 → 2026-08-16
- 742,200 rows
- 8,437 pages

A 365-day leakage-safe backtest would require at least 730 days for a 365-day training period plus a 365-day holdout. Therefore the one-year production forecast remains explicitly unvalidated.

---

## Operational results

| Horizon | Click Total Error | Click WAPE | Impression Total Error | Impression WAPE |
|---:|---:|---:|---:|---:|
| 7 | 1.21% | 5.51% | 0.87% | 3.88% |
| 14 | 1.89% | 5.59% | 2.14% | 3.87% |
| 30 | 4.77% | 9.94% | 2.50% | 7.01% |

---

## Strategic baseline

### 90-day baseline

- Click total error: 13.77%
- Click WAPE: 25.44%
- Impression total error: 68.68%
- Impression WAPE: 68.90%

The 90-day impression baseline was considered too weak for promotion.

### 180-day baseline

- Click total error: 25.28%
- Click WAPE: 29.64%
- Impression total error: 9.59%
- Impression WAPE: 24.82%

---

## Experiments

### StrategicDampedEnsembleML

Rejected. It worsened key click performance and did not make 90-day impression quality sufficiently strong.

### Statistical champion selector

Useful as a benchmark, but not accepted as a primary production forecaster because the final product requirement was that all primary forecast routes remain ML-based.

### Lifecycle impression calibrator

Improved the 90-day impression result but remained above the production-quality threshold.

### CTR ensemble with recent-CTR statistical baselines

Improved 90-day impressions, but statistical recent-CTR members were removed from the final primary route.

### Pure-ML CTR ensemble

Accepted.

Members:

- CTR Ridge
- CTR Gradient Boosting
- CTR Random Forest
- CTR Histogram Gradient Boosting

Selection and bias correction were based on pre-cutoff out-of-fold results.

Final 90-day holdout:

- impression total error: 28.58%
- impression WAPE: 34.14%
- predefined promotion gate: PASS

---

## Final production routing

### 90-day clicks

Direct strategic ML candidate required unsafe tail scaling.

Result:

```text
AppliedMethod = RecursiveDailyML
GuardrailHit = True
```

### 90-day impressions

```text
AppliedMethod = MLOnlyCTREnsembleImpressions
PrimaryForecastType = ML
```

### 180-day clicks

Direct strategic ML candidate required unsafe tail scaling.

Result:

```text
AppliedMethod = RecursiveDailyML
GuardrailHit = True
```

### 180-day impressions

```text
AppliedMethod = RecursiveDailyML
ValidationStatus = Backtested-StrategicML
```

### 365 days

```text
PrimaryForecastType = ML
ValidationStatus = Unvalidated-HistoryTooShort
```

The one-year route is not described as statistically validated until sufficient historical data exists.
