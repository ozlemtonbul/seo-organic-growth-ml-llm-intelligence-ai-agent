
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.strategic_champion_selector import evaluate_strategic_champion

OUTPUT_DIR = PROJECT_ROOT / "outputs"
HISTORY_DIR = PROJECT_ROOT / "data" / "historical"

def latest_history():
    files = sorted(HISTORY_DIR.glob("gsc_page_daily_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("data/historical/gsc_page_daily_*.csv bulunamadı.")
    return files[0]

def cutoff(horizon):
    path = OUTPUT_DIR / f"seo_ml_backtest_strategic_{horizon}d_summary.csv"
    frame = pd.read_csv(path, low_memory=False)
    return pd.to_datetime(frame.iloc[0]["CutoffDate"], errors="raise").normalize()

def main():
    print("="*104)
    print("SEO STRATEGIC CHAMPION SELECTOR - LEAKAGE-SAFE DEVELOPMENT EVALUATION")
    print("="*104)
    print("[INFO] Final 90/180 holdout candidate selection icin kullanilmiyor.")
    print("[INFO] Production forecast degistirilmiyor.")

    hist = pd.read_csv(latest_history(), low_memory=False)
    summaries, selections = [], []

    for horizon in (90,180):
        daily_path = OUTPUT_DIR / f"seo_ml_backtest_strategic_{horizon}d_daily.csv"
        if not daily_path.exists():
            print(f"[SKIP] Missing baseline: {daily_path}")
            continue

        baseline = pd.read_csv(daily_path, low_memory=False)
        c = cutoff(horizon)
        print()
        print(f"[RUN] {horizon}-day | cutoff={c.date()}")

        result = evaluate_strategic_champion(hist, baseline, c, horizon)
        result.summary.to_csv(OUTPUT_DIR / f"seo_ml_strategic_champion_{horizon}d_summary.csv", index=False)
        result.daily.to_csv(OUTPUT_DIR / f"seo_ml_strategic_champion_{horizon}d_daily.csv", index=False)
        result.selection.to_csv(OUTPUT_DIR / f"seo_ml_strategic_champion_{horizon}d_selection.csv", index=False)

        print()
        print("INNER VALIDATION MODEL SELECTION")
        print(result.selection.to_string(index=False))
        print()
        print("FINAL HOLDOUT COMPARISON")
        print(result.summary.to_string(index=False))

        # Acceptance gate: a candidate is promotable only when BOTH total error
        # and WAPE improve versus RecursiveDailyML for the same metric/horizon.
        comparison = result.summary.copy()
        gate_rows = []
        for metric_name in ("clicks", "impressions"):
            subset = comparison.loc[comparison["Metric"].eq(metric_name)]
            baseline_row = subset.loc[subset["Method"].eq("RecursiveDailyML")].iloc[0]
            candidate_row = subset.loc[
                subset["Method"].eq("StrategicChampionPortfolio")
            ].iloc[0]
            accepted = (
                float(candidate_row["TotalErrorPct"]) < float(baseline_row["TotalErrorPct"])
                and float(candidate_row["WAPE"]) < float(baseline_row["WAPE"])
            )
            gate_rows.append(
                {
                    "HorizonDays": horizon,
                    "Metric": metric_name,
                    "BaselineTotalErrorPct": float(baseline_row["TotalErrorPct"]),
                    "CandidateTotalErrorPct": float(candidate_row["TotalErrorPct"]),
                    "BaselineWAPE": float(baseline_row["WAPE"]),
                    "CandidateWAPE": float(candidate_row["WAPE"]),
                    "PromoteCandidate": bool(accepted),
                }
            )

        gate = pd.DataFrame(gate_rows)
        gate.to_csv(
            OUTPUT_DIR / f"seo_ml_strategic_champion_{horizon}d_gate.csv",
            index=False,
        )
        print()
        print("PRODUCTION PROMOTION GATE")
        print(gate.to_string(index=False))

        summaries.append(result.summary)
        selections.append(result.selection)

    if not summaries:
        print("[FAIL] No strategic baseline outputs found.")
        return 1

    cs = pd.concat(summaries, ignore_index=True)
    sel = pd.concat(selections, ignore_index=True)
    cs.to_csv(OUTPUT_DIR / "seo_ml_strategic_champion_evaluation.csv", index=False)
    sel.to_csv(OUTPUT_DIR / "seo_ml_strategic_champion_selection.csv", index=False)

    print()
    print("="*104)
    print("CHAMPION SELECTOR EVALUATION COMPLETE")
    print("="*104)
    print(cs.to_string(index=False))
    print()
    print("[OUTPUT] outputs/seo_ml_strategic_champion_evaluation.csv")
    print("[OUTPUT] outputs/seo_ml_strategic_champion_selection.csv")
    print("[IMPORTANT] Production remains unchanged until results are accepted.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
