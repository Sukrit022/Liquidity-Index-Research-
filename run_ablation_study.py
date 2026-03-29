"""
Master Runner — Liquidity Index Ablation Study
===============================================
Runs all model families in sequence:
  Step 3: Statistical Baselines (Naive, MA, ETS, ARIMA, SARIMA)
  Step 4: ML Models (Linear, Ridge, SVR, RF, XGB, LGB, ...)
  Step 5: RNN Models (LSTM, GRU, BiLSTM, BiGRU, ...)
  Step 6: Advanced DL (CNN-LSTM, Attention, Transformer, TCN, WaveNet)
  Step 7: Ensemble (Average, Weighted, Stacking)
  Step 8: Final Report

Usage:
  python run_ablation_study.py                    # run all steps
  python run_ablation_study.py --skip-stats       # skip statistical (slow ARIMA)
  python run_ablation_study.py --steps ml dl_rnn  # run specific steps
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, fn) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] {name} after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run liquidity ablation study")
    parser.add_argument("--skip-stats", action="store_true", help="Skip slow statistical models")
    parser.add_argument("--steps", nargs="+", choices=["stats", "ml", "dl_rnn", "dl_adv", "ensemble", "report"],
                        help="Run only specific steps")
    args = parser.parse_args()

    steps_to_run = args.steps or ["stats", "ml", "dl_rnn", "dl_adv", "ensemble", "report"]
    if args.skip_stats and "stats" in steps_to_run:
        steps_to_run.remove("stats")

    results: dict[str, bool] = {}

    if "stats" in steps_to_run:
        from src.models.statistical_models import run_all_statistical_models
        results["stats"] = run_step("Statistical Models (Naive/MA/ETS/ARIMA/SARIMA)", run_all_statistical_models)

    if "ml" in steps_to_run:
        from src.models.ml_models import run_all_ml_models
        results["ml"] = run_step("ML Models (Linear/Ridge/SVR/RF/XGB/LGB)", run_all_ml_models)

    if "dl_rnn" in steps_to_run:
        from src.models.dl_rnn_models import run_all_rnn_models
        results["dl_rnn"] = run_step("RNN Models (LSTM/GRU/BiLSTM/BiGRU)", run_all_rnn_models)

    if "dl_adv" in steps_to_run:
        from src.models.dl_advanced_models import run_all_advanced_dl_models
        results["dl_adv"] = run_step("Advanced DL (CNN-LSTM/Attention/Transformer/TCN/WaveNet)", run_all_advanced_dl_models)

    if "ensemble" in steps_to_run:
        from src.models.ensemble_models import run_all_ensemble_models
        results["ensemble"] = run_step("Ensemble Models", run_all_ensemble_models)

    if "report" in steps_to_run:
        from src.models.final_report import run_final_report
        results["report"] = run_step("Final Report Compilation", run_final_report)

    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    for step, ok in results.items():
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {step}")


if __name__ == "__main__":
    main()
