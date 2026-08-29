"""把 run_09 的各个 run 汇总成 result/reward_exploration.csv（Table T-NEW-9）。"""
import csv

import numpy as np

from _bootstrap import ROOT

from analysis.stats import bca_ci
from result.logger import append_rows

COLUMNS = ["panel", "config", "eta", "eta_ci_lo", "eta_ci_hi", "nu", "steps_to_90pct",
           "ratio_max", "ratio_bound", "approx_kl"]
PANELS = {"rw_betaf": "(a) fluid-alignment weight", "rw_beta_psi": "(b) potential shaping",
          "rw_kappa_d": "(b) discard weight", "rw_uncorrected": "(c) importance ratio",
          "rw_onpolicy": "(c) importance ratio"}

rows = []
for run_dir in sorted((ROOT / "result").glob("rw_*")):
    log = run_dir / "log.csv"
    if not log.exists():
        continue
    with log.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    etas = [float(r["eta_val"]) for r in records if r.get("eta_val") not in ("", None)]
    if not etas:
        continue
    final = etas[-1]
    target = 0.9 * final
    reached = next((int(r["steps"]) for r in records
                    if r.get("eta_val") not in ("", None) and float(r["eta_val"]) >= target), "")
    ratios = [float(r["ratio_max"]) for r in records if r.get("ratio_max") not in ("", None)]
    bounds = [float(r["ratio_bound"]) for r in records
              if r.get("ratio_bound") not in ("", None) and np.isfinite(float(r["ratio_bound"]))]
    kls = [float(r["approx_kl"]) for r in records if r.get("approx_kl") not in ("", None)]
    lo, hi = bca_ci(etas[-10:] if len(etas) >= 10 else etas, n_boot=4000)
    panel = next((v for k, v in PANELS.items() if run_dir.name.startswith(k)), "(unclassified)")
    rows.append({"panel": panel, "config": run_dir.name, "eta": round(final, 5),
                 "eta_ci_lo": round(lo, 5), "eta_ci_hi": round(hi, 5), "nu": "",
                 "steps_to_90pct": reached,
                 "ratio_max": round(max(ratios), 4) if ratios else "",
                 "ratio_bound": round(max(bounds), 4) if bounds else "",
                 "approx_kl": round(float(np.mean(kls)), 6) if kls else ""})
    print(f"  {run_dir.name}: eta={final:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)

if not rows:
    raise SystemExit("[FAIL] 未找到 rw_* run，请先运行 run_09 的训练部分")
append_rows(ROOT / "result" / "reward_exploration.csv", rows, COLUMNS)
