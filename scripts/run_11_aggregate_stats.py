"""统计聚合（论文 Wilcoxon 表、方差分解、Friedman/Nemenyi）。

三层证据一起报告：BCa 置信区间（描述）、配对 Wilcoxon + 混合效应（推断）、
效应量 + Holm/BH 校正（量级与多重比较）。结论从校正后的 p 值与效应量读，
不从未校正的 p 值读。
"""
import csv
import time
from collections import defaultdict

import numpy as np

from _bootstrap import ROOT, done, step

from analysis.stats import (bca_ci, benjamini_hochberg, friedman_nemenyi, holm_bonferroni,
                            mixed_effects_estimate, paired_compare, variance_decomposition)
from result.logger import append_rows

STATS_COLUMNS = ["comparison", "n", "R_plus", "R_minus", "p_raw", "p_holm", "p_bh",
                 "r_rb", "cliff_delta", "A12", "mean_diff", "lmm_est", "lmm_ci_lo", "lmm_ci_hi"]
VAR_COLUMNS = ["source", "var_component", "icc"]
FRIED_COLUMNS = ["method", "mean_rank", "critical_difference", "friedman_stat",
                 "friedman_df", "friedman_p", "n_instances", "n_methods"]
CI_COLUMNS = ["variant", "instance_id", "eta_mean", "ci_lo", "ci_hi", "n_runs"]

t0 = time.time()
path = ROOT / "result" / "eval_results.csv"
if not path.exists():
    raise SystemExit("[FAIL] 缺少 result/eval_results.csv，请先运行 run_05")

with path.open(encoding="utf-8") as handle:
    records = [r for r in csv.DictReader(handle) if r["tier"] == "main"]

# variant -> instance -> [eta ...]
by_variant = defaultdict(lambda: defaultdict(list))
for r in records:
    by_variant[r["variant"]][r["instance_id"]].append(float(r["eta"]))
instances = sorted({r["instance_id"] for r in records})
if "FSHGRL" not in by_variant:
    raise SystemExit("[FAIL] eval_results.csv 中没有 FSHGRL 行")

step("逐算例均值与 95% BCa 置信区间")
ci_rows = []
for variant, per_inst in sorted(by_variant.items()):
    for instance in instances:
        values = per_inst.get(instance, [])
        if not values:
            continue
        lo, hi = bca_ci(values, n_boot=10000) if len(values) > 1 else (values[0], values[0])
        ci_rows.append({"variant": variant, "instance_id": instance,
                        "eta_mean": round(float(np.mean(values)), 5),
                        "ci_lo": round(lo, 5), "ci_hi": round(hi, 5), "n_runs": len(values)})
append_rows(ROOT / "result" / "eval_ci.csv", ci_rows, CI_COLUMNS)


def instance_means(variant):
    return np.asarray([float(np.mean(by_variant[variant][i])) if by_variant[variant].get(i)
                       else np.nan for i in instances])


ours = instance_means("FSHGRL")
comparisons = [v for v in sorted(by_variant) if v != "FSHGRL"]

step("配对 Wilcoxon + 效应量 + 混合效应估计")
results, p_values = [], []
for variant in comparisons:
    other = instance_means(variant)
    mask = np.isfinite(ours) & np.isfinite(other)
    if mask.sum() < 3:
        continue
    res = paired_compare(f"FSHGRL vs. {variant}", ours[mask], other[mask])
    runs_ours = np.asarray([by_variant["FSHGRL"][i] for i in np.asarray(instances)[mask]], dtype=object)
    n_seed = min(len(v) for v in runs_ours)
    mat_ours = np.asarray([v[:n_seed] for v in runs_ours], dtype=float)
    runs_other = [by_variant[variant][i][:n_seed] for i in np.asarray(instances)[mask]]
    if all(len(v) == n_seed for v in runs_other) and n_seed > 0:
        est, lo, hi = mixed_effects_estimate(mat_ours, np.asarray(runs_other, dtype=float))
    else:
        est = lo = hi = float("nan")
    results.append((res, est, lo, hi))
    p_values.append(res.p_raw)

holm = holm_bonferroni(p_values)
bh = benjamini_hochberg(p_values)
rows = []
for (res, est, lo, hi), p_h, p_b in zip(results, holm, bh):
    rows.append({"comparison": res.comparison, "n": res.n, "R_plus": res.r_plus,
                 "R_minus": res.r_minus, "p_raw": f"{res.p_raw:.3e}",
                 "p_holm": f"{p_h:.3e}", "p_bh": f"{p_b:.3e}",
                 "r_rb": round(res.r_rb, 4), "cliff_delta": round(res.cliff_delta, 4),
                 "A12": round(res.a12, 4), "mean_diff": round(res.mean_diff, 5),
                 "lmm_est": round(est, 5) if np.isfinite(est) else "",
                 "lmm_ci_lo": round(lo, 5) if np.isfinite(lo) else "",
                 "lmm_ci_hi": round(hi, 5) if np.isfinite(hi) else ""})
    print(f"  {res.comparison}: p_raw={res.p_raw:.2e} p_holm={p_h:.2e} A12={res.a12:.3f}", flush=True)
append_rows(ROOT / "result" / "stats_summary.csv", rows, STATS_COLUMNS)

step("方差分解（实例 vs 种子）")
runs = [by_variant["FSHGRL"][i] for i in instances if by_variant["FSHGRL"].get(i)]
n_seed = min(len(v) for v in runs)
vd = variance_decomposition(np.asarray([v[:n_seed] for v in runs], dtype=float))
append_rows(ROOT / "result" / "variance_decomposition.csv",
            [{"source": "instance", "var_component": round(vd["var_instance"], 8),
              "icc": round(vd["icc_instance"], 4)},
             {"source": "seed", "var_component": round(vd["var_seed"], 8),
              "icc": round(vd["icc_seed"], 4)},
             {"source": "residual", "var_component": round(vd["var_residual"], 8), "icc": ""}],
            VAR_COLUMNS)
print(f"  ICC(instance)={vd['icc_instance']:.4f}  ICC(seed)={vd['icc_seed']:.4f}", flush=True)

step("Friedman 全局检验 + Nemenyi 临界差异")
names = ["FSHGRL"] + comparisons
matrix = np.column_stack([instance_means(v) for v in names])
mask = np.isfinite(matrix).all(axis=1)
fn = friedman_nemenyi(matrix[mask], names)
append_rows(ROOT / "result" / "friedman_nemenyi.csv",
            [{"method": m, "mean_rank": round(v, 4),
              "critical_difference": round(fn["critical_difference"], 4),
              "friedman_stat": round(fn["friedman_stat"], 4), "friedman_df": fn["friedman_df"],
              "friedman_p": f"{fn['friedman_p']:.3e}", "n_instances": fn["n_instances"],
              "n_methods": fn["n_methods"]} for m, v in fn["mean_ranks"].items()],
            FRIED_COLUMNS)
print(f"  Friedman p={fn['friedman_p']:.2e}  CD={fn['critical_difference']:.3f}", flush=True)

done(t0, ROOT / "result" / "stats_summary.csv", ROOT / "result" / "eval_ci.csv",
     ROOT / "result" / "variance_decomposition.csv", ROOT / "result" / "friedman_nemenyi.csv")
