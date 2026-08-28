"""冒烟用：统计模块的最小自检。"""
import numpy as np

from _bootstrap import ROOT  # noqa: F401

from analysis.stats import (bca_ci, benjamini_hochberg, friedman_nemenyi,
                            holm_bonferroni, paired_compare, variance_decomposition)

rng = np.random.default_rng()
a, b = rng.normal(0.72, 0.05, 15), rng.normal(0.60, 0.06, 15)
lo, hi = bca_ci(a, n_boot=2000)
res = paired_compare("smoke", a, b)
holm = holm_bonferroni([res.p_raw, 0.04, 0.3])
bh = benjamini_hochberg([res.p_raw, 0.04, 0.3])
vd = variance_decomposition(rng.normal(0.7, 0.05, (15, 5)))
fn = friedman_nemenyi(np.column_stack([a, b, rng.normal(0.5, 0.06, 15)]), ["a", "b", "c"])
print(f"  BCa CI=({lo:.4f}, {hi:.4f})  Wilcoxon p={res.p_raw:.2e}  A12={res.a12:.3f}", flush=True)
print(f"  Holm={[round(v, 4) for v in holm]}  BH={[round(v, 4) for v in bh]}", flush=True)
print(f"  ICC(instance)={vd['icc_instance']:.3f}  Friedman p={fn['friedman_p']:.2e} "
      f"CD={fn['critical_difference']:.3f}", flush=True)
for value in (lo, hi, res.p_raw, fn["critical_difference"]):
    if not np.isfinite(value):
        raise SystemExit("[FAIL] 统计模块输出非有限值")
print("  统计模块自检通过", flush=True)
