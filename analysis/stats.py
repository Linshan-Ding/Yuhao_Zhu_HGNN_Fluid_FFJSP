"""统计分析（稿件 §5.9）：自助置信区间、效应量、多重比较校正、方差分解、Friedman/Nemenyi。

只依赖 numpy/scipy —— 混合效应模型用矩估计实现，避免为一列数字引入 statsmodels 依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# ------------------------------------------------------------------ 置信区间
def bca_ci(x: Sequence[float], alpha: float = 0.05, n_boot: int = 10000) -> Tuple[float, float]:
    """偏差校正加速（BCa）自助置信区间。对有界量（如达成率）比正态区间更合适。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng()
    theta = x.mean()
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)

    prop = float((boots < theta).mean())
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = sps.norm.ppf(prop)

    jack = np.asarray([np.delete(x, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = 6.0 * (float(((jbar - jack) ** 2).sum()) ** 1.5)
    a = num / den if den != 0 else 0.0

    def endpoint(z_alpha: float) -> float:
        adj = z0 + (z0 + z_alpha) / max(1.0 - a * (z0 + z_alpha), 1e-12)
        return float(np.clip(sps.norm.cdf(adj), 0.0, 1.0))

    lo = float(np.quantile(boots, endpoint(sps.norm.ppf(alpha / 2))))
    hi = float(np.quantile(boots, endpoint(sps.norm.ppf(1 - alpha / 2))))
    return (lo, hi)


# ------------------------------------------------------------------ 效应量
def rank_biserial(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """配对秩双列相关 r = 1 - 2 R^- / (n(n+1))，同时返回 R^+ 与 R^-。"""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 0.0, 0.0
    ranks = sps.rankdata(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    total = r_plus + r_minus
    r = (r_plus - r_minus) / total if total > 0 else 0.0
    return float(r), r_plus, r_minus


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = float((x[:, None] > y[None, :]).sum())
    lt = float((x[:, None] < y[None, :]).sum())
    return (gt - lt) / max(x.size * y.size, 1)


def vargha_delaney_a12(x: Sequence[float], y: Sequence[float]) -> float:
    """A12：随机取一次 x 胜过随机取一次 y 的概率（并列计 0.5）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = float((x[:, None] > y[None, :]).sum())
    eq = float((x[:, None] == y[None, :]).sum())
    return (gt + 0.5 * eq) / max(x.size * y.size, 1)


# ------------------------------------------------------------------ 多重比较
def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """控制族错误率。返回与输入同序的校正 p 值。"""
    p = np.asarray(p_values, float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted]


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """控制错误发现率。"""
    p = np.asarray(p_values, float)
    m = p.size
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        running = min(running, p[idx] * m / (m - rank))
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted]


# ------------------------------------------------------------------ 配对检验
@dataclass
class PairedResult:
    comparison: str
    n: int
    r_plus: float
    r_minus: float
    p_raw: float
    r_rb: float
    cliff_delta: float
    a12: float
    mean_diff: float


def paired_compare(name: str, ours: Sequence[float], other: Sequence[float]) -> PairedResult:
    ours, other = np.asarray(ours, float), np.asarray(other, float)
    if ours.size != other.size:
        raise ValueError("paired samples must have equal length")
    diff = ours - other
    if np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(sps.wilcoxon(ours, other, zero_method="wilcox").pvalue)
    r_rb, r_plus, r_minus = rank_biserial(ours, other)
    return PairedResult(name, ours.size, r_plus, r_minus, p, r_rb,
                        cliffs_delta(ours, other), vargha_delaney_a12(ours, other),
                        float(diff.mean()))


# ------------------------------------------------------------------ 方差分解
def variance_decomposition(values: np.ndarray) -> Dict[str, float]:
    """两因素随机效应分解：values[instance, seed] -> 实例方差 / 种子方差 / ICC。"""
    v = np.asarray(values, float)
    n_inst, n_seed = v.shape
    grand = v.mean()
    ss_inst = n_seed * float(((v.mean(axis=1) - grand) ** 2).sum())
    ss_seed = n_inst * float(((v.mean(axis=0) - grand) ** 2).sum())
    ss_res = float(((v - v.mean(axis=1, keepdims=True)
                     - v.mean(axis=0, keepdims=True) + grand) ** 2).sum())
    df_inst, df_seed = n_inst - 1, n_seed - 1
    df_res = max(df_inst * df_seed, 1)
    ms_inst, ms_seed, ms_res = ss_inst / max(df_inst, 1), ss_seed / max(df_seed, 1), ss_res / df_res
    var_inst = max((ms_inst - ms_res) / n_seed, 0.0)
    var_seed = max((ms_seed - ms_res) / n_inst, 0.0)
    total = var_inst + var_seed + ms_res
    return {"var_instance": var_inst, "var_seed": var_seed, "var_residual": ms_res,
            "icc_instance": var_inst / total if total > 0 else 0.0,
            "icc_seed": var_seed / total if total > 0 else 0.0}


def mixed_effects_estimate(ours: np.ndarray, other: np.ndarray) -> Tuple[float, float, float]:
    """以实例为随机截距的方法固定效应估计（矩估计）。

    values 形状均为 [instance, seed]。返回 (估计, 95% CI 下界, 上界)。
    实例内配对差消去随机截距，故效应估计即配对差的均值，其标准误按实例间方差算。
    """
    diff = np.asarray(ours, float).mean(axis=1) - np.asarray(other, float).mean(axis=1)
    n = diff.size
    est = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    crit = float(sps.t.ppf(0.975, df=max(n - 1, 1)))
    return est, est - crit * se, est + crit * se


# ------------------------------------------------------------------ 全局检验
def friedman_nemenyi(matrix: np.ndarray, names: Sequence[str],
                     alpha: float = 0.05) -> Dict[str, object]:
    """Friedman 全局检验 + Nemenyi 事后临界差异。matrix[instance, method]。"""
    m = np.asarray(matrix, float)
    n_inst, k = m.shape
    stat, p = sps.friedmanchisquare(*[m[:, j] for j in range(k)])
    ranks = np.asarray([sps.rankdata(-row) for row in m])       # 越大越好 -> 秩 1 最优
    mean_ranks = ranks.mean(axis=0)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164, 12: 3.268, 15: 3.391, 16: 3.426, 20: 3.542}
    q = q_alpha.get(k, 3.5)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_inst))
    return {"friedman_stat": float(stat), "friedman_p": float(p), "friedman_df": k - 1,
            "mean_ranks": {names[j]: float(mean_ranks[j]) for j in range(k)},
            "critical_difference": float(cd), "n_instances": n_inst, "n_methods": k}
