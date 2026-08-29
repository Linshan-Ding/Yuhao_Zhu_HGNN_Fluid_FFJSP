"""出版级绘图（论文 Fig F-NEW-2 / F-NEW-4 / F-NEW-5 与两张面板图）。

矢量 PDF + 300dpi PNG；色盲友好配色；多次 run 画均值线 + 标准差带；图内不放标题
（标题写在 caption）。绘图只读 result/ 下的 CSV，不重跑实验、不编造统计量。
"""
import csv
import time
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, done, step

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True})
FIGDIR = ROOT / "result" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def read(name):
    path = ROOT / "result" / name
    if not path.exists():
        print(f"[SKIP] 缺少 {name}，跳过依赖它的图", flush=True)
        return None
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIGDIR / (stem + '.pdf')}", flush=True)


t0 = time.time()

step("Fig F-NEW-2：eps_f 敏感性（2x2 面板）")
rows = read("pruning_sensitivity.csv")
if rows:
    eps = [max(float(r["eps_f"]), 1e-7) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    panels = [("prune_ratio", "Pruning ratio (%)"), ("retention_all", "Oracle retention (%)"),
              ("eta", r"Fulfillment rate $\eta$"), ("decision_time_ms", "Decision time (ms)")]
    for ax, (key, label) in zip(axes.ravel(), panels):
        values = [float(r[key]) if r[key] != "" else np.nan for r in rows]
        ax.plot(eps, values, "o-", color=OKABE_ITO[0])
        if key == "retention_all":
            crit = [float(r["retention_crit"]) if r["retention_crit"] != "" else np.nan
                    for r in rows]
            ax.plot(eps, crit, "s--", color=OKABE_ITO[1], label="critical epochs")
            ax.axhline(100, color="grey", lw=0.8, ls=":")
            ax.legend()
        ax.axvline(1e-5, color=OKABE_ITO[2], lw=1.0, ls="-.")
        ax.set_xscale("log")
        ax.set_ylabel(label)
    for ax in axes[1]:
        ax.set_xlabel(r"pruning threshold $\epsilon_f$")
    save(fig, "F-NEW-2_pruning_sensitivity")

step("Fig F-NEW-4：分布偏移矩阵热力图")
rows = read("shift_matrix.csv")
if rows:
    axes_names = sorted({r["shift_axis"] for r in rows})
    fig, axs = plt.subplots(1, len(axes_names), figsize=(5.2 * len(axes_names), 4.4), squeeze=False)
    for ax, axis_name in zip(axs[0], axes_names):
        subset = [r for r in rows if r["shift_axis"] == axis_name]
        conds = sorted({r["train_cond"] for r in subset}, key=float)
        mat = np.full((len(conds), len(conds)), np.nan)
        for r in subset:
            mat[conds.index(r["train_cond"]), conds.index(r["test_cond"])] = \
                float(r["retention"]) if r["retention"] != "" else np.nan
        im = ax.imshow(mat, cmap="RdBu_r", vmin=0.0, vmax=2.0)
        ax.set_xticks(range(len(conds)), conds, rotation=45)
        ax.set_yticks(range(len(conds)), conds)
        ax.set_xlabel("test condition")
        ax.set_ylabel("train condition")
        for i in range(len(conds)):
            for j in range(len(conds)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=r"retention $\eta_{test}/\eta_{matched}$")
    save(fig, "F-NEW-4_distribution_shift")

step("Fig F-NEW-5：Nemenyi 临界差异图")
rows = read("friedman_nemenyi.csv")
if rows:
    rows = sorted(rows, key=lambda r: float(r["mean_rank"]))
    names = [r["method"] for r in rows]
    ranks = [float(r["mean_rank"]) for r in rows]
    cd = float(rows[0]["critical_difference"])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 2.0))
    ax.errorbar(ranks, range(len(names)), xerr=cd / 2, fmt="o", color=OKABE_ITO[0], capsize=3)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean rank (Nemenyi CD = {cd:.3f}, Friedman p = {rows[0]['friedman_p']})")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "F-NEW-5_critical_difference")

step("消融面板图 (b)：逐机制贡献分解")
rows = read("stats_summary.csv")
if rows:
    keep = [r for r in rows if r["comparison"].startswith("FSHGRL vs. FSHGRL-")]
    if keep:
        keep = sorted(keep, key=lambda r: float(r["mean_diff"]))
        labels = [r["comparison"].replace("FSHGRL vs. FSHGRL-", "") for r in keep]
        diffs = [float(r["mean_diff"]) for r in keep]
        errs = [abs(float(r["lmm_ci_hi"]) - float(r["lmm_ci_lo"])) / 2
                if r["lmm_ci_hi"] != "" else 0.0 for r in keep]
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(labels) + 2.0))
        ax.barh(labels, diffs, xerr=errs, color=OKABE_ITO[0], capsize=3)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_xlabel(r"$\eta_{\mathrm{full}} - \eta_{\mathrm{variant}}$")
        save(fig, "ablation_contribution_panel_b")

step("训练曲线面板：主方法 + 基线（等交互预算）")
curves = {}
for run_dir in sorted((ROOT / "result").glob("*_run*")):
    log = run_dir / "log.csv"
    if not log.exists():
        continue
    with log.open(encoding="utf-8") as handle:
        records = [r for r in csv.DictReader(handle) if r.get("eta_val") not in ("", None)]
    if records:
        tag = run_dir.name.rsplit("_run", 1)[0]
        curves.setdefault(tag, []).append(
            (np.asarray([float(r["steps"]) for r in records]),
             np.asarray([float(r["eta_val"]) for r in records])))
if curves:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for color, (tag, series) in zip(OKABE_ITO, sorted(curves.items())):
        length = min(len(y) for _, y in series)
        steps = series[0][0][:length]
        ys = np.vstack([y[:length] for _, y in series])
        ax.plot(steps, ys.mean(0), color=color, label=tag)
        if ys.shape[0] > 1:
            ax.fill_between(steps, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0),
                            color=color, alpha=0.18)
    ax.set_xlabel("environment interaction steps")
    ax.set_ylabel(r"validation $\eta$")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "training_curves_equal_budget")

done(t0, FIGDIR)
