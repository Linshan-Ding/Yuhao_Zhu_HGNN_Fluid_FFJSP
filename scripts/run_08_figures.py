"""出版级绘图（result/figures/）：非延迟的代价热图、保留产能、门控决策图、承诺评估器校准、
学习曲线、临界差异图。只读 result/ 下的 CSV 与各 run 的 log.csv。
"""
import csv
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import EXCLUDED_RUN_PREFIXES, ROOT, RUN_SPECS, done, run_tag, step

RESULT = ROOT / "result"
FIGDIR = RESULT / "figures"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42})


def read(name):
    path = RESULT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf/.png", flush=True)


def fig_price(cells):
    rhos = sorted({float(r["rho"]) for r in cells})
    ddts = sorted({float(r["DDT"]) for r in cells})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, key, title in zip(axes, ("price_vs_nohold", "price_vs_oracle_rule"),
                              ("CoH minus learned non-delay policy", "CoH minus best rule per instance")):
        grid = np.full((len(rhos), len(ddts)), np.nan)
        for r in cells:
            if r["variant"] == "CoH" and r.get(key):
                grid[rhos.index(float(r["rho"])), ddts.index(float(r["DDT"]))] = float(r[key])
        lim = max(np.nanmax(np.abs(grid)), 1e-3)
        im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(len(rhos)):
            for j in range(len(ddts)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(ddts)), [f"{int(d)}" for d in ddts])
        ax.set_yticks(range(len(rhos)), [f"{r:.1f}" for r in rhos])
        ax.set_xlabel("Due-date allowance DDT")
        if ax is axes[0]:
            ax.set_ylabel("Offered load $\\rho_{sys}$")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="$\\Delta\\eta$")
    save(fig, "F1_price_of_non_delay")


def fig_held(cells):
    rows = [r for r in cells if r["variant"] == "CoH" and r.get("held_share")]
    if not rows:
        return
    rhos = sorted({float(r["rho"]) for r in rows})
    ddts = sorted({float(r["DDT"]) for r in rows})
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    width = 0.8 / len(ddts)
    for j, ddt in enumerate(ddts):
        vals = [next((float(r["held_share"]) for r in rows if float(r["rho"]) == rho and float(r["DDT"]) == ddt), np.nan)
                for rho in rhos]
        ax.bar(np.arange(len(rhos)) + (j - (len(ddts) - 1) / 2) * width, vals, width,
               color=OKABE_ITO[j], label=f"DDT {int(ddt)}")
    ax.set_xticks(range(len(rhos)), [f"{r:.1f}" for r in rhos])
    ax.set_xlabel("Offered load $\\rho_{sys}$")
    ax.set_ylabel("Held capacity share")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F2_held_capacity")


def fig_gate_map(rows):
    if not rows:
        return
    edges = np.linspace(0, 1, 6)
    grid = np.full((5, 5), np.nan)
    for r in rows:
        i = int(round(float(r["p_max_lo"]) * 5))
        j = int(round(float(r["exp_arr_lo"]) * 5))
        grid[j, i] = float(r["held_rate"])
    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto", extent=[0, 1, 0, 1])
    ax.set_xticks(edges, [f"{e:.1f}" for e in edges])
    ax.set_yticks(edges, [f"{e:.1f}" for e in edges])
    ax.set_xlabel("Best commitment probability $\\max_o \\hat p(o\\mid s)$")
    ax.set_ylabel("Expected arrivals during the hold (scaled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hold rate")
    save(fig, "F3_gate_map")


def fig_calibration(rows):
    bins = [r for r in rows if r["bin_lo"] != "all"]
    if not bins:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = [float(r["p_hat_mean"]) for r in bins]
    y = [float(r["outcome_rate"]) for r in bins]
    n = np.asarray([int(r["n"]) for r in bins], float)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.scatter(x, y, s=20 + 200 * n / n.max(), color=OKABE_ITO[0], alpha=0.8)
    ax.set_xlabel("Predicted on-time probability $\\hat p$")
    ax.set_ylabel("Realised on-time rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "F4_calibration")


def fig_learning_curves():
    curves = defaultdict(list)
    for d in sorted(RESULT.glob("*_run*")):
        if not d.is_dir() or d.name.startswith(EXCLUDED_RUN_PREFIXES) or run_tag(d.name) not in RUN_SPECS:
            continue
        log = read(d.relative_to(RESULT) / "log.csv") if (d / "log.csv").exists() else []
        pts = [(int(r["steps"]), float(r["eta_val"])) for r in log if r.get("eta_val")]
        if pts:
            curves[RUN_SPECS[run_tag(d.name)][1]].append(pts)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, (name, runs) in enumerate(sorted(curves.items())):
        xs = np.unique(np.concatenate([[s for s, _ in r] for r in runs]))
        ys = np.asarray([np.interp(xs, [s for s, _ in r], [v for _, v in r]) for r in runs])
        color = OKABE_ITO[k % len(OKABE_ITO)]
        ax.plot(xs, ys.mean(0), color=color, label=name, lw=1.4, ls="-" if k < len(OKABE_ITO) else "--")
        if len(runs) > 1:
            ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Validation $\\eta$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "F5_learning_curves")


def fig_cd(rows):
    if not rows:
        return
    names = [r["method"] for r in rows]
    ranks = np.asarray([float(r["mean_rank"]) for r in rows])
    cd = float(rows[0]["critical_difference"])
    order = np.argsort(ranks)
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(names) + 1.0))
    for y, i in enumerate(order):
        ax.plot([ranks[i]], [y], "o", color=OKABE_ITO[0])
        ax.text(ranks[i] + 0.1, y, f"{names[i]} ({ranks[i]:.2f})", va="center", fontsize=8)
    best = ranks[order[0]]
    ax.axvspan(best, best + cd, color="grey", alpha=0.15, lw=0)
    ax.set_yticks([])
    ax.set_xlabel(f"Mean rank (lower is better); shaded = Nemenyi CD {cd:.2f}")
    ax.set_xlim(0.5, len(names) + 1.5)
    save(fig, "F6_critical_difference")


def main():
    t0 = time.time()
    cells = read("cell_means.csv")
    if not cells:
        raise SystemExit("[FAIL] 缺 result/cell_means.csv，请先运行 run_07")
    step("F1 非延迟的代价热图 / F2 保留产能")
    fig_price(cells)
    fig_held(cells)
    step("F3 门控决策图 / F4 校准")
    fig_gate_map(read("gate_map.csv"))
    fig_calibration(read("calibration.csv"))
    step("F5 学习曲线 / F6 临界差异图")
    fig_learning_curves()
    fig_cd(read("friedman_nemenyi.csv"))
    done(t0, FIGDIR)


if __name__ == "__main__":
    main()
