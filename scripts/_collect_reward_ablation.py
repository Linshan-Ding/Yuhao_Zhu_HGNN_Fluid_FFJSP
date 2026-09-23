"""把 run_09 的 run 汇总成 result/reward_exploration.csv（论文 Table T-NEW-9）。

每一行都在 15 个测试算例（main 档）上贪心评测，与主对比表同一批算例：
  * 新训练的 run：加载 checkpoint_best，按该 run 自己的 config_snapshot.yaml 构造环境；
  * 与主配置相同的行：直接读 result/eval_results.csv 里 fshgrl_run1-5 的逐算例结果，
    逐算例先对 run 取均值。
eta 与 nu 是 15 个算例的均值，置信区间是跨算例的 BCa。训练诊断取自训练日志：达到最终
验证 eta 的 90% 所需的交互步数、重要性比率最大值与其上界 |A_f|/epsilon_k、近似 KL；
复用行对 5 个 run 的步数与 KL 取均值，比率取最大值。
清单外的 rw_* 目录（例如旧版脚本留下的 rw_betaf0p0）不会被读到。
"""
import csv
from collections import defaultdict

import numpy as np
import yaml

from _bootstrap import ROOT, checkpoints

from analysis.stats import bca_ci
from configs.config import Config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from eval import _run_episode, make_chooser
from result.logger import append_rows
from run_09_reward_exploration import EPOCHS, REUSE, ROWS, is_complete, logged_epochs

OUT = ROOT / "result" / "reward_exploration.csv"
COLUMNS = ["panel", "config", "runs", "eta", "eta_ci_lo", "eta_ci_hi", "nu", "steps_to_90pct",
           "ratio_max", "ratio_bound", "approx_kl"]


def _floats(records, key):
    out = []
    for r in records:
        try:
            value = float(r.get(key, ""))
        except ValueError:
            continue
        if np.isfinite(value):
            out.append(value)
    return out


def diagnostics(run_dir):
    with (run_dir / "log.csv").open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    val = [(int(r["steps"]), float(r["eta_val"])) for r in records if r.get("eta_val") not in ("", None)]
    reached = next(steps for steps, eta in val if eta >= 0.9 * val[-1][1]) if val else float("nan")
    ratios, bounds, kls = (_floats(records, k) for k in ("ratio_max", "ratio_bound", "approx_kl"))
    return {"steps": reached, "ratio_max": max(ratios, default=float("nan")),
            "ratio_bound": max(bounds, default=float("nan")),
            "approx_kl": float(np.mean(kls)) if kls else float("nan")}


def evaluate(run_dir, instances):
    cfg = Config(yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")))
    chooser = make_chooser("FSHGRL", cfg, run_dir / "checkpoint_best.pt", np.random.default_rng())
    per = {}
    for iid, inst in instances:
        row = _run_episode(SchedulingEnv(inst, cfg), chooser)
        per[iid] = (float(row["eta"]), float(row["nu"]))
    return per


def reused_from_eval():
    """eval_results.csv 中主方法（variant=FSHGRL，main 档）逐算例对 run 取均值。"""
    path = ROOT / "result" / "eval_results.csv"
    if not path.exists():
        raise SystemExit("[FAIL] 缺 result/eval_results.csv，请先运行 run_05")
    acc = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["variant"] == "FSHGRL" and r["tier"] == "main":
                acc[r["instance_id"]].append((float(r["eta"]), float(r["nu"])))
    return {iid: tuple(np.mean(v, axis=0)) for iid, v in acc.items()}


def main():
    instances = [(m["instance_id"], load_instance_csv(m["path"], m["tier"], m["instance_id"]))
                 for m in read_index("main")]
    reused = reused_from_eval()
    main_runs = [c.parent for c in checkpoints("fshgrl_run*")]
    main_diag = [diagnostics(d) for d in main_runs]

    rows, missing = [], []
    for panel, label, name, _ in ROWS:
        if name == REUSE:
            per, runs = reused, f"{', '.join(d.name for d in main_runs)} (reused)"
            diag = {"steps": float(np.mean([d["steps"] for d in main_diag])),
                    "ratio_max": max(d["ratio_max"] for d in main_diag),
                    "ratio_bound": max(d["ratio_bound"] for d in main_diag),
                    "approx_kl": float(np.mean([d["approx_kl"] for d in main_diag]))}
        else:
            if not is_complete(name):
                missing.append(f"{name}（{logged_epochs(name)}/{EPOCHS} epoch）")
                continue
            run_dir = ROOT / "result" / name
            per, runs, diag = evaluate(run_dir, instances), name, diagnostics(run_dir)
        etas = [per[iid][0] for iid, _ in instances if iid in per]
        nus = [per[iid][1] for iid, _ in instances if iid in per]
        lo, hi = bca_ci(etas, n_boot=10000)
        rows.append({"panel": panel, "config": label, "runs": runs,
                     "eta": round(float(np.mean(etas)), 4), "eta_ci_lo": round(lo, 4),
                     "eta_ci_hi": round(hi, 4), "nu": round(float(np.mean(nus)), 4),
                     "steps_to_90pct": int(round(diag["steps"])) if np.isfinite(diag["steps"]) else "",
                     "ratio_max": round(diag["ratio_max"], 4) if np.isfinite(diag["ratio_max"]) else "",
                     "ratio_bound": round(diag["ratio_bound"], 4) if np.isfinite(diag["ratio_bound"]) else "",
                     "approx_kl": round(diag["approx_kl"], 6) if np.isfinite(diag["approx_kl"]) else ""})
        print(f"  {panel} | {label}: eta={rows[-1]['eta']} [{rows[-1]['eta_ci_lo']}, "
              f"{rows[-1]['eta_ci_hi']}] nu={rows[-1]['nu']} ({len(etas)} 个算例；{runs})", flush=True)

    if OUT.exists():
        OUT.unlink()
    if rows:
        append_rows(OUT, rows, COLUMNS)
    if missing:
        raise SystemExit("[FAIL] 以下 run 未训满预算，表中缺对应行：" + "；".join(missing)
                         + "。先运行 python scripts/run_09_reward_exploration.py")


if __name__ == "__main__":
    main()
