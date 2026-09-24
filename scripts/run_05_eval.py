"""主评测：规则（含 SPT-Idle 阈值网格）+ 全部已训练的 run，在 grid 与 ood 档逐算例贪心评测。

产物：result/eval_results.csv（逐算例一行）、result/gate_trace.csv（CoH 主方法每个决策的门控量）。
重跑会先清空两份产物再全部重算。用法：python scripts/run_05_eval.py [--jobs N]
"""
import argparse
import time
from pathlib import Path

from _bootstrap import ROOT, RUN_SPECS, discover_runs, done, step

from agent.baselines.rules import RULES
from configs.config import load_config
from eval import evaluate

OUT = ROOT / "result" / "eval_results.csv"
TRACE = ROOT / "result" / "gate_trace.csv"
TIERS = ["grid", "ood"]
IDLE_GRID = [1.0, 1.25, 2.0, 3.0]          # 除默认 1.5 之外再扫的阈值


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    t0 = time.time()
    for path in (OUT, TRACE):
        if path.exists():
            path.unlink()
            print(f"[INFO] 已清空旧的 {path.name}", flush=True)
    cfg = load_config(["coh.yaml"])

    step("评测优先调度规则（无需 checkpoint；SPT-Idle 另扫阈值网格）")
    for rule in RULES:
        evaluate(rule, TIERS, cfg, None, "rule", rule, int(cfg.get("runtime.eval_rollouts", 10)),
                 OUT, jobs=args.jobs)
    for theta in IDLE_GRID:
        evaluate("SPT-Idle", TIERS, cfg, None, "rule", f"SPT-Idle({theta})", 1, OUT,
                 jobs=args.jobs, idle_threshold=theta)

    step("评测全部已训练的 run")
    runs = discover_runs()
    if not runs:
        raise SystemExit("[FAIL] result/ 下没有可评测的 run，请先运行 run_02 / run_03 / run_04")
    for tag, dirs in runs.items():
        method, variant, overlays = RUN_SPECS[tag]
        vcfg = load_config(overlays)
        for d in dirs:
            evaluate(method, TIERS, vcfg, d / "checkpoint_best.pt", d.name, variant, 1, OUT,
                     jobs=args.jobs, trace=TRACE if tag == "coh" else None)
    done(t0, OUT, TRACE)


if __name__ == "__main__":
    main()
