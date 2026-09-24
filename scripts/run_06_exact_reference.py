"""精确参照（small 档）：CP-SAT 离线最优、滚动精确重优化、CoH、学习型非延迟策略、最强规则与最强基线。

每个算例一行写入 result/exact_results.csv，可断点续跑（跳过已有的 instance_id）。
用法：python scripts/run_06_exact_reference.py [--jobs N] [--cpsat-time-limit S] [--online-time-limit S]
"""
import argparse
import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _bootstrap import ROOT, RUN_SPECS, discover_runs, done, step

from agent.baselines.rules import RULES, STOCHASTIC_RULES
from configs.config import Config, load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from environment.problem import Problem
from eval import GreedyChooser, RuleChooser, _run_episode
from exact.cpsat import replay_check, solve_cpsat, solve_online_reoptimization
from result.logger import append_rows

OUT = ROOT / "result" / "exact_results.csv"
COLUMNS = ["instance_id", "S", "DDT", "eta_off", "cpsat_status", "cpsat_time_s", "replay_match",
           "eta_online", "online_solves", "online_all_optimal", "online_time_s", "replay_match_online",
           "eta_coh", "eta_coh_sd", "n_coh_runs", "eta_nohold", "eta_best_rule", "best_rule",
           "eta_best_drl", "best_drl", "gap_coh_off", "gap_online_off"]


def _policy_eta(inst, method, overlays, ckpt):
    cfg = load_config(overlays)
    row = _run_episode(SchedulingEnv(inst, cfg), GreedyChooser(method, cfg, ckpt))
    return float(row["eta"])


def _evaluate_instance(job):
    import torch
    torch.set_num_threads(1)
    meta, opts, runs = job["meta"], job["opts"], job["runs"]
    inst = load_instance_csv(ROOT / meta["path"], meta["tier"], meta["instance_id"])
    problem = Problem(inst)
    cfg = load_config(["coh.yaml"])
    off = solve_cpsat(problem, time_limit_s=opts["cpsat_time_limit"], workers=opts["cpsat_workers"])
    off_check = replay_check(problem, off, cfg)
    online = solve_online_reoptimization(problem, time_limit_per_solve=opts["online_time_limit"], workers=1)
    online_check = replay_check(problem, online, cfg)

    def runs_eta(tag):
        method, _, overlays = RUN_SPECS[tag]
        return [_policy_eta(inst, method, overlays, d / "checkpoint_best.pt") for d in runs.get(tag, [])]

    coh = runs_eta("coh")
    nohold = runs_eta("coh_nohold")
    rule_eta = {}
    for rule in RULES:
        n = opts["rollouts"] if rule in STOCHASTIC_RULES else 1
        rule_eta[rule] = float(np.mean([_run_episode(SchedulingEnv(inst, cfg), RuleChooser(rule, 1.5, k))["eta"]
                                        for k in range(n)]))
    best_rule = max(rule_eta, key=rule_eta.get)
    drl = {RUN_SPECS[t][1]: float(np.mean(runs_eta(t))) for t in ("rulesel", "tdqn", "hdqn") if runs.get(t)}
    best_drl = max(drl, key=drl.get) if drl else ""
    return {"instance_id": meta["instance_id"], "S": meta["S"], "DDT": meta["DDT"],
            "eta_off": round(off.eta, 4), "cpsat_status": off.status, "cpsat_time_s": round(off.seconds, 2),
            "replay_match": int(bool(off_check["match"])),
            "eta_online": round(online.eta, 4), "online_solves": online.n_solves,
            "online_all_optimal": int(bool(online.all_optimal)), "online_time_s": round(online.seconds, 2),
            "replay_match_online": int(bool(online_check["match"])),
            "eta_coh": round(float(np.mean(coh)), 4) if coh else "",
            "eta_coh_sd": round(float(np.std(coh, ddof=1)), 4) if len(coh) > 1 else "",
            "n_coh_runs": len(coh),
            "eta_nohold": round(float(np.mean(nohold)), 4) if nohold else "",
            "eta_best_rule": round(rule_eta[best_rule], 4), "best_rule": best_rule,
            "eta_best_drl": round(drl[best_drl], 4) if best_drl else "", "best_drl": best_drl,
            "gap_coh_off": round(off.eta - float(np.mean(coh)), 4) if coh else "",
            "gap_online_off": round(off.eta - online.eta, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cpsat-time-limit", type=float, default=3600.0)
    parser.add_argument("--cpsat-workers", type=int, default=4)
    parser.add_argument("--online-time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*", default=None)
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config()
    runs = discover_runs()
    if "coh" not in runs:
        raise SystemExit("[FAIL] 没有 coh_run*/checkpoint_best.pt，请先运行 run_02")
    have = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            have = {r["instance_id"] for r in csv.DictReader(handle)}
    metas = [m for m in read_index("small") if m["instance_id"] not in have
             and (not args.instances or m["instance_id"] in args.instances)]
    step(f"small 档精确参照：{len(metas)} 个算例待算（已有 {len(have)}）")
    opts = {"cpsat_time_limit": args.cpsat_time_limit, "cpsat_workers": args.cpsat_workers,
            "online_time_limit": args.online_time_limit, "rollouts": int(cfg.get("runtime.eval_rollouts", 10))}
    jobs = [{"meta": m, "opts": opts, "runs": {k: list(v) for k, v in runs.items()}} for m in metas]
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as pool:
            for row in pool.map(_evaluate_instance, jobs):
                append_rows(OUT, [row], COLUMNS)
                print(f"  {row['instance_id']}: off={row['eta_off']} online={row['eta_online']} "
                      f"coh={row['eta_coh']} rule={row['eta_best_rule']} ({row['best_rule']})", flush=True)
    else:
        for job in jobs:
            row = _evaluate_instance(job)
            append_rows(OUT, [row], COLUMNS)
            print(f"  {row['instance_id']}: off={row['eta_off']} online={row['eta_online']} "
                  f"coh={row['eta_coh']} rule={row['eta_best_rule']} ({row['best_rule']})", flush=True)
    done(t0, OUT)


if __name__ == "__main__":
    main()
