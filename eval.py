"""评测入口：在固定算例上逐算例求解，逐算例一行写入 result/eval_results.csv。

支持三类方法，全部共用同一环境与同一算例：
  * `fshgrl` 与其消融变体（从 checkpoint 加载）
  * 优先调度规则（无需训练）
  * 三个学习基线（从 checkpoint 加载）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from agent.baselines.drl_baselines import BASELINES
from agent.baselines.rules import RULES, select as rule_select
from agent.networks import ActorCritic
from agent.ppo import PPOAgent
from configs.config import ROOT, load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv
from result.logger import append_rows

EVAL_COLUMNS = ["instance_id", "tier", "S", "DDT", "method", "variant", "run_id",
                "eta", "nu", "decision_time_ms", "steps", "a_f_mean", "a_feas_mean",
                "p_singleton", "phi_star_mean", "feasible"]


def _run_episode(env: SchedulingEnv, chooser) -> dict:
    t0 = time.perf_counter()
    while not env.done:
        actions, sol = env.candidate_actions()
        if not actions:
            break
        if env.step(actions[chooser(env, actions, sol)])[1]:
            break
    elapsed = time.perf_counter() - t0
    steps = max(env.step_count, 1)
    return {
        "eta": round(env.eta, 6), "nu": round(env.nu, 6),
        "decision_time_ms": round(1000.0 * elapsed / steps, 4), "steps": env.step_count,
        "a_f_mean": round(float(np.mean(env.stats.n_pruned)), 3) if env.stats.n_pruned else 0.0,
        "a_feas_mean": round(float(np.mean(env.stats.n_feasible)), 3) if env.stats.n_feasible else 0.0,
        "p_singleton": round(env.stats.singleton / max(len(env.stats.n_pruned), 1), 4),
        "phi_star_mean": round(float(np.mean(env.stats.phi_star)), 4) if env.stats.phi_star else "",
        "feasible": 1,
    }


def make_chooser(method: str, cfg, checkpoint: Path | None, rng):
    if method in RULES:
        return lambda env, actions, sol: rule_select(method, env, actions, rng)
    if method in BASELINES:
        net = BASELINES[method](cfg)
        if checkpoint and checkpoint.exists():
            net.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=False)
        net.eval()
        return lambda env, actions, sol: net.act(
            env, actions, env.observation(actions, sol), rng, greedy=True)[0]
    net = ActorCritic(cfg)
    if checkpoint and checkpoint.exists():
        net.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    net.eval()
    agent = PPOAgent(net, cfg, torch.device("cpu"))
    return lambda env, actions, sol: agent.act(
        env.observation(actions, sol), env.problem.n_stage, 0.0, greedy=True)[0]


def evaluate(method: str, tiers: List[str], cfg, checkpoint: Path | None,
             run_id: str, variant: str, n_rollout: int, out: Path) -> Path:
    rng = np.random.default_rng()
    chooser = make_chooser(method, cfg, checkpoint, rng)
    rows = []
    for tier in tiers:
        for meta in read_index(tier):
            inst = load_instance_csv(meta["path"], meta["tier"], meta["instance_id"])
            for k in range(n_rollout):
                env = SchedulingEnv(inst, cfg)
                row = _run_episode(env, chooser)
                row.update({"instance_id": meta["instance_id"], "tier": tier,
                            "S": meta["S"], "DDT": meta["DDT"], "method": method,
                            "variant": variant, "run_id": f"{run_id}_r{k+1}"})
                rows.append(row)
    append_rows(out, rows, EVAL_COLUMNS)
    print(f"[OK] {method}/{variant}/{run_id}: {len(rows)} 行 -> {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--method", default="FSHGRL")
    parser.add_argument("--variant", default="FSHGRL")
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--tiers", nargs="+", default=["main"])
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--out", default="result/eval_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_rollout = args.rollouts or int(cfg.get("runtime.eval_rollouts", 10))
    ckpt = Path(args.ckpt) if args.ckpt else None
    evaluate(args.method, args.tiers, cfg, ckpt, args.run_id, args.variant,
             n_rollout, ROOT / args.out)


if __name__ == "__main__":
    main()
