"""评测入口：规则、CoH 及其消融、学习基线，在固定档上逐算例贪心求解，逐算例一行写入 CSV。

    python eval.py --method coh --variant CoH --run-id coh_run1 --ckpt result/coh_run1/checkpoint_best.pt \\
                   --tiers grid ood --jobs 4 [--trace result/gate_trace.csv]
    python eval.py --method SPT-Idle --idle-threshold 2.0 --variant "SPT-Idle(2.0)" --tiers grid

按算例并行：`--jobs N` 用 spawn 进程池，每个进程各建一份网络（Windows 也可用）。带门控的策略
用 `--trace` 落盘每个决策的门控输入与输出，供决策图与校准图使用。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agent.baselines.rules import IDLE_THRESHOLD, RULES, STOCHASTIC_RULES, select as rule_select
from agent.workers import METHODS, POLICIES
from configs.config import ROOT, Config, load_config
from data.dataset import read_index
from data.generator import load_instance_csv
from environment.env import SchedulingEnv, is_noop
from result.logger import append_rows

EVAL_COLUMNS = ["instance_id", "tier", "S", "DDT", "rho_target", "rho_sys", "method", "variant", "run_id",
                "seed", "eta", "nu", "decision_time_ms", "steps", "n_cand_mean", "noop_rate",
                "noop_offer_rate", "held_share"]
TRACE_COLUMNS = ["instance_id", "variant", "run_id", "t", "now", "held", "p_hold", "p_max", "hold_logit",
                 "gate_logit", "gap", "with_work", "n_dispatch", "exp_arrivals", "backlog", "viable",
                 "max_cr", "min_proc", "eta_t", "open_share", "rate", "idle_share",
                 "p_hat_chosen", "order", "outcome"]


class RuleChooser:
    def __init__(self, rule: str, idle_threshold: float, seed: int) -> None:
        self.rule, self.threshold = rule, float(idle_threshold)
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, actions):
        return rule_select(self.rule, env, actions, self.rng, self.threshold)


class GreedyChooser:
    """学习策略的贪心读出；CoH 类策略暴露最近一次网络输出供 trace 使用。"""

    def __init__(self, method: str, cfg: Config, checkpoint: Optional[Path]) -> None:
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.method = method
        self.policy = POLICIES[method](cfg)
        if checkpoint is None or not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.policy.net.load_state_dict(state)
        self.rng = np.random.default_rng(0)
        self.last = None

    def __call__(self, env, actions):
        obs = env.observation(actions)
        if self.method == "coh":
            from agent.batch import single_batch
            from agent.networks import greedy_actions
            with self.torch.no_grad():
                out = self.policy.net(single_batch(obs))
            self.last = (obs, out)
            return int(greedy_actions(out)[0].item())
        return self.policy.act(obs, actions, env, self.rng, {}, True)[0]


def _run_episode(env: SchedulingEnv, chooser, trace: Optional[list] = None, tag: dict | None = None) -> dict:
    t0 = time.perf_counter()
    pending = []                                       # (trace 行, 订单) 等 episode 结束填结果
    while not env.done:
        actions = env.candidate_actions()
        if not actions:
            break
        t = env.step_count
        now = env.now
        idx = chooser(env, actions)
        if trace is not None and getattr(chooser, "last", None) is not None:
            obs, out = chooser.last
            noop = is_noop(actions[idx])
            row = {"t": t, "now": round(now, 2), "held": int(noop),
                   "p_hold": round(float(1.0 / (1.0 + np.exp(-float(out.gate_logit[0])))), 4)
                   if out.gate_logit is not None and bool(out.has_noop[0]) else "",
                   "p_max": round(float(1.0 / (1.0 + np.exp(-float(out.commit_logit[0][out.dispatch[0]].max())))), 4)
                   if out.commit_logit is not None else "",
                   "hold_logit": round(float(out.hold_logit[0]), 4) if out.hold_logit is not None else "",
                   "gate_logit": round(float(out.gate_logit[0]), 4) if out.gate_logit is not None else ""}
            g, gl = obs["gate_feat"], obs["global_feat"]
            row.update({"gap": g[0], "with_work": g[1], "n_dispatch": g[2], "exp_arrivals": g[3],
                        "backlog": g[4], "viable": g[5], "max_cr": g[6], "min_proc": g[7],
                        "eta_t": gl[0], "open_share": gl[2], "rate": gl[3], "idle_share": gl[4]})
            row = {k: (round(float(v), 4) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()}
            row.update(tag or {})
            if not noop:
                row["order"] = int(actions[idx][2])
                row["p_hat_chosen"] = round(float(1.0 / (1.0 + np.exp(-float(out.commit_logit[0, idx])))), 4) \
                    if out.commit_logit is not None else ""
                pending.append((row, int(actions[idx][2])))
            else:
                row["order"], row["p_hat_chosen"], row["outcome"] = -1, "", ""
            trace.append(row)
        if env.step(actions[idx])[1]:
            break
    for row, order in pending:
        row["outcome"] = int(env.order_outcome[order])
    elapsed = time.perf_counter() - t0
    steps = max(env.step_count, 1)
    return {
        "eta": round(env.eta, 6), "nu": round(env.nu, 6),
        "decision_time_ms": round(1000.0 * elapsed / steps, 4), "steps": env.step_count,
        "n_cand_mean": round(float(np.mean(env.stats.n_candidates)), 3) if env.stats.n_candidates else 0.0,
        "noop_rate": round(env.stats.noop_used / max(env.stats.noop_offered, 1), 4),
        "noop_offer_rate": round(env.stats.noop_offered / steps, 4),
        "held_share": round(env.held_share, 4),
    }


def _eval_instance(job: dict) -> dict:
    """一个算例上的评测（顶层函数，供进程池调用）。返回 {"rows": [...], "trace": [...]}."""
    import torch
    torch.set_num_threads(1)
    cfg = Config(job["cfg"])
    meta = job["meta"]
    inst = load_instance_csv(ROOT / meta["path"], meta["tier"], meta["instance_id"])
    method = job["method"]
    if method in RULES:
        chooser = RuleChooser(method, job["idle_threshold"], job["seed"])
    else:
        chooser = GreedyChooser(method, cfg, job["ckpt"])
    rows, trace = [], ([] if job.get("trace") else None)
    tag = {"instance_id": meta["instance_id"], "variant": job["variant"], "run_id": job["run_id"]}
    for k in range(int(job["rollouts"])):
        env = SchedulingEnv(inst, cfg)
        row = _run_episode(env, chooser, trace, tag)
        row.update({"instance_id": meta["instance_id"], "tier": meta["tier"], "S": meta["S"],
                    "DDT": meta["DDT"], "rho_target": meta.get("rho_target", ""),
                    "rho_sys": meta.get("rho_sys", ""), "method": method, "variant": job["variant"],
                    "run_id": f"{job['run_id']}_r{k + 1}" if job["rollouts"] > 1 else job["run_id"],
                    "seed": job["seed"]})
        rows.append(row)
    return {"rows": rows, "trace": trace or []}


def evaluate(method: str, tiers: List[str], cfg: Config, checkpoint: Optional[Path], run_id: str,
             variant: str, n_rollout: int, out: Path, jobs: int = 1, seed: int = 0,
             idle_threshold: float = IDLE_THRESHOLD, trace: Optional[Path] = None) -> Path:
    if method not in RULES and method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if method not in STOCHASTIC_RULES and n_rollout > 1:
        n_rollout = 1                                  # 确定性方法重复 rollout 只是抄同一个数
    metas = [m for tier in tiers for m in read_index(tier)]
    job_list = [{"meta": m, "cfg": cfg.to_dict(), "method": method, "variant": variant, "run_id": run_id,
                 "ckpt": str(checkpoint) if checkpoint else None, "rollouts": n_rollout, "seed": seed,
                 "idle_threshold": idle_threshold, "trace": trace is not None} for m in metas]
    if jobs > 1 and len(job_list) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(job_list)),
                                 mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_eval_instance, job_list))
    else:
        results = [_eval_instance(j) for j in job_list]
    rows = [r for res in results for r in res["rows"]]
    append_rows(out, rows, EVAL_COLUMNS)
    if trace is not None:
        append_rows(trace, [r for res in results for r in res["trace"]], TRACE_COLUMNS)
    print(f"[OK] {method}/{variant}/{run_id}: {len(rows)} 行 -> {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--method", required=True, help="规则名或 coh|rule_ppo|dqn|hdqn")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--run-id", default="rule")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--tiers", nargs="+", default=["grid"])
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--out", default="result/eval_results.csv")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle-threshold", type=float, default=IDLE_THRESHOLD)
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_rollout = args.rollouts or int(cfg.get("runtime.eval_rollouts", 10))
    evaluate(args.method, args.tiers, cfg, Path(args.ckpt) if args.ckpt else None, args.run_id,
             args.variant or args.method, n_rollout, ROOT / args.out, args.jobs, args.seed,
             args.idle_threshold, ROOT / args.trace if args.trace else None)


if __name__ == "__main__":
    main()
