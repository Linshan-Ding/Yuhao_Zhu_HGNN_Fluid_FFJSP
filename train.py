"""训练入口：多进程采样 + 批量更新，预算按环境交互步数计。

    python train.py --config coh.yaml --run-name coh_run1 [--seed S] [--workers W] [--total-steps T]
                    [--method coh|rule_ppo|dqn|hdqn] [--episodes-per-epoch E] [--device auto|cpu]
                    [--override key=value ...] [--resume]

每个 epoch：广播权重 -> worker 采 episodes_per_epoch 条 episode -> 批量更新 -> 定期在 val 档贪心验证。
checkpoint_best.pt 按验证 η 刷新，checkpoint_last.pt 每个 epoch 覆盖（含优化器，可 --resume）。
"""
from __future__ import annotations

import argparse
import secrets
import time
from pathlib import Path

import numpy as np
import torch

from agent.dqn import DQNLearner, HierDQNLearner
from agent.ppo import PPOLearner
from agent.workers import METHODS, WorkerPool
from configs.config import ROOT, load_config
from data.dataset import read_index
from result.logger import CsvLogger

LOG_COLUMNS = ["iter", "steps", "episodes", "eta_val", "eta_train", "return", "ep_len", "a_mean",
               "noop_rate", "p_hold", "held_share", "policy_loss", "value_loss", "entropy", "approx_kl",
               "clip_frac", "ratio_max", "update_epochs_done", "commit_bce", "commit_brier", "hold_bce",
               "n_labels", "q_loss", "q_mean", "n_updates", "epsilon", "lr", "collapse",
               "sps_collect", "sps_total", "t_collect", "t_update", "t_val", "elapsed_s"]


def build(method: str, cfg, device: torch.device, total_steps: int):
    if method == "coh":
        from agent.networks import ActorCritic
        return PPOLearner(ActorCritic(cfg), cfg, device, total_steps)
    if method == "rule_ppo":
        from agent.baselines.rule_ppo import RulePolicyNet
        return PPOLearner(RulePolicyNet(cfg), cfg, device, total_steps)
    if method == "dqn":
        from agent.baselines.dqn_nets import TripletQNet
        return DQNLearner(TripletQNet(cfg), cfg, device, total_steps)
    if method == "hdqn":
        from agent.baselines.dqn_nets import HierQNet
        return HierDQNLearner(HierQNet(cfg), cfg, device, total_steps)
    raise ValueError(f"unknown method: {method}")


def _parse_override(items):
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            out[key] = int(value) if value.isdigit() else float(value)
        except ValueError:
            out[key] = {"true": True, "false": False}.get(value.lower(), value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--episodes-per-epoch", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu"], default=None)
    parser.add_argument("--override", nargs="*", default=[], help="key=value，覆盖任意配置项（冒烟用）")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, value in _parse_override(args.override).items():
        cfg.set(key, value)
    if args.total_steps is not None:
        cfg.set("training.total_steps", int(args.total_steps))
    if args.method is not None:
        cfg.set("training.method", args.method)
    if args.episodes_per_epoch is not None:
        cfg.set("rollout.episodes_per_epoch", int(args.episodes_per_epoch))
    if args.workers is not None:
        cfg.set("rollout.workers", int(args.workers))
    if args.device is not None:
        cfg.set("runtime.update_device", args.device)
    seed = args.seed if args.seed is not None else cfg.get("training.seed")
    seed = int(seed) if seed is not None else secrets.randbits(31)
    cfg.set("training.seed", seed)

    method = str(cfg.get("training.method", "coh"))
    total_steps = int(cfg.get("training.total_steps"))
    episodes_per_epoch = int(cfg.get("rollout.episodes_per_epoch", 56))
    val_every = int(cfg.get("rollout.val_every", 2))
    workers = cfg.get("rollout.workers", "auto")
    n_workers = WorkerPool.default_workers() if workers in (None, "auto") else int(workers)
    use_cuda = cfg.get("runtime.update_device", "auto") == "auto" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - n_workers) if device.type == "cpu" else 2)

    run_dir = ROOT / "result" / args.run_name
    resume = args.resume and (run_dir / "checkpoint_last.pt").exists()
    torch.manual_seed(seed)
    learner = build(method, cfg, device, total_steps)
    learner.seed(seed)
    epoch, steps, best, started_offset = 0, 0, -1.0, 0.0
    if resume:
        state = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
        learner.load_state_dict(state["learner"])
        epoch, steps, best = int(state["epoch"]) + 1, int(state["steps"]), float(state["best"])
        started_offset = float(state.get("elapsed_s", 0.0))
        print(f"[{args.run_name}] 从 epoch {epoch}（{steps} 步）续跑", flush=True)
    logger = CsvLogger(run_dir, LOG_COLUMNS, append=resume)
    cfg.snapshot(run_dir / "config_snapshot.yaml")

    val_meta = read_index("val")
    pool = WorkerPool(n_workers, cfg.to_dict(), method, seed, val_meta)
    print(f"[{args.run_name}] method={method} seed={seed} workers={n_workers} device={device} "
          f"budget={total_steps} steps, {episodes_per_epoch} episodes/epoch", flush=True)
    started = time.time() - started_offset
    low_streak = 0
    try:
        while steps < total_steps:
            t0 = time.time()
            pool.broadcast(learner.policy_state_numpy())
            bp = learner.behaviour_params(steps)
            episodes = pool.collect(epoch, episodes_per_epoch, bp)
            n_new = sum(len(e) for e in episodes)
            steps += n_new
            t1 = time.time()
            stats = learner.update(episodes, steps)
            t2 = time.time()
            infos = [e.info for e in episodes]
            eta_train = float(np.mean([i["eta"] for i in infos]))
            row = {"iter": epoch, "steps": steps, "episodes": len(episodes),
                   "eta_train": round(eta_train, 5),
                   "return": round(float(np.mean([e.reward.sum() for e in episodes])), 5),
                   "ep_len": round(n_new / len(episodes), 1),
                   "a_mean": round(float(np.mean([i["n_cand_mean"] for i in infos])), 3),
                   "noop_rate": round(float(np.mean([i["noop_used"] / max(i["noop_offered"], 1) for i in infos])), 4),
                   "held_share": round(float(np.mean([i["held_share"] for i in infos])), 4),
                   "epsilon": round(float(bp.get("epsilon", 0.0)), 4)}
            row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})
            del episodes
            eta_val = ""
            t_val = 0.0
            if epoch % val_every == 0 or steps >= total_steps:
                pool.broadcast(learner.policy_state_numpy())
                val = pool.validate()
                t_val = time.time() - t2
                eta_val = float(np.mean([v["eta"] for v in val]))
                if eta_val > best:
                    best = eta_val
                    torch.save({"model": learner.net.state_dict(), "eta_val": eta_val, "epoch": epoch,
                                "steps": steps, "seed": seed, "method": method}, run_dir / "checkpoint_best.pt")
                low_streak = low_streak + 1 if eta_val < best - 0.15 else 0
            row["collapse"] = int(low_streak >= 2)
            torch.save({"learner": learner.state_dict(), "model": learner.net.state_dict(), "epoch": epoch,
                        "steps": steps, "best": best, "seed": seed, "method": method,
                        "elapsed_s": time.time() - started}, run_dir / "checkpoint_last.pt")
            t_collect, t_update = t1 - t0, t2 - t1
            row.update({"eta_val": round(eta_val, 5) if eta_val != "" else "",
                        "sps_collect": round(n_new / max(t_collect, 1e-9), 1),
                        "sps_total": round(n_new / max(t_collect + t_update + t_val, 1e-9), 1),
                        "t_collect": round(t_collect, 2), "t_update": round(t_update, 2),
                        "t_val": round(t_val, 2), "elapsed_s": round(time.time() - started, 1)})
            logger.log(row)
            print(f"[{args.run_name}] ep {epoch} steps={steps}/{total_steps} eta_train={eta_train:.4f} "
                  f"eta_val={row['eta_val']} held={row['held_share']} sps={row['sps_collect']} "
                  f"(collect {t_collect:.1f}s, update {t_update:.1f}s)", flush=True)
            epoch += 1
    except KeyboardInterrupt:
        print(f"[{args.run_name}] 中断，checkpoint_last.pt 可用 --resume 续跑", flush=True)
    finally:
        pool.close()
    print(f"[DONE] {args.run_name} best eta_val={best:.4f} steps={steps} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
