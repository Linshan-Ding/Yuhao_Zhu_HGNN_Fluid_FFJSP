"""训练入口：装配 config -> 算例 -> 环境 -> 网络 -> PPO 循环 -> 落盘。

面向复现者的是 scripts/run_XX_*.py 零参数层；本文件是内部/调参接口。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from agent.buffer import RolloutBuffer
from agent.networks import ActorCritic
from agent.ppo import PPOAgent
from configs.config import ROOT, load_config
from data.dataset import read_index
from data.generator import load_instance_csv, sample_training_instance
from environment.env import SchedulingEnv
from result.logger import CsvLogger, VisdomLogger

LOG_COLUMNS = ["iter", "steps", "eta_val", "eta_train", "reward", "policy_loss", "value_loss",
               "entropy", "approx_kl", "clip_frac", "ratio_max", "ratio_bound", "epsilon",
               "sps", "fluid_solve_count", "fluid_cache_hit", "zeta", "phi_star_mean",
               "a_f_mean", "elapsed_s"]


def evaluate(net, cfg, instances, device, n_rollout: int = 1) -> float:
    agent = PPOAgent(net, cfg, device)
    scores = []
    for inst in instances:
        for _ in range(n_rollout):
            env = SchedulingEnv(inst, cfg)
            while not env.done:
                actions, sol = env.candidate_actions()
                if not actions:
                    break
                idx, _, _, _ = agent.act(env.observation(actions, sol),
                                         env.problem.n_stage, 0.0, greedy=True)
                if env.step(actions[idx])[1]:
                    break
            scores.append(env.eta)
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.set("ppo.total_epochs", int(args.epochs))
    total_epochs = int(cfg.get("ppo.total_epochs"))
    rollout_episodes = int(cfg.get("ppo.rollout_episodes", 2))

    device = torch.device("cuda" if (cfg.get("runtime.update_device") == "auto"
                                     and torch.cuda.is_available()) else "cpu")
    net = ActorCritic(cfg)
    agent = PPOAgent(net, cfg, device)

    run_dir = ROOT / "result" / args.run_name
    logger = CsvLogger(run_dir, LOG_COLUMNS)
    cfg.snapshot(run_dir / "config_snapshot.yaml")
    vis = VisdomLogger(env_name=args.run_name)

    val_instances = [load_instance_csv(r["path"], r["tier"], r["instance_id"])
                     for r in read_index("val")]
    rng = np.random.default_rng()
    param_table = cfg.get("param_table")

    best_eta, total_steps, started = -1.0, 0, time.time()
    for epoch in range(total_epochs):
        epsilon = agent.epsilon(epoch)
        buffer = RolloutBuffer()
        etas, rewards, phis, cand, bound = [], [], [], [], 0.0
        t0 = time.time()
        for _ in range(rollout_episodes):
            env = SchedulingEnv(sample_training_instance(rng, param_table), cfg)
            ep_reward = 0.0
            while not env.done:
                actions, sol = env.candidate_actions()
                if not actions:
                    break
                obs = env.observation(actions, sol)
                idx, logp, value, info = agent.act(obs, env.problem.n_stage, epsilon)
                reward, done, _ = env.step(actions[idx])
                buffer.add(obs=obs, action_index=idx, logp_behaviour=logp, reward=reward,
                           value=value, done=done, n_candidates=info["n_candidates"],
                           n_stage=env.problem.n_stage)
                ep_reward += reward
                bound = max(bound, info["ratio_bound"])
                cand.append(info["n_candidates"])
                total_steps += 1
                if done:
                    break
            etas.append(env.eta)
            rewards.append(ep_reward)
            phis.extend(env.stats.phi_star)

        stats = agent.update(buffer)
        elapsed = time.time() - t0
        eta_val = evaluate(net, cfg, val_instances, device) if (epoch % 10 == 0 or
                                                               epoch == total_epochs - 1) else ""
        if isinstance(eta_val, float) and eta_val > best_eta:
            best_eta = eta_val
            torch.save({"model": net.state_dict(), "eta_val": eta_val, "epoch": epoch},
                       run_dir / "checkpoint_best.pt")
        torch.save({"model": net.state_dict(), "epoch": epoch}, run_dir / "checkpoint_last.pt")

        row = {"iter": epoch, "steps": total_steps, "eta_val": eta_val,
               "eta_train": round(float(np.mean(etas)), 5),
               "reward": round(float(np.mean(rewards)), 5),
               "ratio_bound": round(bound, 3), "epsilon": round(epsilon, 5),
               "sps": round(len(buffer) / max(elapsed, 1e-9), 2),
               "phi_star_mean": round(float(np.mean(phis)), 4) if phis else "",
               "a_f_mean": round(float(np.mean(cand)), 3) if cand else "",
               "elapsed_s": round(time.time() - started, 1)}
        row.update({k: round(v, 6) for k, v in stats.items()})
        logger.log(row)
        vis.line("eta_train", epoch, float(np.mean(etas)))
        if isinstance(eta_val, float):
            vis.line("eta_val", epoch, eta_val)
        print(f"[{args.run_name}] ep {epoch}/{total_epochs} eta_train={np.mean(etas):.4f} "
              f"eta_val={eta_val} eps={epsilon:.3f} sps={row['sps']}", flush=True)

    print(f"[DONE] {args.run_name} best eta_val={best_eta:.4f} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
