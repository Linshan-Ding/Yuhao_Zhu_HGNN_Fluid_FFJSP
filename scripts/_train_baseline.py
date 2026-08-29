"""基线训练器：DRLG（策略梯度）与两个价值型基线（DQN 族）。

与主方法共用环境、算例分布、奖励 Eq.(46) 与交互预算；唯一不同的是网络与动作空间，
这正是 Table T-NEW-6 适配协议里声明的差异。
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from _bootstrap import ROOT as _ROOT  # noqa: F401  锚定工程根并注入 sys.path

from agent.baselines.drl_baselines import BASELINES
from configs.config import ROOT, load_config
from data.dataset import read_index
from data.generator import load_instance_csv, sample_training_instance
from environment.env import SchedulingEnv
from result.logger import CsvLogger

COLUMNS = ["iter", "steps", "eta_train", "eta_val", "loss", "epsilon", "sps", "elapsed_s"]


def evaluate(net, cfg, instances, rng) -> float:
    scores = []
    for inst in instances:
        env = SchedulingEnv(inst, cfg)
        while not env.done:
            actions, sol = env.candidate_actions()
            if not actions:
                break
            idx, _ = net.act(env, actions, env.observation(actions, sol), rng, greedy=True)
            if env.step(actions[idx])[1]:
                break
        scores.append(env.eta)
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(BASELINES))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config()
    net = BASELINES[args.method](cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.get("ppo.lr")))
    run_dir = ROOT / "result" / args.run_name
    logger = CsvLogger(run_dir, COLUMNS)
    cfg.snapshot(run_dir / "config_snapshot.yaml")

    val = [load_instance_csv(r["path"], r["tier"], r["instance_id"]) for r in read_index("val")]
    rng = np.random.default_rng()
    param_table = cfg.get("param_table")
    anneal = max(int(cfg.get("exploration.anneal_epochs", 1000)), 1)
    best, steps, started = -1.0, 0, time.time()
    replay: deque = deque(maxlen=4096)

    for epoch in range(args.epochs):
        epsilon = max(1.0 * (1.0 - epoch / anneal), 0.05)
        env = SchedulingEnv(sample_training_instance(rng, param_table), cfg)
        t0 = time.time()
        traj = []
        while not env.done:
            actions, sol = env.candidate_actions()
            if not actions:
                break
            obs = env.observation(actions, sol)
            idx, value = net.act(env, actions, obs, rng, epsilon=epsilon)
            reward, done, _ = env.step(actions[idx])
            traj.append((obs, idx, reward, value))
            replay.append((obs, idx, reward))
            steps += 1
            if done:
                break

        # 统一用"回报回归"更新：策略型走 REINFORCE-with-baseline，价值型走 Q 回归。
        returns, running = [], 0.0
        for _, _, reward, _ in reversed(traj):
            running += reward
            returns.append(running)
        returns.reverse()
        loss_total = 0.0
        if traj:
            batch = rng.choice(len(traj), size=min(64, len(traj)), replace=False)
            losses = []
            for i in batch:
                obs, idx, _, _ = traj[i]
                target = torch.tensor(returns[i], dtype=torch.float32)
                from agent.networks import obs_to_tensors
                tobs = obs_to_tensors(obs, torch.device("cpu"))
                if args.method == "DRLG":
                    logits, value = net(tobs)
                    logp = torch.log_softmax(logits, dim=-1)
                    advantage = (target - value).detach()
                    losses.append(-(logp.mean()) * advantage + F.mse_loss(value, target))
                else:
                    qs = net.q_values(tobs)
                    losses.append(F.mse_loss(qs[idx], target))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            loss_total = float(loss.item())

        eta_val = evaluate(net, cfg, val, rng) if (epoch % 10 == 0 or epoch == args.epochs - 1) else ""
        if isinstance(eta_val, float) and eta_val > best:
            best = eta_val
            torch.save({"model": net.state_dict(), "eta_val": eta_val}, run_dir / "checkpoint_best.pt")
        logger.log({"iter": epoch, "steps": steps, "eta_train": round(env.eta, 5),
                    "eta_val": eta_val, "loss": round(loss_total, 6),
                    "epsilon": round(epsilon, 4),
                    "sps": round(len(traj) / max(time.time() - t0, 1e-9), 2),
                    "elapsed_s": round(time.time() - started, 1)})
        print(f"[{args.run_name}] ep {epoch} eta_train={env.eta:.4f} eta_val={eta_val}", flush=True)

    print(f"[DONE] {args.run_name} best eta_val={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
