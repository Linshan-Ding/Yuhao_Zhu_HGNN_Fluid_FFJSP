"""奖励恒等式校验：Σ_t r_t == η，用随机策略在多档算例与训练采样算例上强制验证。

奖励只计按时完工（r_t = ΔN_c / S），所以恒等式不依赖任何塑形项，也不受到达即无望的
订单在首个决策点之前被丢弃的影响。不通过即视为环境建模缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import load_config                           # noqa: E402
from data.dataset import read_index                              # noqa: E402
from data.generator import load_instance_csv, sample_training_instance  # noqa: E402
from environment.env import SchedulingEnv                        # noqa: E402


def rollout_random(env: SchedulingEnv, rng: np.random.Generator) -> float:
    total = 0.0
    while not env.done:
        actions = env.candidate_actions()
        if not actions:
            break
        reward, done, _ = env.step(actions[rng.integers(len(actions))])
        total += reward
        if done:
            break
    return total


def check(tol: float = 1e-9, n_train: int = 10, seed: int = 0) -> bool:
    cfg = load_config()
    rng = np.random.default_rng(seed)
    cases = []
    for tier, n in (("small", 6), ("grid", 6), ("ood", 2)):
        for row in read_index(tier)[:n]:
            cases.append(load_instance_csv(row["path"], tier=row["tier"], instance_id=row["instance_id"]))
    for _ in range(n_train):
        cases.append(sample_training_instance(rng, cfg.get("param_table")))
    ok = True
    for inst in cases:
        env = SchedulingEnv(inst, cfg)
        total = rollout_random(env, rng)
        delta = abs(total - env.eta)
        if delta > tol:
            ok = False
            print(f"[FAIL] {inst.instance_id:<28s} sum_r={total:+.10f} eta={env.eta:+.10f} "
                  f"|delta|={delta:.2e} steps={env.step_count}")
        if not env.done or (env.order_outcome < 0).any():
            ok = False
            print(f"[FAIL] {inst.instance_id}: episode 未结束或有订单结果未定")
    print(f"[{'OK' if ok else 'FAIL'}] 奖励恒等式 Σr = η 在 {len(cases)} 个算例上"
          f"{'成立' if ok else '被破坏 —— 环境建模有误'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
