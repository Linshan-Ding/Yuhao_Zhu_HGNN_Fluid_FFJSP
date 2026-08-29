"""奖励恒等式校验（codegen 技能的硬性验收项）。

稿件 Prop 3(a)：gamma = 1 且 beta_Psi = beta_f = 0 时，
    sum_t r_t == eta - kappa_d * nu
用随机策略 rollout 强制验证。不通过即视为环境建模缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import load_config          # noqa: E402
from data.dataset import read_index             # noqa: E402
from data.generator import load_instance_csv    # noqa: E402
from environment.env import SchedulingEnv       # noqa: E402


def rollout_random(env: SchedulingEnv, rng: np.random.Generator) -> float:
    total = 0.0
    while not env.done:
        actions, _ = env.candidate_actions()
        if not actions:
            break
        reward, done, _ = env.step(actions[rng.integers(len(actions))])
        total += reward
        if done:
            break
    return total


def check(tier: str = "main", n_instances: int = 3, tol: float = 1e-9) -> bool:
    cfg = load_config()
    cfg.set("reward.potential_weight", 0.0)     # 恒等式只在无塑形项时成立
    cfg.set("reward.fluid_align_weight", 0.0)
    rng = np.random.default_rng()
    ok = True
    for row in read_index(tier)[:n_instances]:
        inst = load_instance_csv(row["path"], tier=row["tier"], instance_id=row["instance_id"])
        env = SchedulingEnv(inst, cfg)
        total = rollout_random(env, rng)
        expected = env.eta - env.kappa_d * env.nu
        delta = abs(total - expected)
        flag = "OK " if delta <= tol else "FAIL"
        if delta > tol:
            ok = False
        print(f"[{flag}] {row['instance_id']:<24s} sum_r={total:+.10f} "
              f"eta-kd*nu={expected:+.10f} |delta|={delta:.2e} "
              f"(eta={env.eta:.4f}, nu={env.nu:.4f}, steps={env.step_count})")
    print(("[OK] 奖励恒等式成立" if ok else "[FAIL] 奖励恒等式被破坏 —— 环境建模有误"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
