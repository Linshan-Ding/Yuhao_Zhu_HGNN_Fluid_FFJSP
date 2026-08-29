"""剪枝量化与 oracle 动作保留率（稿件 §5.4 / Table T-NEW-4、Fig F-NEW-2）。

oracle 的定义随算例规模切换，两种都在论文里写明：
  * `exact`   —— 小档：逐个候选动作执行一步，再对残余问题精确求解，取最优者；
  * `lookahead` —— 主档：逐个候选动作执行一步，再用固定参考策略做 N 次公共随机数
                   rollout，取均值最高者，并同时报告 rollout 标准误。
第二种本身是估计量，因此保留率必须与其标准误一起报告，不能只报点值。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from agent.baselines.rules import select as rule_select
from environment.env import SchedulingEnv


@dataclass
class PruningRecord:
    instance_id: str
    a_feas: List[int] = field(default_factory=list)
    a_pruned: List[int] = field(default_factory=list)
    singleton: int = 0
    fallback: int = 0
    epochs: int = 0
    critical_epochs: int = 0
    oracle_hits: int = 0
    oracle_checked: int = 0
    oracle_hits_crit: int = 0
    oracle_checked_crit: int = 0
    oracle_se: List[float] = field(default_factory=list)
    support: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        feas = float(np.mean(self.a_feas)) if self.a_feas else 0.0
        pruned = float(np.mean(self.a_pruned)) if self.a_pruned else 0.0
        return {
            "instance_id": self.instance_id,
            "A_feas_mean": round(feas, 3),
            "A_feas_max": int(max(self.a_feas)) if self.a_feas else 0,
            "A_f_mean": round(pruned, 3),
            "A_f_max": int(max(self.a_pruned)) if self.a_pruned else 0,
            "prune_ratio": round(100.0 * (1.0 - pruned / feas), 2) if feas > 0 else 0.0,
            "p_singleton": round(self.singleton / max(self.epochs, 1), 4),
            "fallback_rate": round(self.fallback / max(self.epochs, 1), 4),
            "retention_all": round(100.0 * self.oracle_hits / max(self.oracle_checked, 1), 2),
            "retention_crit": round(100.0 * self.oracle_hits_crit / max(self.oracle_checked_crit, 1), 2)
            if self.oracle_checked_crit else float("nan"),
            "retention_se": round(float(np.mean(self.oracle_se)), 4) if self.oracle_se else 0.0,
            "support_mean": round(float(np.mean(self.support)), 2) if self.support else 0.0,
            "n_epochs": self.epochs,
        }


def _clone(env: SchedulingEnv) -> SchedulingEnv:
    """深拷贝环境状态用于前瞻；流体缓存共享以免重复求解。"""
    cache, stats = env.fluid._cache, env.fluid.stats
    env.fluid._cache, env.fluid.stats = {}, type(stats)()
    clone = copy.deepcopy(env)
    env.fluid._cache, env.fluid.stats = cache, stats
    clone.fluid._cache, clone.fluid.stats = cache, stats
    return clone


def _reference_rollout(env: SchedulingEnv, rng: np.random.Generator, rule: str = "EDD") -> float:
    while not env.done:
        actions = env._feasible_actions()
        if not actions:
            break
        if env.step(actions[rule_select(rule, env, actions, rng)])[1]:
            break
    return env.eta


def lookahead_oracle(env: SchedulingEnv, actions: Sequence[Tuple[int, int, int]],
                     n_rollout: int, rule: str = "EDD") -> Tuple[Tuple[int, int, int], float]:
    """一步前瞻 oracle：公共随机数下逐候选动作评估，返回 (最优动作, 该动作的 rollout 标准误)。"""
    best_action, best_mean, best_se = actions[0], -np.inf, 0.0
    seeds = np.random.SeedSequence().spawn(n_rollout)      # 公共随机数：所有动作共用同一批
    for action in actions:
        outcomes = []
        for seed in seeds:
            probe = _clone(env)
            if probe.step(action)[1]:
                outcomes.append(probe.eta)
                continue
            outcomes.append(_reference_rollout(probe, np.random.default_rng(seed), rule))
        mean = float(np.mean(outcomes))
        if mean > best_mean:
            best_action, best_mean = action, mean
            best_se = float(np.std(outcomes, ddof=1) / np.sqrt(len(outcomes))) if len(outcomes) > 1 else 0.0
    return best_action, best_se


def analyse_instance(env_factory: Callable[[], SchedulingEnv], instance_id: str,
                     policy: Callable[[SchedulingEnv, list, object], int],
                     oracle_rollouts: int = 8, oracle_every: int = 10,
                     tie_tol: float = 1e-6, max_epochs: int = 100000) -> PruningRecord:
    """跑一条 episode，沿途记录剪枝统计并抽样计算 oracle 保留率。

    `oracle_every` 控制抽样密度：前瞻 oracle 的代价是 |A| x n_rollout 条 rollout，
    逐点计算不现实，故按固定间隔抽样并在论文中写明抽样率。
    """
    env = env_factory()
    rec = PruningRecord(instance_id=instance_id)
    epoch = 0
    while not env.done and epoch < max_epochs:
        feasible = env._feasible_actions()
        if not feasible:
            break
        pruned, sol = env.candidate_actions()
        rec.epochs += 1
        rec.a_feas.append(len(feasible))
        rec.a_pruned.append(len(pruned))
        if len(pruned) == 1:
            rec.singleton += 1
        if sol.status == "optimal":
            rec.support.append(sol.support_size)
        has_critical = any(env._is_critical(a[2], a[0]) for a in feasible)
        if has_critical:
            rec.critical_epochs += 1

        if epoch % oracle_every == 0 and len(feasible) > 1:
            best, se = lookahead_oracle(env, feasible, oracle_rollouts)
            hit = int(best in pruned)
            rec.oracle_checked += 1
            rec.oracle_hits += hit
            rec.oracle_se.append(se)
            if has_critical:
                rec.oracle_checked_crit += 1
                rec.oracle_hits_crit += hit

        idx = policy(env, pruned, sol)
        if env.step(pruned[idx])[1]:
            break
        epoch += 1
    rec.fallback = env.stats.fallback
    return rec
