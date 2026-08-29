"""优先调度规则基线（稿件 §5.6）。全部共用同一环境与同一奖励，只换动作选择逻辑。"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

RULES = ["MOR", "FIFO", "MWKR", "SPT", "EDD", "Random", "RRC"]


def _remaining_ops(env, order: int) -> int:
    return env.problem.n_stage - int(env.stage[order])


def _remaining_work(env, order: int) -> float:
    return env.problem.residual_from(int(order), int(env.stage[order]))


def select(rule: str, env, actions: List[Tuple[int, int, int]],
           rng: np.random.Generator) -> int:
    """在候选动作里按规则挑一个，返回下标。"""
    if not actions:
        raise ValueError("empty action set")
    if rule == "Random":
        return int(rng.integers(len(actions)))
    if rule == "RRC":                                  # 每个决策点随机挑一条规则
        return select(str(rng.choice([r for r in RULES if r != "RRC"])), env, actions, rng)

    keys = []
    for task, machine, order in actions:
        if rule == "MOR":
            keys.append(-_remaining_ops(env, order))
        elif rule == "FIFO":
            keys.append(float(env.inst.arrival_times[order]))
        elif rule == "MWKR":
            keys.append(-_remaining_work(env, order))
        elif rule == "SPT":
            keys.append(float(env.inst.proc_times[task, machine]))
        elif rule == "EDD":
            keys.append(float(env.inst.due_dates[order]))
        else:
            raise ValueError(f"unknown rule: {rule}")
    return int(np.argmin(np.asarray(keys, dtype=np.float64)))
