"""优先调度规则基线（稿件 §5.6）。全部共用同一环境与同一奖励，只换动作选择逻辑。

规则按构造都是 **non-delay** 的：只要存在可派工动作就立刻派工。环境若开启主动
空闲（`action_space.allow_noop`），候选集里会带一个 no-op 哨兵，规则一律跳过它
——这不是给规则设限，而是 non-delay 规则本来就无法表达"故意等待"。动作集对
所有方法完全相同，差异只在策略，比较仍然公平。
"""
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
    # non-delay 规则不表达主动空闲：在派工动作的子集上决策，再映回原下标
    live = [i for i, a in enumerate(actions) if int(a[0]) >= 0]
    if not live:                                       # 只剩 no-op（理论上不会发生）
        return 0
    if len(live) < len(actions):
        sub = [actions[i] for i in live]
        return live[select(rule, env, sub, rng)]

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
