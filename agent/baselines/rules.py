"""优先调度规则基线。全部共用同一环境与同一奖励，只换动作选择逻辑。

规则按构造都是 **non-delay** 的：只要存在可派工动作就立刻派工。候选集里的 no-op 哨兵
规则一律跳过——这不是给规则设限，而是 non-delay 规则本来就无法表达"保留产能"。唯一的
例外是 SPT-Idle：把同一自由度交给规则的对照，阈值可调（评测时扫一组 θ）。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

RULES = ["MOR", "FIFO", "MWKR", "SPT", "EDD", "Random", "RRC", "SPT-Idle"]

# 带随机性的规则：同一算例上要多次 rollout 取均值；其余规则在固定算例上逐位确定，
# 重复 rollout 只是把同一个数抄几遍（评测、精确解对照与案例研究共用这一份名单）
STOCHASTIC_RULES = ("Random", "RRC")

# SPT-Idle 的等待阈值：所有派工候选的临界比都低于它时选择等待。
# 取值由扫描确定（验证档，三次重复）：1.0 -> 0.5100（等价于不等待）、
# 1.5 -> 0.5225、2.0 -> 0.4850；纯随机等待 p=0.25/0.50 分别为 0.4933/0.4967，
# 均劣于不等待。即"会等待"本身不是免费的增益。
IDLE_THRESHOLD = 1.5


def rank(rule: str, env, actions: List[Tuple[int, int, int]], rng: np.random.Generator) -> List[int]:
    """按规则排序键给派工候选排序，返回下标（最优先在前）。分层 DQN 的下层窗口用它。

    Random 返回随机排列；RRC 随机挑一条规则再排序；no-op 哨兵不参与。
    """
    live = [i for i, a in enumerate(actions) if int(a[0]) >= 0]
    if rule in ("Random",):
        return [live[i] for i in rng.permutation(len(live))]
    if rule == "RRC":
        return rank(str(rng.choice([r for r in RULES if r != "RRC"])), env, actions, rng)
    if rule == "SPT-Idle":
        rule = "SPT"
    keys = [_key(rule, env, actions[i]) for i in live]
    return [live[i] for i in np.argsort(np.asarray(keys, dtype=np.float64), kind="stable")]


def _key(rule: str, env, action) -> float:
    task, machine, order = int(action[0]), int(action[1]), int(action[2])
    if rule == "MOR":
        return -_remaining_ops(env, order)
    if rule == "FIFO":
        return float(env.inst.arrival_times[order])
    if rule == "MWKR":
        return -_remaining_work(env, order)
    if rule == "SPT":
        return float(env.inst.proc_times[task, machine])
    if rule == "EDD":
        return float(env.inst.due_dates[order])
    raise ValueError(f"unknown rule: {rule}")


def _remaining_ops(env, order: int) -> int:
    return env.problem.n_stage - int(env.stage[order])


def _remaining_work(env, order: int) -> float:
    return env.problem.residual_from(int(order), int(env.stage[order]))


def _critical_ratio(env, action) -> float:
    """剩余松弛期 / 剩余路径最小加工时间。<1 表示该订单已不可能按时交付。"""
    _, _, order = int(action[0]), int(action[1]), int(action[2])
    slack = float(env.inst.due_dates[order]) - env.now
    need = max(env.problem.residual_from(order, int(env.stage[order])), 1e-9)
    return slack / need


def select(rule: str, env, actions: List[Tuple[int, int, int]],
           rng: np.random.Generator, idle_threshold: float = IDLE_THRESHOLD) -> int:
    """在候选动作里按规则挑一个，返回下标。`idle_threshold` 只对 SPT-Idle 有效。"""
    if not actions:
        raise ValueError("empty action set")
    # non-delay 规则不表达主动空闲：在派工动作的子集上决策，再映回原下标。
    # 唯一的例外是 SPT-Idle —— 它是"把同一自由度也交给规则"的对照基线：
    # 若本文方法相对规则的增益仅仅来自动作集变大而非来自学到的策略，
    # 这条基线就应当拿到同样的增益。它必须先于下面的过滤处理。
    live = [i for i, a in enumerate(actions) if int(a[0]) >= 0]
    if not live:                                       # 只剩 no-op（理论上不会发生）
        return 0
    if rule == "SPT-Idle":
        noop = [i for i, a in enumerate(actions) if int(a[0]) < 0]
        if noop and live:
            worst = max(_critical_ratio(env, actions[i]) for i in live)
            if worst < idle_threshold:                 # 手上的活都已经救不回来，等下一件
                return noop[0]
        return live[select("SPT", env, [actions[i] for i in live], rng)]
    if len(live) < len(actions):
        sub = [actions[i] for i in live]
        return live[select(rule, env, sub, rng, idle_threshold)]

    if rule == "Random":
        return int(rng.integers(len(actions)))
    if rule == "RRC":                                  # 每个决策点随机挑一条规则
        return select(str(rng.choice([r for r in RULES if r != "RRC"])), env, actions, rng)

    keys = [_key(rule, env, a) for a in actions]
    return int(np.argmin(np.asarray(keys, dtype=np.float64)))
