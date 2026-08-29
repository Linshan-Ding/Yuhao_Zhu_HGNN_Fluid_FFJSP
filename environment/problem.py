"""问题定义：目标计值、可行性、剩余路径下界。稿件 §3.1–3.2 的代码对应物。

本模块是"物理规则"的唯一真源：环境、精确求解器与基线都从这里取常量，
避免同一个下界在三处各写一遍而悄悄不一致。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from data.generator import Instance


class Problem:
    """把一个算例包装成可查询的问题对象。"""

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        self.n_task = inst.task_count
        self.n_machine = inst.machine_count
        self.n_order = inst.order_count
        self.n_stage = inst.stage_count
        self.n_product = inst.product_count

        # 合格机器表与处理速率 mu = 1/p
        self.eligible: Dict[int, List[int]] = {}
        self.rates = np.zeros_like(inst.proc_times, dtype=np.float32)
        for t in range(self.n_task):
            machines = np.nonzero(inst.proc_times[t] > 0)[0]
            self.eligible[t] = [int(m) for m in machines]
            for m in machines:
                self.rates[t, m] = 1.0 / float(inst.proc_times[t, m])

        # 每工序类型的最快加工时间，用于剩余路径下界
        self.min_proc = np.full(self.n_task, np.inf, dtype=np.float64)
        for t in range(self.n_task):
            if self.eligible[t]:
                self.min_proc[t] = float(inst.proc_times[t, self.eligible[t]].min())

        # residual[r, j] = sum_{j' >= j} min_m p_{r j' m}，稿件 assumption (viii) 与 Eq. (18)
        self.residual = np.zeros((self.n_product, self.n_stage + 1), dtype=np.float64)
        for r in range(self.n_product):
            for j in range(self.n_stage - 1, -1, -1):
                self.residual[r, j] = self.residual[r, j + 1] + self.min_proc[self.task_index(r, j)]

        # 相邻阶段对（用于流平衡约束）
        self.stage_pairs = [
            (self.task_index(r, j - 1), self.task_index(r, j))
            for r in range(self.n_product) for j in range(1, self.n_stage)
        ]

    def task_index(self, product: int, stage: int) -> int:
        return int(product) * self.n_stage + int(stage)

    def task_of(self, order: int, stage: int) -> int:
        return self.task_index(int(self.inst.order_product[order]), stage)

    def residual_from(self, order: int, stage: int) -> float:
        """P_lower：订单 order 从 stage 起（含）剩余路径的最小加工时间和。"""
        if stage >= self.n_stage:
            return 0.0
        return float(self.residual[int(self.inst.order_product[order]), int(stage)])

    def downstream_residual(self, task: int) -> float:
        """delta_hat 中被扣除的下游部分：stage+1 起的剩余最小加工时间。"""
        r, j = divmod(int(task), self.n_stage)
        return float(self.residual[r, j + 1])

    def is_hopeless(self, order: int, stage: int, now: float) -> bool:
        """稿件假设 (viii)：剩余路径已无法在交期内完成 -> 丢弃。"""
        return now + self.residual_from(order, stage) > float(self.inst.due_dates[order])

    def fulfillment_rate(self, completed: int) -> float:
        return completed / max(self.n_order, 1)
