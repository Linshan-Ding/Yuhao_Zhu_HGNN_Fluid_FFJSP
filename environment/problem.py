"""问题定义：合格机器、剩余路径下界、无望判定。

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
        if inst.proc_times.shape != (inst.task_count, inst.machine_count):
            raise ValueError("processing catalog shape mismatch")
        if not np.isfinite(inst.proc_times).all() or (inst.proc_times < 0).any() or not (inst.proc_times > 0).any(1).all():
            raise ValueError("every operation needs a finite positive eligible processing time")
        if len(inst.machines_per_stage) != inst.stage_count or min(inst.machines_per_stage) < 1:
            raise ValueError("invalid stage machine partition")
        for t in range(inst.task_count):
            lo, hi = inst.stage_machine_slice(t % inst.stage_count)
            if np.any(inst.proc_times[t, :lo] > 0) or np.any(inst.proc_times[t, hi:] > 0):
                raise ValueError("an operation must use machines in its flow-shop stage")
        if not (len(inst.arrival_times) == len(inst.due_dates) == inst.order_count):
            raise ValueError("order array length mismatch")
        if not np.isfinite(inst.arrival_times).all() or not np.isfinite(inst.due_dates).all():
            raise ValueError("arrival times and deadlines must be finite")
        if ((inst.order_product < 0) | (inst.order_product >= inst.product_count)).any():
            raise ValueError("invalid product index")
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


    def task_index(self, product: int, stage: int) -> int:
        return int(product) * self.n_stage + int(stage)

    def task_of(self, order: int, stage: int) -> int:
        return self.task_index(int(self.inst.order_product[order]), stage)

    def residual_from(self, order: int, stage: int) -> float:
        """P_lower：订单 order 从 stage 起（含）剩余路径的最小加工时间和。"""
        if stage >= self.n_stage:
            return 0.0
        return float(self.residual[int(self.inst.order_product[order]), int(stage)])

    def is_hopeless(self, order: int, stage: int, now: float) -> bool:
        """稿件假设 (viii)：剩余路径已无法在交期内完成 -> 丢弃。"""
        return now + self.residual_from(order, stage) > float(self.inst.due_dates[order])
