"""Causal nonpreemptive flexible-flow-shop discrete-event simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from data.generator import Instance
from environment.problem import Problem

NOT_ARRIVED, WAITING, IN_PROCESS, COMPLETED, DISCARDED = 0, 1, 2, 3, 4

# Internal tuple representation of the public Wait action; rules may also select it.
NOOP = (-1, -1, -1)


def is_noop(action) -> bool:
    return int(action[0]) < 0


@dataclass
class StepStats:
    """逐决策点的诊断量。"""
    n_feasible: List[int] = field(default_factory=list)     # 暴露的派工候选数
    n_candidates: List[int] = field(default_factory=list)   # 含 no-op 的候选数
    singleton: int = 0
    noop_offered: int = 0
    noop_used: int = 0
    held_time: float = 0.0        # 有活可干却被主动闲置的机器时间（保留产能）
    t_obs: float = 0.0


class SchedulingEnv:
    def __init__(self, inst: Instance, cfg) -> None:
        self.inst = inst
        self.cfg = cfg
        self.problem = Problem(inst)
        p = self.problem

        # Candidate enumeration order (relevant only to the SPT-mixed tie variant): all keys are required.
        self.exposure = str(cfg.action_space.exposure)
        if self.exposure not in ("edd", "hopeful_first"):
            raise ValueError(f"unknown action_space.exposure: {self.exposure}")
        self.exposure_threshold = float(cfg.action_space.exposure_threshold)
        self.allow_noop = bool(cfg.action_space.allow_noop)
        self.max_steps = int(cfg.episode.max_decision_steps)

        # 因果尺度：DDT 参数（交期政策）与最大工时（工艺数据）都是决策前已知的量
        self.t_ref = max(float(inst.meta.get("DDT", 0.0)) or float(p.residual[:, 0].mean()), 1.0)
        self.p_ref = max(float(inst.proc_times.max()), 1.0)
        self.wait_interval = float(cfg.action_space.wait_interval) or float(np.median(inst.proc_times[inst.proc_times > 0])) / 4.0
        if not np.isfinite(self.wait_interval) or self.wait_interval <= 0:
            raise ValueError("wait_interval must be finite and positive")

        # 静态结构表
        self._elig_matrix = inst.proc_times > 0                                   # [N, M]
        self._task_machines = [np.nonzero(self._elig_matrix[t])[0] for t in range(p.n_task)]
        self.reset()

    # ------------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        p, inst = self.problem, self.inst
        self.now = float(inst.arrival_times.min()) if p.n_order else 0.0
        self.status = np.full(p.n_order, NOT_ARRIVED, dtype=np.int8)
        self.stage = np.zeros(p.n_order, dtype=np.int16)
        self.machine_free_at = np.zeros(p.n_machine, dtype=np.float64)
        self.machine_busy_with = np.full(p.n_machine, -1, dtype=np.int64)
        self.machine_busy_time = np.zeros(p.n_machine, dtype=np.float64)
        self.n_completed = 0
        self.n_discarded = 0
        self.step_count = 0
        self.done = False
        self.truncated = False
        self._events = []
        self.stats = StepStats()
        self.order_outcome = np.full(p.n_order, -1, dtype=np.int8)   # 1 按时 / 0 超期或丢弃 / -1 未定
        self._grouped: Dict[int, np.ndarray] = {}
        self._cand_stamp = -1
        self._advance_to_decision()

    # ------------------------------------------------------------------ 事件推进
    def _activate_arrivals(self) -> None:
        pending = self.status == NOT_ARRIVED
        if pending.any():
            arrived = pending & (self.inst.arrival_times <= self.now + 1e-9)
            if arrived.any():
                self.status[arrived] = WAITING
                for order in np.flatnonzero(arrived):
                    self._event('arrival', order=int(order))

    def _release_machines(self) -> None:
        for m in np.nonzero((self.machine_busy_with >= 0) &
                            (self.machine_free_at <= self.now + 1e-9))[0]:
            order = int(self.machine_busy_with[m])
            self._event('operation_finish', order=order, stage=int(self.stage[order]), machine=int(m))
            self.machine_busy_with[m] = -1
            self.stage[order] += 1
            if self.stage[order] >= self.problem.n_stage:
                self.status[order] = COMPLETED
                if self.now <= self.inst.due_dates[order] + 1e-9:
                    self.n_completed += 1
                    self.order_outcome[order] = 1
                else:                                   # 超期完工按未达成计，并计入丢弃
                    self.n_discarded += 1
                    self.order_outcome[order] = 0
                self._event('order_resolved', order=order, outcome=int(self.order_outcome[order]),
                            reason='on_time' if self.order_outcome[order] else 'late_completion')
            else:
                self.status[order] = WAITING

    def _discard_hopeless(self) -> None:
        waiting = np.nonzero(self.status == WAITING)[0]
        if waiting.size == 0:
            return
        prod = self.inst.order_product[waiting]
        need = self.problem.residual[prod, self.stage[waiting]]
        hopeless = self.now + need > self.inst.due_dates[waiting]
        if hopeless.any():
            lost = waiting[hopeless]
            self.status[lost] = DISCARDED
            self.order_outcome[lost] = 0
            self.n_discarded += int(lost.size)
            for order in lost:
                self._event('order_resolved', order=int(order), stage=int(self.stage[order]), outcome=0,
                            reason='hopeless_remaining_route')

    def _event(self, kind, **values):
        self._events.append(dict(kind=kind, time=float(self.now), **values))

    def take_events(self):
        """Recorder-owned consumption; events are never policy inputs."""
        events = self._events
        self._events = []
        self._last_events = events
        return events

    def recorded_events(self):
        return tuple(getattr(self, '_last_events', ()))

    def _next_event_time(self) -> float | None:
        candidates = []
        pending = self.status == NOT_ARRIVED
        if pending.any():
            candidates.append(float(self.inst.arrival_times[pending].min()))
        busy = self.machine_busy_with >= 0
        if busy.any():
            candidates.append(float(self.machine_free_at[busy].min()))
        return min(candidates) if candidates else None

    def _advance_to_decision(self) -> None:
        """推进到下一个决策时点（有空闲机器且有就绪工序），或 episode 结束。"""
        for _ in range(self.max_steps):
            self._activate_arrivals()
            self._release_machines()
            self._discard_hopeless()
            if self._has_feasible():
                return
            if not (self.status == NOT_ARRIVED).any() and not (self.machine_busy_with >= 0).any():
                self.done = True
                return
            nxt = self._next_event_time()
            if nxt is None or nxt <= self.now:
                self.done = True
                return
            self.now = nxt
        raise RuntimeError("event advancement exceeded its safety limit")

    # ------------------------------------------------------------------ 候选构造
    def _idle_mask(self) -> np.ndarray:
        return (self.machine_busy_with < 0) & (self.machine_free_at <= self.now + 1e-9)

    def _has_feasible(self) -> bool:
        waiting = np.nonzero(self.status == WAITING)[0]
        if waiting.size == 0:
            return False
        idle = self._idle_mask()
        if not idle.any():
            return False
        tasks = self.inst.order_product[waiting] * self.problem.n_stage + self.stage[waiting]
        return bool((self._elig_matrix[np.unique(tasks)] & idle[None, :]).any())

    def _refresh_waiting(self) -> None:
        """等待订单按工序类型分组，组内按暴露顺序排序（可救优先再交期，或只按交期）。"""
        waiting = np.nonzero(self.status == WAITING)[0]
        if waiting.size == 0:
            self._grouped = {}
            return
        prod = self.inst.order_product[waiting]
        stage = self.stage[waiting]
        tasks = prod * self.problem.n_stage + stage
        slack = self.inst.due_dates[waiting] - self.now
        need = np.maximum(self.problem.residual[prod, stage], 1e-9)
        cr = slack / need
        if self.exposure == "hopeful_first":
            order = np.lexsort((slack, (cr < self.exposure_threshold).astype(np.int8), tasks))
        else:
            order = np.lexsort((slack, tasks))
        waiting, tasks = waiting[order], tasks[order]
        starts = np.flatnonzero(np.r_[True, tasks[1:] != tasks[:-1]])
        ends = np.r_[starts[1:], tasks.size]
        self._grouped = {int(tasks[s]): waiting[s:e] for s, e in zip(starts, ends)}

    def _feasible_actions(self) -> List[Tuple[int, int, int]]:
        """All ready orders crossed with their idle eligible machines."""
        self._refresh_waiting()
        if not self._grouped:
            return []
        idle = self._idle_mask()
        if not idle.any():
            return []
        actions: List[Tuple[int, int, int]] = []
        for task, orders in self._grouped.items():
            machines = self._task_machines[task]
            machines = machines[idle[machines]]
            if machines.size == 0:
                continue
            for order in orders:
                for m in machines:
                    actions.append((int(task), int(m), int(order)))
        return actions

    def _noop_available(self) -> bool:
        """A public positive timer makes waiting independent of hidden arrivals."""
        return self.allow_noop and not self.done and self._has_feasible()

    def candidate_actions(self) -> List[Tuple[int, int, int]]:
        """当前决策点的候选动作：派工三元组，末尾可能带一个 no-op。"""
        if self._cand_stamp == self.step_count and hasattr(self, "_cached_actions"):
            return list(self._cached_actions)
        feasible = self._feasible_actions()
        self._cand_stamp = self.step_count
        if not feasible:
            return []
        self.stats.n_feasible.append(len(feasible))
        if len(feasible) == 1:
            self.stats.singleton += 1
        actions = feasible
        if self._noop_available():
            self.stats.noop_offered += 1
            actions = actions + [NOOP]
        self.stats.n_candidates.append(len(actions))
        self._cached_actions = tuple(actions)
        return actions

    # ------------------------------------------------------------------ 动作执行
    def step(self, action: Tuple[int, int, int]) -> Tuple[float, bool, dict]:
        from environment.interfaces import Dispatch, Wait
        if self.done:
            raise ValueError("cannot step a terminated or truncated episode")
        if isinstance(action, Wait):
            action = NOOP
        elif isinstance(action, Dispatch):
            if not 0 <= action.order < self.problem.n_order:
                raise ValueError("order index out of range")
            action = (self.problem.task_of(action.order, int(self.stage[action.order])), action.machine, action.order)
        if is_noop(action):
            if tuple(action) != NOOP or not self._noop_available():
                raise ValueError("wait is not admissible")
            return self._step_noop()
        task, machine, order = int(action[0]), int(action[1]), int(action[2])
        if not (0 <= order < self.problem.n_order and 0 <= machine < self.problem.n_machine and 0 <= task < self.problem.n_task):
            raise ValueError("dispatch index out of range")
        if self.status[order] != WAITING:
            raise ValueError(f"order {order} is not waiting (status={self.status[order]})")
        if task != self.problem.task_of(order, int(self.stage[order])):
            raise ValueError("dispatch must select the next unfinished operation")
        if not self._idle_mask()[machine]:
            raise ValueError("cannot dispatch onto a busy machine")
        if self.inst.proc_times[task, machine] <= 0:
            raise ValueError(f"machine {machine} cannot process task {task}")
        before_c, before_d = self.n_completed, self.n_discarded
        proc = float(self.inst.proc_times[task, machine])
        self.status[order] = IN_PROCESS
        self.machine_busy_with[machine] = order
        self.machine_free_at[machine] = self.now + proc
        self.machine_busy_time[machine] += proc
        self._event('operation_start', order=order, stage=int(self.stage[order]), machine=machine,
                    duration=proc, scheduled_end=float(self.now + proc))
        self.step_count += 1
        self._advance_to_decision()
        self._check_limit()
        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        return d_c / max(self.problem.n_order, 1), self.done, {
            "order": order, "d_completed": d_c, "d_discarded": d_d, "noop": False,
            "terminated": self.done and not self.truncated, "truncated": self.truncated}

    def _step_noop(self) -> Tuple[float, bool, dict]:
        """保留产能：本时刻不派工，推进到下一事件。奖励与派工同一计数式。"""
        before_c, before_d = self.n_completed, self.n_discarded
        self.stats.noop_used += 1
        self.step_count += 1
        machines = {a[1] for a in self._feasible_actions()}
        nxt = self._next_event_time()
        wake = self.now + self.wait_interval
        if nxt is not None and nxt > self.now + 1e-9:
            wake = min(wake, nxt)
        self.stats.held_time += len(machines) * (wake - self.now)
        self._event('wait', end=float(wake), machines=sorted(machines), held_machine_time=len(machines)*(wake-self.now))
        self.now = wake
        self._advance_to_decision()
        self._check_limit()
        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        return d_c / max(self.problem.n_order, 1), self.done, {
            "order": -1, "d_completed": d_c, "d_discarded": d_d, "noop": True,
            "terminated": self.done and not self.truncated, "truncated": self.truncated}

    def _check_limit(self):
        if not self.done and self.step_count >= self.max_steps:
            self.truncated = self.done = True

    @property
    def eta(self) -> float:
        return self.n_completed / max(self.problem.n_order, 1)

    @property
    def nu(self) -> float:
        return self.n_discarded / max(self.problem.n_order, 1)

    @property
    def held_share(self) -> float:
        """有活可干却被主动闲置的机器时间占到当前时刻总机器时间的份额（保留产能）。"""
        return self.stats.held_time / max(self.problem.n_machine * self.now, 1e-9)
