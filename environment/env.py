"""高频插单柔性流水车间的离散事件环境（第二篇论文：Commit-or-Hold）。

一个决策时点 = 至少一台机器空闲且至少一道工序就绪。动作是 (task, machine, order) 三元组，
或 no-op（保留产能：本时刻不派工，等到下一事件再决策）。奖励只计按时完工，
    r_t = ΔN_c / S，于是 Σ_t r_t = η（无条件成立，run_00 校验）。

观测严格因果：只用当前时刻已知的信息。时间量除以算例的交期宽松度参数 DDT（车间的交期
政策，已知），工时除以工艺数据里的最大工时，计数量除以已到达的订单数。未到达订单的数量、
交期与到达时间都不进观测。
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from data.generator import Instance
from environment.problem import Problem

NOT_ARRIVED, WAITING, IN_PROCESS, COMPLETED, DISCARDED = 0, 1, 2, 3, 4

# no-op（保留产能）动作的哨兵三元组。全部调度规则按构造都是 non-delay，这是学习策略
# 能表达、规则无法表达的自由度
NOOP = (-1, -1, -1)

OP_DIM, MA_DIM, ACT_DIM, GLOBAL_DIM, GATE_DIM = 10, 3, 4, 5, 8


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

        self.top_k = int(cfg.get("action_space.order_top_k", 5))
        self.exposure = str(cfg.get("action_space.exposure", "hopeful_first"))
        if self.exposure not in ("edd", "hopeful_first"):
            raise ValueError(f"unknown action_space.exposure: {self.exposure}")
        self.exposure_threshold = float(cfg.get("action_space.exposure_threshold", 1.5))
        self.allow_noop = bool(cfg.get("action_space.allow_noop", True))
        self.max_consecutive_noop = int(cfg.get("action_space.max_consecutive_noop", 3))
        self.max_steps = int(cfg.get("episode.max_decision_steps", 200000))

        # 因果尺度：DDT 参数（交期政策）与最大工时（工艺数据）都是决策前已知的量
        self.t_ref = max(float(inst.meta.get("DDT", 0.0)) or float(np.median(inst.due_dates - inst.arrival_times)), 1.0)
        self.p_ref = max(float(inst.proc_times.max()), 1.0)

        # 静态结构表
        self._elig_matrix = inst.proc_times > 0                                   # [N, M]
        self._task_machines = [np.nonzero(self._elig_matrix[t])[0] for t in range(p.n_task)]
        self._elig_count = self._elig_matrix.sum(1).astype(np.float32)           # [N]
        self._machine_task_count = self._elig_matrix.sum(0).astype(np.float32)   # [M]
        self._task_product = (np.arange(p.n_task) // p.n_stage).astype(np.float32)
        self._task_stage = (np.arange(p.n_task) % p.n_stage).astype(np.float32)
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
        self.stats = StepStats()
        self._consecutive_noop = 0
        self.order_outcome = np.full(p.n_order, -1, dtype=np.int8)   # 1 按时 / 0 超期或丢弃 / -1 未定
        self._holds: List[dict] = []                                  # 全部等待记录（事后标签）
        self._open_holds: List[dict] = []                             # 尚未被再派工结算的记录
        self._feasible_machines: set = set()
        self._exposed_orders: set = set()
        self._grouped: Dict[int, np.ndarray] = {}
        self._w_orders = np.zeros(0, dtype=np.int64)
        self._w_tasks = np.zeros(0, dtype=np.int64)
        self._w_slack = np.zeros(0, dtype=np.float64)
        self._w_cr = np.zeros(0, dtype=np.float64)
        self._cand_stamp = -1
        self._advance_to_decision()

    # ------------------------------------------------------------------ 事件推进
    def _activate_arrivals(self) -> None:
        pending = self.status == NOT_ARRIVED
        if pending.any():
            arrived = pending & (self.inst.arrival_times <= self.now + 1e-9)
            if arrived.any():
                self.status[arrived] = WAITING

    def _release_machines(self) -> None:
        for m in np.nonzero((self.machine_busy_with >= 0) &
                            (self.machine_free_at <= self.now + 1e-9))[0]:
            order = int(self.machine_busy_with[m])
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
        self.done = True

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
            self._w_orders = waiting.astype(np.int64)
            self._w_tasks = np.zeros(0, dtype=np.int64)
            self._w_slack = np.zeros(0, dtype=np.float64)
            self._w_cr = np.zeros(0, dtype=np.float64)
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
        waiting, tasks, slack, cr = waiting[order], tasks[order], slack[order], cr[order]
        starts = np.flatnonzero(np.r_[True, tasks[1:] != tasks[:-1]])
        ends = np.r_[starts[1:], tasks.size]
        self._grouped = {int(tasks[s]): waiting[s:e] for s, e in zip(starts, ends)}
        self._w_orders, self._w_tasks, self._w_slack, self._w_cr = waiting, tasks, slack, cr

    def _feasible_actions(self) -> List[Tuple[int, int, int]]:
        """A_feas：每工序类型 top-K 暴露订单 x 空闲合格机器。"""
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
            for order in orders[: self.top_k]:
                for m in machines:
                    actions.append((int(task), int(m), int(order)))
        return actions

    def _noop_available(self) -> bool:
        """保留产能的防死锁条件：(i) 存在严格更晚的未来事件；(ii) 连续 no-op 未超上限。"""
        if not self.allow_noop:
            return False
        if self._consecutive_noop >= self.max_consecutive_noop:
            return False
        nxt = self._next_event_time()
        return nxt is not None and nxt > self.now + 1e-9

    def candidate_actions(self) -> List[Tuple[int, int, int]]:
        """当前决策点的候选动作：派工三元组，末尾可能带一个 no-op。"""
        feasible = self._feasible_actions()
        self._cand_stamp = self.step_count
        if not feasible:
            return []
        self.stats.n_feasible.append(len(feasible))
        self._feasible_machines = {a[1] for a in feasible}
        self._exposed_orders = {a[2] for a in feasible}
        if len(feasible) == 1:
            self.stats.singleton += 1
        actions = feasible
        if self._noop_available():
            self.stats.noop_offered += 1
            actions = actions + [NOOP]
        self.stats.n_candidates.append(len(actions))
        return actions

    # ------------------------------------------------------------------ 动作执行
    def step(self, action: Tuple[int, int, int]) -> Tuple[float, bool, dict]:
        if is_noop(action):
            return self._step_noop()
        self._consecutive_noop = 0
        task, machine, order = int(action[0]), int(action[1]), int(action[2])
        if self.status[order] != WAITING:
            raise ValueError(f"order {order} is not waiting (status={self.status[order]})")
        if self.inst.proc_times[task, machine] <= 0:
            raise ValueError(f"machine {machine} cannot process task {task}")
        if self._open_holds:
            # 被保留的机器第一次再派工时结算等待前景：给了等待时尚未暴露的订单才算"等来了"
            still_open = []
            for hold in self._open_holds:
                if machine in hold["machines"]:
                    hold["order"] = order
                    hold["new"] = order not in hold["exposed"]
                else:
                    still_open.append(hold)
            self._open_holds = still_open

        before_c, before_d = self.n_completed, self.n_discarded
        proc = float(self.inst.proc_times[task, machine])
        self.status[order] = IN_PROCESS
        self.machine_busy_with[machine] = order
        self.machine_free_at[machine] = self.now + proc
        self.machine_busy_time[machine] += proc
        self.step_count += 1
        self._advance_to_decision()
        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        return d_c / max(self.problem.n_order, 1), self.done, {
            "order": order, "hold_id": -1, "d_completed": d_c, "d_discarded": d_d, "noop": False}

    def _step_noop(self) -> Tuple[float, bool, dict]:
        """保留产能：本时刻不派工，推进到下一事件。奖励与派工同一计数式。"""
        before_c, before_d = self.n_completed, self.n_discarded
        self._consecutive_noop += 1
        self.stats.noop_used += 1
        self.step_count += 1
        machines = self._feasible_machines or {a[1] for a in self._feasible_actions()}
        hold_id = len(self._holds)
        hold = {"machines": set(machines), "exposed": set(self._exposed_orders), "order": None, "new": False}
        self._holds.append(hold)
        self._open_holds.append(hold)

        nxt = self._next_event_time()
        if nxt is None or nxt <= self.now + 1e-9:        # 防死锁条件已排除，稳妥起见再兜一层
            self.done = True
        else:
            self.stats.held_time += len(machines) * (nxt - self.now)
            self.now = nxt
            self._advance_to_decision()
        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        return d_c / max(self.problem.n_order, 1), self.done, {
            "order": -1, "hold_id": hold_id, "d_completed": d_c, "d_discarded": d_d, "noop": True}

    # ------------------------------------------------------------------ 观测
    def observation(self, actions) -> dict:
        """因果观测：工序类型节点、机器节点、候选动作特征、全局量与门控工况特征。"""
        t0 = _time.perf_counter()
        if self._cand_stamp != self.step_count:
            self._refresh_waiting()
        p, inst = self.problem, self.inst
        n_task, n_machine = p.n_task, p.n_machine
        n_arr = max(int(np.count_nonzero(self.status != NOT_ARRIVED)), 1)
        idle = self._idle_mask()
        t_ref, p_ref = self.t_ref, self.p_ref

        # ---- 工序类型节点
        op = np.zeros((n_task, OP_DIM), dtype=np.float32)
        op[:, 0] = self._elig_count / n_machine
        op[:, 1] = (self._elig_matrix @ idle.astype(np.float32)) / np.maximum(self._elig_count, 1.0)
        waiting_count = np.zeros(n_task, dtype=np.float32)
        if self._w_tasks.size:
            tasks, slack = self._w_tasks, self._w_slack / t_ref
            starts = np.flatnonzero(np.r_[True, tasks[1:] != tasks[:-1]])
            ids = tasks[starts]
            counts = np.diff(np.r_[starts, tasks.size]).astype(np.float64)
            s_sum = np.add.reduceat(slack, starts)
            s_min = np.minimum.reduceat(slack, starts)
            s_max = np.maximum.reduceat(slack, starts)
            s_sq = np.add.reduceat(slack * slack, starts)
            mean = s_sum / counts
            op[ids, 2] = mean
            op[ids, 3] = s_min
            op[ids, 4] = s_max
            op[ids, 5] = np.sqrt(np.maximum(s_sq / counts - mean * mean, 0.0))
            waiting_count[ids] = counts
        active = (self.status == WAITING) | (self.status == IN_PROCESS)
        if active.any():
            per = np.zeros((p.n_product, p.n_stage), dtype=np.float64)
            np.add.at(per, (inst.order_product[active], self.stage[active]), 1.0)
            op[:, 6] = np.cumsum(per, axis=1).reshape(-1) / n_arr    # 本类型及其上游的在制订单
        op[:, 7] = waiting_count / n_arr
        op[:, 8] = self._task_product / max(p.n_product - 1, 1)
        op[:, 9] = self._task_stage / max(p.n_stage - 1, 1)

        # ---- 机器节点
        ma = np.zeros((n_machine, MA_DIM), dtype=np.float32)
        ma[:, 0] = self._machine_task_count / n_task
        ma[:, 1] = np.minimum(np.maximum(self.machine_free_at - self.now, 0.0) / t_ref, 10.0)
        ma[:, 2] = idle

        # ---- 候选动作
        nxt = self._next_event_time()
        gap_raw = max(float(nxt) - self.now, 0.0) if nxt is not None else 0.0
        n_act = len(actions)
        act = np.zeros((n_act, ACT_DIM), dtype=np.float32)
        index = np.zeros((n_act, 2), dtype=np.int64)
        arr = np.asarray(actions, dtype=np.int64).reshape(n_act, 3)
        dispatch = arr[:, 0] >= 0
        if dispatch.any():
            d_task, d_mach, d_ord = arr[dispatch, 0], arr[dispatch, 1], arr[dispatch, 2]
            slack = inst.due_dates[d_ord] - self.now
            need = np.maximum(p.residual[inst.order_product[d_ord], self.stage[d_ord]], 1e-9)
            proc = inst.proc_times[d_task, d_mach].astype(np.float64)
            act[dispatch, 0] = slack / t_ref
            act[dispatch, 1] = proc / p_ref
            act[dispatch, 2] = np.minimum(slack / need, 10.0)
            index[dispatch, 0] = d_task
            index[dispatch, 1] = d_mach
            max_cr = float(np.minimum(slack / need, 10.0).max())
            min_proc = float(proc.min()) / p_ref
            n_dispatch = int(dispatch.sum())
            with_work = len(set(int(m) for m in d_mach))
        else:
            max_cr = min_proc = 0.0
            n_dispatch = with_work = 0
        if (~dispatch).any():
            act[~dispatch, 1] = min(gap_raw / t_ref, 10.0)
            act[~dispatch, 3] = 1.0

        # ---- 全局量（都以已到达订单为分母）
        recent = int(np.count_nonzero((inst.arrival_times > self.now - t_ref)
                                      & (inst.arrival_times <= self.now + 1e-9)))
        rate = recent / t_ref
        eta_t = self.n_completed / n_arr
        global_feat = np.asarray([
            eta_t,
            self.n_discarded / n_arr,
            (n_arr - self.n_completed - self.n_discarded) / n_arr,
            min(rate * p_ref, 10.0) / 10.0,
            float(idle.sum()) / n_machine,
        ], dtype=np.float32)

        # ---- 门控工况特征
        backlog = 0.0
        if self._w_tasks.size:
            load = (waiting_count[:, None] * inst.proc_times).sum(0)      # 每台机器排队的等待工作量
            backlog = float(load.max()) / p_ref
        viable = float(np.mean(self._w_cr >= self.exposure_threshold)) if self._w_cr.size else 0.0
        gate = np.asarray([
            min(gap_raw / t_ref, 10.0) / 10.0,
            with_work / n_machine,
            min(n_dispatch / n_machine, 5.0) / 5.0,
            min(rate * gap_raw, 10.0) / 10.0,
            min(backlog, 20.0) / 20.0,
            viable,
            max_cr / 10.0,
            min_proc,
        ], dtype=np.float32)

        self.stats.t_obs += _time.perf_counter() - t0
        return {"op": op, "ma": ma, "act_feat": act, "act_index": index,
                "global_feat": global_feat, "gate_feat": gate, "eta_t": np.float32(eta_t)}

    # ------------------------------------------------------------------ 标签与指标
    def hold_labels(self) -> Dict[int, int]:
        """等待前景标签：被保留的机器下一次派工给了等待时尚未暴露的订单且按时完成 -> 1；
        给了已暴露订单，或新订单未按时 -> 0；尚未再派工或结果未定 -> -1（不进损失）。"""
        out: Dict[int, int] = {}
        for i, hold in enumerate(self._holds):
            if hold["order"] is None:
                out[i] = -1
            elif not hold["new"]:
                out[i] = 0
            else:
                out[i] = int(self.order_outcome[hold["order"]])
        return out

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
