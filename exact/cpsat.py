"""精确参照：CP-SAT 离线最优（知道全部到达）与滚动精确重优化（只知道已到达的订单）。

两者都在整数时间网格上求解（工时为整数，网格保序），排程回放进离散事件环境核对一致。
滚动重优化是"优化完美但不能预判"的参照：每个到达时刻对已知订单重排，已开工工序冻结。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from environment.problem import Problem

# 按时判定的容差，与仿真器 `SchedulingEnv._release_machines` 一致
DUE_TOL = 1e-9

Schedule = Dict[Tuple[int, int], Tuple[int, float, float]]   # (order, stage) -> (machine, start, end)


@dataclass
class ExactResult:
    eta: float
    status: str
    solver: str
    seconds: float
    assignment: Schedule
    n_completed: int = 0
    upper: float = float("nan")          # 求解器证得的 eta 上界；OPTIMAL 时等于 eta


@dataclass
class OnlineResult:
    eta: float
    status: str                          # 每次重解都证得最优为 OPTIMAL，否则 FEASIBLE
    solver: str
    seconds: float
    assignment: Schedule
    n_completed: int = 0
    n_solves: int = 0
    all_optimal: bool = True
    statuses: List[str] = field(default_factory=list)


def _horizon(problem: Problem) -> int:
    inst = problem.inst
    return int(inst.due_dates.max() + problem.residual_from(0, 0) + inst.proc_times.max() * inst.order_count)


def _require_integer_proc_times(problem: Problem) -> None:
    p = np.asarray(problem.inst.proc_times, dtype=np.float64)
    if not np.all(np.abs(p - np.round(p)) <= 1e-9):
        raise ValueError("CP-SAT 的保序时间尺度要求整数工时；本仓库的算例生成器按构造满足"
                         "（data/generator.py 用 rng.integers 抽工时）")


class _TimeScale:
    """保序整数时间尺度：把连续时间模型无损地搬到 CP-SAT 的整数域上。

    工时为整数时，左对齐排程的每个开工/完工时刻都形如 a_k + n（某订单的到达时刻加一个
    整数）。设 f_k 为 a_k 的小数部分，K 为 {f_k} 中不同值的个数，rank(f) 为 f 在其中的
    名次（0..K-1）。映射
        phi(I + f_k) = I * K + rank(f_k)          （I 为整数）
    是集合 {f_k + 整数} 到整数集的严格单调双射，且 phi(e + p) = phi(e) + p * K。于是到达、
    前后序、同机不重叠与交期四类比较在映射前后逐一等价，CP-SAT 在整数域上的最优值
    就是连续时间模型的最优值。交期取该集合中不超过 d_s 的最大元素。

    旧实现把到达向上取整、交期向下取整，只会低估 eta_off，而低估 eta_off 会让最优性
    间隙偏小——那条注释里"方向保守、gap 只会偏大"的说法是反的。现已不存在这项偏差。
    """

    def __init__(self, arrivals: np.ndarray, orders: Iterable[int]) -> None:
        self.arrivals = arrivals
        frac = {int(k): float(arrivals[k] - math.floor(arrivals[k])) for k in orders}
        self.fracs = sorted(set(frac.values()))
        self.K = max(len(self.fracs), 1)
        pos = {f: i for i, f in enumerate(self.fracs)}
        self.rank = {k: pos[f] for k, f in frac.items()}
        self.rep: Dict[int, int] = {}
        for k in sorted(frac):
            self.rep.setdefault(self.rank[k], k)

    def point(self, integer: int, order: int) -> int:
        """phi(integer + f_order)。"""
        return int(integer) * self.K + self.rank[order]

    def arrival(self, order: int) -> int:
        return self.point(math.floor(self.arrivals[order]), order)

    def duration(self, p: float) -> int:
        return int(round(float(p))) * self.K

    def due(self, d: float) -> int:
        return max(math.floor(d + DUE_TOL - f) * self.K + r for r, f in enumerate(self.fracs))

    def time(self, point: int) -> float:
        integer, r = divmod(int(point), self.K)
        return float(integer) + self.fracs[r]

    def pair(self, point: int) -> Tuple[int, int]:
        """整数点 -> (整数部分, 同小数部分的一张订单)，换一个尺度后仍能精确还原。"""
        integer, r = divmod(int(point), self.K)
        return integer, self.rep[r]


def _left_shift(ops: Dict[Tuple[int, int], Tuple[int, float]],
                duration: Callable[[int, int, int], float],
                ready: Dict[int, float],
                machine_free: Dict[int, float]) -> Dict[Tuple[int, int], Tuple[int, float]]:
    """保持分配与各机加工顺序，把每道工序提前到最早可开工时刻（半主动排程）。

    完工时刻只会提前，按时完工的订单仍按时完工；左对齐后每个开工时刻都是某个事件
    时刻（到达或完工），离散事件环境因此能逐道回放。
    """
    order_free = dict(ready)
    free = dict(machine_free)
    shifted = {}
    for (s, j) in sorted(ops, key=lambda key: (ops[key][1], key)):
        m, _ = ops[(s, j)]
        start = max(order_free[s], free.get(m, order_free[s]))
        end = start + duration(s, j, m)
        shifted[(s, j)] = (m, start)
        order_free[s] = end
        free[m] = end
    return shifted


def _cpsat_solver(time_limit_s: float, workers: int, work_limit: float | None = None):
    """work_limit 是 CP-SAT 的确定性时间（与机器快慢无关的工作量单位）。单线程且只有它
    起作用时，求解过程逐位可复现；墙钟时限只作兜底。"""
    from ortools.sat.python import cp_model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = int(workers)
    if work_limit is not None:
        solver.parameters.max_deterministic_time = float(work_limit)
    return solver


def _build_cpsat(problem: Problem, scale: _TimeScale, orders: Sequence[int],
                 first_stage: Dict[int, int], ready: Dict[int, int],
                 fixed: Dict[int, List[Tuple[int, int]]], horizon: int):
    """已知订单的剩余工序模型：最大化按时完工的订单数。

    z_s = 1 时订单 s 的剩余工序全部排上并按时完工（稿件 Eq. 9）；z_s = 0 时剩余工序全部
    不排（稿件 Eq. 2 的丢弃，不占产能），其开工/完工变量钉在 0，便于第二阶段只对真正
    排上的工序求和。`fixed` 是已开工、不可撤销的工序在各机器上占用的区间。
    """
    from ortools.sat.python import cp_model

    inst, model = problem.inst, cp_model.CpModel()
    z, starts, ends, lits = {}, {}, {}, {}
    per_machine: Dict[int, list] = {m: [] for m in range(problem.n_machine)}
    for s in orders:
        z[s] = model.new_bool_var(f"z_{s}")
        prev = None
        for j in range(first_stage[s], problem.n_stage):
            task = problem.task_of(s, j)
            st = model.new_int_var(0, horizon, f"st_{s}_{j}")
            en = model.new_int_var(0, horizon, f"en_{s}_{j}")
            present = []
            for m in problem.eligible[task]:
                lit = model.new_bool_var(f"x_{s}_{j}_{m}")
                per_machine[m].append(model.new_optional_interval_var(
                    st, scale.duration(inst.proc_times[task, m]), en, lit, f"iv_{s}_{j}_{m}"))
                present.append(lit)
                lits[(s, j, m)] = lit
            model.add(sum(present) == z[s])                       # 稿件 Eq. (2)
            model.add(st == 0).only_enforce_if(~z[s])
            model.add(en == 0).only_enforce_if(~z[s])
            if prev is None:
                model.add(st >= ready[s]).only_enforce_if(z[s])   # 稿件 Eq. (4)
            else:
                model.add(st >= prev)                             # 稿件 Eq. (5)
            starts[(s, j)], ends[(s, j)], prev = st, en, en
        model.add(prev <= scale.due(float(inst.due_dates[s]))).only_enforce_if(z[s])   # 稿件 Eq. (9)
    for m, busy in fixed.items():
        for k, (start, dur) in enumerate(busy):
            per_machine[m].append(model.new_fixed_size_interval_var(start, dur, f"fix_{m}_{k}"))
    for ivs in per_machine.values():                              # 稿件 Eqs. (6)-(8)
        if len(ivs) > 1:
            model.add_no_overlap(ivs)
    return model, z, starts, ends, lits


def _extract(problem: Problem, solver, orders, first_stage, z, starts, lits):
    ops: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for s in orders:
        if not solver.boolean_value(z[s]):
            continue
        for j in range(first_stage[s], problem.n_stage):
            m = next(m for m in problem.eligible[problem.task_of(s, j)]
                     if solver.boolean_value(lits[(s, j, m)]))
            ops[(s, j)] = (m, int(solver.value(starts[(s, j)])))
    return ops


def _count_on_time(problem: Problem, assignment: Schedule) -> int:
    last = problem.n_stage - 1
    return sum(1 for (s, j), (_, _, end) in assignment.items()
               if j == last and end <= float(problem.inst.due_dates[s]) + DUE_TOL)


def solve_cpsat(problem: Problem, time_limit_s: float = 3600.0, workers: int = 8) -> ExactResult:
    """离线 clairvoyant 最优：知道全部到达，最大化按时完工订单数。"""
    from ortools.sat.python import cp_model

    _require_integer_proc_times(problem)
    inst, S = problem.inst, problem.n_order
    started = time.perf_counter()
    orders = list(range(S))
    scale = _TimeScale(inst.arrival_times, orders)
    horizon = (_horizon(problem) + 1) * scale.K
    first = {s: 0 for s in orders}
    ready = {s: scale.arrival(s) for s in orders}
    model, z, starts, _, lits = _build_cpsat(problem, scale, orders, first, ready, {}, horizon)
    model.maximize(sum(z.values()))

    solver = _cpsat_solver(time_limit_s, workers)
    status = solver.solve(model)
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(eta=float("nan"), status=name, solver="cpsat",
                           seconds=time.perf_counter() - started, assignment={})

    ops = _extract(problem, solver, orders, first, z, starts, lits)
    dur = lambda s, j, m: scale.duration(inst.proc_times[problem.task_of(s, j), m])  # noqa: E731
    shifted = _left_shift(ops, dur, ready, {})
    assignment = {key: (m, scale.time(point), scale.time(point + dur(key[0], key[1], m)))
                  for key, (m, point) in shifted.items()}
    completed = int(round(solver.objective_value))
    return ExactResult(eta=completed / max(S, 1), status=name, solver="cpsat",
                       seconds=time.perf_counter() - started, assignment=assignment,
                       n_completed=completed,
                       upper=min(float(solver.best_objective_bound), S) / max(S, 1))


# --------------------------------------------------------------------------- #
# 排程校验的辅助类型（与 CP-SAT 模型共用）
# --------------------------------------------------------------------------- #
def solve_online_reoptimization(problem: Problem, time_limit_per_solve: float = 60.0,
                                workers: int = 1, tie_break_work: float = 1.0) -> OnlineResult:
    """精确但短视的在线参照（稿件 Table T-NEW-5 的 Online re-opt. 列）。

    每个到达时刻 t_k：只看已到达的订单；已开工的工序冻结为固定区间；其余工序在
    t_k 之后重排，第一阶段最大化已知订单的按时完工数（证得最优），第二阶段在该数不变
    的前提下最小化已排工序的完工时刻之和，把产能尽早腾给未来订单。计划左对齐后，
    在下一个到达时刻之前开工的工序即被提交（开工不可撤销）；最后一个到达之后整份
    计划提交。未来到达的任何信息都不进入模型，连时间尺度也只由已知订单构造。

    上一轮计划中未提交的部分在本轮仍然可行，作为提示交给 CP-SAT，所以每轮至少不差于
    沿用旧计划。它不是在线策略能达到的上界：预留产能或有意等待的策略可以比它好。

    可复现性：默认单线程；第二阶段只按确定性工作量（tie_break_work）截断，不按墙钟，
    所以提交的计划、从而 eta_online 与机器快慢无关。small 档实测：tie_break_work 取
    0.5、2、5 时 eta_online 逐算例相同，取 0（不做第二阶段）时有算例低 0.05。
    """
    from ortools.sat.python import cp_model

    _require_integer_proc_times(problem)
    inst, S, J = problem.inst, problem.n_order, problem.n_stage
    started = time.perf_counter()
    arrivals = inst.arrival_times
    epochs = sorted(set(float(a) for a in arrivals))
    base_horizon = _horizon(problem) + 1

    frozen: Dict[Tuple[int, int], Tuple[int, int, int]] = {}   # (s, j) -> (machine, 整数部分, 订单)
    plan: Dict[Tuple[int, int], Tuple[int, int, int]] = {}     # 上轮未提交的计划，同一编码
    known: List[int] = []
    statuses: List[str] = []
    proc = lambda s, j, m: float(inst.proc_times[problem.task_of(s, j), m])   # noqa: E731

    for e, t in enumerate(epochs):
        arriving = [s for s in range(S) if float(arrivals[s]) == t]
        known.extend(arriving)
        nxt = epochs[e + 1] if e + 1 < len(epochs) else math.inf
        scale = _TimeScale(arrivals, known)
        release = scale.arrival(arriving[0])
        horizon = base_horizon * scale.K
        dur = lambda s, j, m: scale.duration(proc(s, j, m))                   # noqa: E731

        first, ready = {}, {}
        for s in known:
            k = 0
            while k < J and (s, k) in frozen:
                k += 1
            if k == J:
                continue
            first[s] = k
            ready[s] = release
            if k > 0:
                m, integer, rep = frozen[(s, k - 1)]
                ready[s] = max(release, scale.point(integer, rep) + dur(s, k - 1, m))
        active = sorted(first)
        fixed: Dict[int, List[Tuple[int, int]]] = {}
        machine_free: Dict[int, int] = {}
        for (s, j), (m, integer, rep) in frozen.items():
            begin = scale.point(integer, rep)
            end = begin + dur(s, j, m)
            if end > release:
                fixed.setdefault(m, []).append((begin, dur(s, j, m)))
                machine_free[m] = max(machine_free.get(m, release), end)
        if not active:
            statuses.append("NO_WORK")
            continue

        model, z, starts, ends, lits = _build_cpsat(problem, scale, active, first, ready, fixed, horizon)
        for s in active:                                       # 旧计划仍可行，作提示
            keep = all((s, j) in plan for j in range(first[s], J))
            model.add_hint(z[s], 1 if keep else 0)
            if keep:
                for j in range(first[s], J):
                    m, integer, rep = plan[(s, j)]
                    model.add_hint(starts[(s, j)], scale.point(integer, rep))
                    for mm in problem.eligible[problem.task_of(s, j)]:
                        model.add_hint(lits[(s, j, mm)], 1 if mm == m else 0)
        model.maximize(sum(z.values()))
        solver = _cpsat_solver(time_limit_per_solve, workers)
        status = solver.solve(model)
        statuses.append(solver.status_name(status))
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            ops = _extract(problem, solver, active, first, z, starts, lits)
            best = int(round(solver.objective_value))
            # 第二阶段：按时数不变，完工时刻之和最小
            model.clear_hints()
            for s in active:
                model.add_hint(z[s], solver.boolean_value(z[s]))
            for (s, j), (m, point) in ops.items():
                model.add_hint(starts[(s, j)], point)
                for mm in problem.eligible[problem.task_of(s, j)]:
                    model.add_hint(lits[(s, j, mm)], 1 if mm == m else 0)
            model.add(sum(z.values()) == best)
            model.clear_objective()
            model.minimize(sum(ends.values()))
            second = _cpsat_solver(time_limit_per_solve, workers, work_limit=tie_break_work)
            if second.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                ops = _extract(problem, second, active, first, z, starts, lits)
        else:                                                  # 无解返回时沿用旧计划
            ops = {key: (m, scale.point(integer, rep)) for key, (m, integer, rep) in plan.items()}

        shifted = _left_shift(ops, dur, ready, machine_free)
        plan = {}
        for (s, j), (m, point) in shifted.items():
            integer, rep = scale.pair(point)
            if scale.time(point) < nxt - 1e-9:
                frozen[(s, j)] = (m, integer, rep)
            else:
                plan[(s, j)] = (m, integer, rep)

    frac = lambda k: float(arrivals[k] - math.floor(arrivals[k]))        # noqa: E731
    assignment = {(s, j): (m, float(integer) + frac(rep), float(integer + round(proc(s, j, m))) + frac(rep))
                  for (s, j), (m, integer, rep) in frozen.items()}
    completed = _count_on_time(problem, assignment)
    all_optimal = all(st in ("OPTIMAL", "NO_WORK") for st in statuses)
    return OnlineResult(eta=completed / max(S, 1), status="OPTIMAL" if all_optimal else "FEASIBLE",
                        solver="cpsat-online", seconds=time.perf_counter() - started,
                        assignment=assignment, n_completed=completed,
                        n_solves=sum(1 for st in statuses if st != "NO_WORK"),
                        all_optimal=all_optimal, statuses=statuses)


# --------------------------------------------------------------------------- #
# 回放校验
# --------------------------------------------------------------------------- #
def _replay_in_env(problem: Problem, assignment: Schedule, cfg, tol: float) -> Tuple[int, List[str]]:
    """驱动真实的离散事件环境：排程在某时刻开工的工序就在该时刻派工，其余时刻空闲。"""
    from configs.config import Config
    from environment.env import NOOP, WAITING, SchedulingEnv

    env = SchedulingEnv(problem.inst, Config(cfg.to_dict()))
    queue = sorted(assignment.items(), key=lambda kv: (kv[1][1], kv[0]))
    issues: List[str] = []
    head = 0
    while not env.done and head < len(queue):
        (s, j), (m, start, _) = queue[head]
        if start < env.now - tol:
            issues.append(f"env: order {s} stage {j} due to start at {start:.4f}, env already at {env.now:.4f}")
            break
        if start > env.now + tol:
            env.step(NOOP)
            continue
        task = problem.task_of(s, j)
        if env.status[s] != WAITING or int(env.stage[s]) != j:
            issues.append(f"env: order {s} not waiting at stage {j} at {env.now:.4f}")
            break
        if env.machine_busy_with[m] >= 0 or env.machine_free_at[m] > env.now + 1e-9 \
                or problem.rates[task, m] <= 0:
            issues.append(f"env: machine {m} cannot take order {s} stage {j} at {env.now:.4f}")
            break
        env.step((task, m, s))
        head += 1
    if head < len(queue) and not issues:
        issues.append(f"env: episode ended with {len(queue) - head} operations never dispatched")
    while not env.done:                               # 让已派工序全部完工，结清计数
        env.step(NOOP)
    return int(env.n_completed), issues


def replay_check(problem: Problem, result, cfg=None, tol: float = 1e-6) -> Dict[str, object]:
    """核对一份排程与问题定义、与仿真器是否一致。

    静态核对：机器合格、工时、到达与前后序、同机不重叠、按时完工数与求解器报告的
    一致。给出 cfg 时再驱动真实的离散事件环境回放，环境自己记下的按时完工数必须与
    求解器的相同——任何一条约束只存在于其中一侧，都会在这里暴露。允许订单只排了前几
    道工序（在线排程会出现），全部工序按时完工才计按时。
    """
    inst = problem.inst
    assignment = result.assignment               # 空排程也要核对：eta = 0 的最优解就是空排程
    issues: List[str] = []
    by_machine: Dict[int, List[Tuple[float, float, int, int]]] = {}
    orders = sorted({s for s, _ in assignment})
    for s in orders:
        ready = float(inst.arrival_times[s])
        for j in range(problem.n_stage):
            if (s, j) not in assignment:
                if any((s, jj) in assignment for jj in range(j + 1, problem.n_stage)):
                    issues.append(f"order {s}: stage {j} missing before a later stage")
                break
            m, start, end = assignment[(s, j)]
            task = problem.task_of(s, j)
            if m < 0 or m >= problem.n_machine or problem.rates[task, m] <= 0:
                issues.append(f"order {s} stage {j}: machine {m} not eligible")
                break
            if start + tol < ready:
                issues.append(f"order {s} stage {j}: start {start:.4f} < ready {ready:.4f}")
            if abs((end - start) - float(inst.proc_times[task, m])) > tol:
                issues.append(f"order {s} stage {j}: duration {end - start:.4f} != {inst.proc_times[task, m]}")
            by_machine.setdefault(m, []).append((start, end, s, j))
            ready = end
    for m, ops in by_machine.items():
        ops.sort()
        for (s1, e1, o1, j1), (s2, _, o2, j2) in zip(ops, ops[1:]):
            if s2 + tol < e1:
                issues.append(f"machine {m}: ({o1},{j1}) ends {e1:.4f} after ({o2},{j2}) starts {s2:.4f}")
    completed = _count_on_time(problem, assignment)
    eta_replay = completed / max(problem.n_order, 1)
    out = {"eta_solver": result.eta, "eta_replay": eta_replay, "n_orders_checked": len(orders)}
    match = not issues and completed == result.n_completed
    if cfg is not None:
        env_completed, env_issues = _replay_in_env(problem, assignment, cfg, tol)
        issues.extend(env_issues)
        out["eta_env"] = env_completed / max(problem.n_order, 1)
        match = match and not env_issues and env_completed == result.n_completed
    out.update({"match": bool(match), "n_issues": len(issues), "issues": issues[:5]})
    return out
