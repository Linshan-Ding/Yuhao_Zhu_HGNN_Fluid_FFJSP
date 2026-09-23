"""离线 clairvoyant 精确解、在线滚动重优化与解回放校验（稿件 §3.2、§5.8）。

同一个连续时间模型，两个免授权求解器：
  * CP-SAT（OR-Tools）—— 在保序的整数时间尺度上求解（见 `_TimeScale`）。工时为整数时
    它与连续时间模型逐解等价，不是近似；
  * HiGHS MILP（SciPy 自带的 `scipy.optimize.milp`）—— 稿件 Eqs. (1)-(11) 的逐条直译，
    含丢弃变量 v_s、析取排序变量 y 与大常数 L。
一致性检查是双向的：HiGHS 证得的最优值必须等于 CP-SAT 的最优值；CP-SAT 的最优排程
代入字面 MILP 的约束矩阵必须逐行满足（`milp_certificate`）。两个求解器都证得最优而
数值不同，说明两份公式化之一有错。

eta_off 是知道全部未来到达才能达到的离线最优。`solve_online_reoptimization` 是在线
参照：每个到达时刻只对已知订单精确重排，不看未来。它精确但短视，既不是在线策略的
上界也不是下界。

全部排程都以连续时间的左对齐形式返回；`replay_check` 除了逐条核对约束，还能驱动
真实的离散事件环境逐道工序回放，核对开工时刻与按时完工数。
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
# 稿件 Eqs. (1)-(11) 的 MILP：HiGHS 求解与证书核对共用同一份矩阵
# --------------------------------------------------------------------------- #
@dataclass
class _MilpStructure:
    c: np.ndarray
    matrix: object                        # scipy.sparse.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray
    integrality: np.ndarray
    x_col: Dict[Tuple[int, int, int], int]
    y_col: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], int]
    big_l: float


def _milp_structure(problem: Problem) -> _MilpStructure:
    """列依次为 z_s、v_s、C_sj（下标 2S + s*J + j）、x_sjm、y；目标为 min -sum z。"""
    from scipy.sparse import coo_matrix

    inst = problem.inst
    S, J = problem.n_order, problem.n_stage
    L = float(_horizon(problem))
    n = 2 * S + S * J
    x_col: Dict[Tuple[int, int, int], int] = {}
    for s in range(S):
        for j in range(J):
            for m in problem.eligible[problem.task_of(s, j)]:
                x_col[(s, j, m)] = n
                n += 1
    ops_on_machine: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(problem.n_machine)}
    for s in range(S):
        for j in range(J):
            for m in problem.eligible[problem.task_of(s, j)]:
                ops_on_machine[m].append((s, j))
    y_col = {}
    for m, ops in ops_on_machine.items():
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                y_col[(m, ops[a], ops[b])] = n
                n += 1

    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(coeffs, lo, hi):
        r = len(lb)
        for col, val in coeffs:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        lb.append(lo)
        ub.append(hi)

    C = lambda s, j: 2 * S + s * J + j                                      # noqa: E731
    for s in range(S):
        add([(s, 1.0), (S + s, 1.0)], -np.inf, 1.0)                          # Eq. (3)
        for j in range(J):
            task = problem.task_of(s, j)
            add([(x_col[(s, j, m)], 1.0) for m in problem.eligible[task]] + [(S + s, 1.0)],
                1.0, 1.0)                                                    # Eq. (2)
            dur = [(x_col[(s, j, m)], -float(inst.proc_times[task, m])) for m in problem.eligible[task]]
            if j == 0:
                add([(C(s, 0), 1.0)] + dur, float(inst.arrival_times[s]), np.inf)       # Eq. (4)
            else:
                add([(C(s, j), 1.0), (C(s, j - 1), -1.0)] + dur, 0.0, np.inf)          # Eq. (5)
        add([(C(s, J - 1), 1.0), (s, L)], -np.inf, float(inst.due_dates[s]) + L)       # Eq. (9)
    for (m, (s1, j1), (s2, j2)), y in y_col.items():                                    # Eqs. (7)-(8)
        x1, x2 = x_col[(s1, j1, m)], x_col[(s2, j2, m)]
        p1 = float(inst.proc_times[problem.task_of(s1, j1), m])
        p2 = float(inst.proc_times[problem.task_of(s2, j2), m])
        # C2 >= C1 + p2 - L(1 - y) - L(2 - x1 - x2)
        add([(C(s2, j2), 1.0), (C(s1, j1), -1.0), (y, -L), (x1, -L), (x2, -L)], p2 - 3 * L, np.inf)
        # C1 >= C2 + p1 - L y - L(2 - x1 - x2)
        add([(C(s1, j1), 1.0), (C(s2, j2), -1.0), (y, L), (x1, -L), (x2, -L)], p1 - 2 * L, np.inf)

    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lb), n)).tocsr()
    c = np.zeros(n)
    c[:S] = -1.0                                                             # Eq. (1)：max sum z
    integrality = np.ones(n)
    col_lb, col_ub = np.zeros(n), np.ones(n)
    integrality[2 * S: 2 * S + S * J] = 0                                    # C 连续、非负
    col_ub[2 * S: 2 * S + S * J] = np.inf
    return _MilpStructure(c=c, matrix=matrix, row_lb=np.asarray(lb), row_ub=np.asarray(ub),
                          col_lb=col_lb, col_ub=col_ub, integrality=integrality,
                          x_col=x_col, y_col=y_col, big_l=L)


def solve_milp(problem: Problem, time_limit_s: float = 3600.0) -> ExactResult:
    """稿件 Eqs. (1)-(11) 的 MILP 直译，SciPy 自带的 HiGHS 分支定界求解，免授权。"""
    from scipy.optimize import Bounds, LinearConstraint, milp

    inst, S, J = problem.inst, problem.n_order, problem.n_stage
    started = time.perf_counter()
    ms = _milp_structure(problem)
    res = milp(ms.c, integrality=ms.integrality, bounds=Bounds(ms.col_lb, ms.col_ub),
               constraints=LinearConstraint(ms.matrix, ms.row_lb, ms.row_ub),
               options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0, "disp": False})
    seconds = time.perf_counter() - started
    bound = getattr(res, "mip_dual_bound", None)
    upper = min(-float(bound), S) / max(S, 1) if bound is not None and np.isfinite(bound) else float("nan")
    if res.x is None:
        status = {2: "INFEASIBLE", 3: "UNBOUNDED"}.get(res.status, "NO_SOLUTION")
        return ExactResult(eta=float("nan"), status=status, solver="highs-milp",
                           seconds=seconds, assignment={}, upper=upper)

    x = res.x
    ops = {}
    for s in range(S):
        if x[s] < 0.5:
            continue
        for j in range(J):
            m = next(m for m in problem.eligible[problem.task_of(s, j)] if x[ms.x_col[(s, j, m)]] > 0.5)
            end = float(x[2 * S + s * J + j])
            ops[(s, j)] = (m, end - float(inst.proc_times[problem.task_of(s, j), m]))
    dur = lambda s, j, m: float(inst.proc_times[problem.task_of(s, j), m])  # noqa: E731
    shifted = _left_shift(ops, dur, {s: float(inst.arrival_times[s]) for s in range(S)}, {})
    assignment = {key: (m, start, start + dur(key[0], key[1], m)) for key, (m, start) in shifted.items()}
    completed = int(round(float(np.sum(x[:S]))))
    status = "OPTIMAL" if res.status == 0 else ("TIME_LIMIT" if res.status == 1 else f"HIGHS_{res.status}")
    return ExactResult(eta=completed / max(S, 1), status=status, solver="highs-milp",
                       seconds=seconds, assignment=assignment, n_completed=completed,
                       upper=upper if res.status != 0 else completed / max(S, 1))


def milp_certificate(problem: Problem, result, tol: float = 1e-6) -> Dict[str, object]:
    """把一份排程代入字面 MILP（Eqs. 1-11）的约束矩阵，逐行核对。

    由排程重建全部决策变量：按时完工的订单 z=1，全部工序排上但超期的订单 z=v=0，
    其余订单 v=1（丢弃，完工变量取到达时刻）；x 取所用机器；C 取完工时刻；同机的
    两道工序按开工先后定 y。任何一行不满足都说明 CP-SAT 模型与 MILP 不是同一个问题。
    """
    inst, S, J = problem.inst, problem.n_order, problem.n_stage
    ms = _milp_structure(problem)
    vec = np.zeros(ms.c.size)
    assignment = result.assignment
    for s in range(S):
        complete = all((s, j) in assignment for j in range(J))
        if complete:
            on_time = assignment[(s, J - 1)][2] <= float(inst.due_dates[s]) + DUE_TOL
            vec[s] = 1.0 if on_time else 0.0
            for j in range(J):
                m, _, end = assignment[(s, j)]
                vec[ms.x_col[(s, j, m)]] = 1.0
                vec[2 * S + s * J + j] = end
        else:
            vec[S + s] = 1.0
            vec[2 * S + s * J: 2 * S + (s + 1) * J] = float(inst.arrival_times[s])
    for (m, op1, op2), y in ms.y_col.items():
        if op1 in assignment and op2 in assignment and assignment[op1][0] == m == assignment[op2][0]:
            vec[y] = 1.0 if assignment[op1][1] < assignment[op2][1] else 0.0

    activity = ms.matrix @ vec
    row_violation = float(np.max(np.concatenate([ms.row_lb - activity, activity - ms.row_ub, [0.0]])))
    col_violation = float(np.max(np.concatenate([ms.col_lb - vec, vec - ms.col_ub, [0.0]])))
    violation = max(row_violation, col_violation)
    objective = int(round(float(vec[:S].sum())))
    return {"ok": bool(violation <= tol and objective == result.n_completed),
            "max_violation": violation, "n_rows": int(ms.matrix.shape[0]),
            "objective": objective}


# --------------------------------------------------------------------------- #
# 在线滚动重优化：每个到达时刻只对已知订单精确重排
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

    quiet = Config(cfg.to_dict())                     # 塑形项不改变动态，关掉以免逐步求解流体 LP
    quiet.set("reward.potential_weight", 0.0)
    quiet.set("reward.fluid_align_weight", 0.0)
    env = SchedulingEnv(problem.inst, quiet)
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
    """核对一份排程与问题定义、与仿真器是否一致（论文占位符 P-MILPCHK）。

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
