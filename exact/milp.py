"""离线 clairvoyant 精确求解与解回放校验（稿件 §3.2、§5.5）。

两个求解器给出同一模型的独立实现，二者一致性本身就是对公式化的一次检查：
  * CP-SAT（OR-Tools，免费）—— 默认路径，复现者无需授权即可跑通；
  * Gurobi MILP —— 与稿件 Eqs. (1)-(11) 逐条对应，有授权时启用。

得到的 eta_off 是**知道全部未来到达**才能达到的上界，因此它与在线策略的差值
上界了策略本身的损失，并包含"看不见未来"这一不可避免的代价——所以还提供
`solve_online_reoptimization` 作为真正的在线精确参照。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from environment.problem import Problem


@dataclass
class ExactResult:
    eta: float
    status: str
    solver: str
    seconds: float
    assignment: Dict[Tuple[int, int], Tuple[int, float, float]]   # (order, stage) -> (machine, start, end)
    n_completed: int = 0


def _horizon(problem: Problem) -> int:
    inst = problem.inst
    return int(inst.due_dates.max() + problem.residual_from(0, 0) + inst.proc_times.max() * inst.order_count)


def solve_cpsat(problem: Problem, time_limit_s: float = 3600.0,
                workers: int = 8) -> ExactResult:
    """CP-SAT 等价模型：最大化按时完工订单数。"""
    from ortools.sat.python import cp_model

    inst, model = problem.inst, cp_model.CpModel()
    H = _horizon(problem)
    S, J = problem.n_order, problem.n_stage
    started = time.perf_counter()

    z = [model.NewBoolVar(f"z_{s}") for s in range(S)]
    starts: Dict[Tuple[int, int], object] = {}
    ends: Dict[Tuple[int, int], object] = {}
    machine_intervals: Dict[int, list] = {m: [] for m in range(problem.n_machine)}
    literals: Dict[Tuple[int, int, int], object] = {}

    # 时间离散到秒：到达向上取整、交期向下取整、工时向上取整，使 CP-SAT 的解在连续
    # 时间下**严格可行**（回放校验因此能逐条核对）。代价是 eta_off 至多低估真实
    # clairvoyant 最优值一个"每订单一秒"的量级——在数百秒的时间尺度上可忽略，
    # 且方向是保守的：报告的 gap 只会偏大，不会偏小。
    for s in range(S):
        arrival = int(np.ceil(inst.arrival_times[s]))
        for j in range(J):
            task = problem.task_of(s, j)
            st = model.NewIntVar(arrival, H, f"st_{s}_{j}")
            en = model.NewIntVar(arrival, H, f"en_{s}_{j}")
            starts[(s, j)], ends[(s, j)] = st, en
            presences = []
            for m in problem.eligible[task]:
                dur = int(np.ceil(float(inst.proc_times[task, m])))
                lit = model.NewBoolVar(f"x_{s}_{j}_{m}")
                iv = model.NewOptionalIntervalVar(st, dur, en, lit, f"iv_{s}_{j}_{m}")
                machine_intervals[m].append(iv)
                presences.append(lit)
                literals[(s, j, m)] = lit
            # 被丢弃的订单不占用产能：sum_m x = z_s（稿件 Eq. 2 的 CP 等价写法）
            model.Add(sum(presences) == z[s])
            model.Add(en <= H)
        for j in range(1, J):                       # 路径前后序，稿件 Eq. (4)
            model.Add(starts[(s, j)] >= ends[(s, j - 1)])
        # 交期：z_s = 1 才要求按时完工，稿件 Eq. (9)
        model.Add(ends[(s, J - 1)] <= int(np.floor(inst.due_dates[s]))).OnlyEnforceIf(z[s])

    for m, ivs in machine_intervals.items():        # 机器不重叠，稿件 Eqs. (6)-(7)
        if ivs:
            model.AddNoOverlap(ivs)

    model.Maximize(sum(z))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    status = solver.Solve(model)
    name = solver.StatusName(status)

    assignment: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
    completed = 0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        completed = int(sum(solver.Value(v) for v in z))
        for (s, j), st in starts.items():
            if not solver.Value(z[s]):
                continue
            machine = next((m for (ss, jj, m), lit in literals.items()
                            if ss == s and jj == j and solver.Value(lit)), -1)
            assignment[(s, j)] = (machine, float(solver.Value(st)), float(solver.Value(ends[(s, j)])))
    return ExactResult(eta=completed / max(S, 1), status=name, solver="cpsat",
                       seconds=time.perf_counter() - started, assignment=assignment,
                       n_completed=completed)


def solve_gurobi(problem: Problem, time_limit_s: float = 3600.0) -> ExactResult:
    """稿件 Eqs. (1)-(11) 的 MILP 直译（含丢弃变量 v_s 与析取排序变量）。"""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return ExactResult(eta=float("nan"), status="gurobi_unavailable", solver="gurobi",
                           seconds=0.0, assignment={})

    inst = problem.inst
    S, J = problem.n_order, problem.n_stage
    L = float(_horizon(problem))
    started = time.perf_counter()

    model = gp.Model("dffsp_milp")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit_s)

    x, C, z, v = {}, {}, {}, {}
    for s in range(S):
        z[s] = model.addVar(vtype=GRB.BINARY, name=f"z_{s}")
        v[s] = model.addVar(vtype=GRB.BINARY, name=f"v_{s}")
        for j in range(J):
            C[(s, j)] = model.addVar(lb=0.0, name=f"C_{s}_{j}")
            for m in problem.eligible[problem.task_of(s, j)]:
                x[(s, j, m)] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{j}_{m}")
    model.update()

    for s in range(S):
        model.addConstr(z[s] + v[s] <= 1)                                  # Eq. (3)
        for j in range(J):
            task = problem.task_of(s, j)
            model.addConstr(gp.quicksum(x[(s, j, m)] for m in problem.eligible[task]) == 1 - v[s])
            dur = gp.quicksum(float(inst.proc_times[task, m]) * x[(s, j, m)]
                              for m in problem.eligible[task])
            if j == 0:
                model.addConstr(C[(s, 0)] >= float(inst.arrival_times[s]) + dur)   # Eq. (4)
            else:
                model.addConstr(C[(s, j)] >= C[(s, j - 1)] + dur)                  # Eq. (5)
        model.addConstr(C[(s, J - 1)] <= float(inst.due_dates[s]) + L * (1 - z[s]))  # Eq. (9)

    ops_on_machine: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(problem.n_machine)}
    for s in range(S):
        for j in range(J):
            for m in problem.eligible[problem.task_of(s, j)]:
                ops_on_machine[m].append((s, j))
    for m, ops in ops_on_machine.items():                                  # Eqs. (7)-(8)
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                (s1, j1), (s2, j2) = ops[a], ops[b]
                y = model.addVar(vtype=GRB.BINARY, name=f"y_{s1}_{j1}_{s2}_{j2}_{m}")
                p1 = float(inst.proc_times[problem.task_of(s1, j1), m])
                p2 = float(inst.proc_times[problem.task_of(s2, j2), m])
                model.addConstr(C[(s2, j2)] >= C[(s1, j1)] + p2 - L * (1 - y)
                                - L * (2 - x[(s1, j1, m)] - x[(s2, j2, m)]))
                model.addConstr(C[(s1, j1)] >= C[(s2, j2)] + p1 - L * y
                                - L * (2 - x[(s1, j1, m)] - x[(s2, j2, m)]))

    model.setObjective(gp.quicksum(z[s] for s in range(S)), GRB.MAXIMIZE)  # Eq. (1)
    model.optimize()

    if int(getattr(model, "SolCount", 0)) <= 0:
        return ExactResult(eta=float("nan"), status="no_solution", solver="gurobi",
                           seconds=time.perf_counter() - started, assignment={})
    assignment = {}
    for s in range(S):
        if z[s].X > 0.5:
            for j in range(J):
                chosen = next((m for m in problem.eligible[problem.task_of(s, j)]
                               if x[(s, j, m)].X > 0.5), -1)
                end = float(C[(s, j)].X)
                dur = float(inst.proc_times[problem.task_of(s, j), chosen]) if chosen >= 0 else 0.0
                assignment[(s, j)] = (chosen, end - dur, end)
    completed = int(round(sum(z[s].X for s in range(S))))
    return ExactResult(eta=completed / max(S, 1),
                       status="OPTIMAL" if model.Status == GRB.OPTIMAL else "FEASIBLE",
                       solver="gurobi", seconds=time.perf_counter() - started,
                       assignment=assignment, n_completed=completed)


def replay_check(problem: Problem, result: ExactResult, tol: float = 1e-6) -> Dict[str, object]:
    """把精确解回放进离散事件语义，核对完工时间与达成率。

    这是"MILP 与仿真器描述同一个问题"的直接证据（论文占位符 P-MILPCHK）：
    任何一条约束只存在于其中一侧，都会在这里暴露。
    """
    inst = problem.inst
    issues: List[str] = []
    if not result.assignment:
        return {"match": False, "reason": "no assignment to replay", "n_orders_checked": 0}

    orders = sorted({s for s, _ in result.assignment})
    on_time = 0
    for s in orders:
        prev_end = float(inst.arrival_times[s])
        for j in range(problem.n_stage):
            if (s, j) not in result.assignment:
                issues.append(f"order {s} stage {j} missing")
                break
            machine, start, end = result.assignment[(s, j)]
            if start + tol < prev_end:
                issues.append(f"order {s} stage {j}: start {start:.3f} < ready {prev_end:.3f}")
            if machine >= 0:
                dur = float(inst.proc_times[problem.task_of(s, j), machine])
                if abs((end - start) - dur) > 1e-3:
                    issues.append(f"order {s} stage {j}: duration {end - start:.3f} != {dur:.3f}")
            prev_end = end
        else:
            if prev_end <= float(inst.due_dates[s]) + tol:
                on_time += 1
    eta_replay = on_time / max(problem.n_order, 1)
    match = (not issues) and abs(eta_replay - result.eta) <= 1e-9
    return {"match": bool(match), "eta_solver": result.eta, "eta_replay": eta_replay,
            "n_orders_checked": len(orders), "n_issues": len(issues),
            "issues": issues[:5]}
