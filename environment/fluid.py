"""交期感知流体松弛（稿件 §3.3 / Eqs. 19-25）。

相对旧实现的三处规格级改动（docs/experiment-spec.md §2）：
  F1  约束含有效松弛期 delta_hat —— 目标从"归一化处理速率"变为"交期可行比"；
  F2  增加阶段间流平衡约束；
  F3  返回 Phi*（可行性证书，供势函数塑形）与基本解支撑集大小（Prop 1(d) 的实证）。

求解器：优先 Gurobi（若已授权），否则回退到 SciPy 的 HiGHS —— 二者对本 LP 等价，
后者无需授权，保证复现者开箱即用。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

_GUROBI = None
_GUROBI_TRIED = False


def _try_gurobi():
    global _GUROBI, _GUROBI_TRIED
    if not _GUROBI_TRIED:
        _GUROBI_TRIED = True
        try:
            import gurobipy as gp  # noqa: F401
            gp.Model("probe").dispose()
            _GUROBI = gp
        except Exception:
            _GUROBI = None
    return _GUROBI


@dataclass
class FluidSolution:
    """一次流体求解的完整结果。"""
    alloc: Dict[Tuple[int, int], float] = field(default_factory=dict)  # (machine, task) -> u*
    phi_star: float = 0.0                                             # Phi*(t)，可行性证书
    rate: np.ndarray | None = None                                    # 每工序类型的 lambda^f
    support_size: int = 0                                             # 严格为正的分量数
    solver: str = "none"
    status: str = "empty"
    solve_seconds: float = 0.0

    def u(self, machine: int, task: int) -> float:
        return self.alloc.get((int(machine), int(task)), 0.0)


@dataclass
class FluidStats:
    solve_count: int = 0
    cache_hit_count: int = 0
    fallback_count: int = 0
    solve_seconds: float = 0.0

    @property
    def zeta(self) -> float:
        """缓存未命中率：稿件 §4.9 中摊销 LP 成本的系数。"""
        total = self.solve_count + self.cache_hit_count
        return self.solve_count / total if total else 0.0


def effective_slack(deadline_slack: float, residual_downstream: float, slack_floor: float) -> float:
    """delta_hat = max(delta - sum_{j'>j} min_m p, delta_min)，稿件 Eq. (18)。"""
    return max(float(deadline_slack) - float(residual_downstream), float(slack_floor))


class FluidRelaxation:
    """事件触发 + 缓存 + 热启动的流体松弛求解器（稿件 §3.3.1）。"""

    def __init__(self, cfg) -> None:
        self.mode = str(cfg.get("fluid.mode", "throughput"))
        self.enabled = bool(cfg.get("fluid.enabled", True)) and self.mode != "off"
        self.slack_floor = float(cfg.get("fluid.slack_floor", 1.0))
        self.use_flow_balance = bool(cfg.get("fluid.use_flow_balance", True))
        self.time_limit = float(cfg.get("fluid.time_limit_s", 2.0))
        self.slack_bucket = float(cfg.get("fluid.slack_bucket_s", 60.0))
        self.cache_enabled = bool(cfg.get("fluid.cache_enabled", True))
        self.stats = FluidStats()
        self._cache: Dict[tuple, FluidSolution] = {}

    # ---------------------------------------------------------------- 触发键
    def trigger_key(self, workload: Dict[int, float], slack: Dict[int, float],
                    idle_machines: Sequence[int] | None = None) -> tuple:
        """chi(t)，稿件 Eq. (26)：活跃类型集 + 取整负载 + 离散化松弛期。

        **不含空闲机器集**。LP 的机器容量约束 sum_task u[m, task] <= 1 是对全部出现在
        合格对中的机器施加的，与"此刻哪些机器空闲"无关——`_solve_impl` 根本不接收该
        参数，解对它恒不变。把它放进缓存键，只会让每次派工都使缓存失效：实测原实现
        zeta = 0.997（命中率 0.3%），LP 几乎每个决策点重解一次，占 rollout 用时的一半，
        而稿件 §4.9 的摊销成本论证正建立在 zeta 较小之上。参数保留只为兼容旧调用。
        """
        bucket = max(self.slack_bucket, 1e-9)
        tasks = sorted(workload)
        return (
            tuple(tasks),
            tuple(int(np.ceil(workload[t])) for t in tasks),
            tuple(int(np.floor(slack.get(t, 0.0) / bucket)) for t in tasks),
        )

    # ---------------------------------------------------------------- 主入口
    def solve(self, *, workload: Dict[int, float], slack_hat: Dict[int, float],
              rates: np.ndarray, eligible: Dict[int, List[int]],
              stage_pairs: Sequence[Tuple[int, int]],
              key: tuple | None = None) -> FluidSolution:
        """求解一次流体松弛。

        workload      : task -> W_rj(t)，只含 W>0 的活跃工序类型
        slack_hat     : task -> delta_hat_rj(t)（已扣除下游剩余路径下界）
        rates         : [task, machine] 处理速率 mu = 1/p，不可加工处为 0
        eligible      : task -> 合格机器列表
        stage_pairs   : [(prev_task, next_task)]，两端负载均为正的相邻阶段对
        """
        if not self.enabled or not workload:
            return FluidSolution(status="disabled")

        if self.cache_enabled and key is not None and key in self._cache:
            self.stats.cache_hit_count += 1
            return self._cache[key]

        started = time.perf_counter()
        self.stats.solve_count += 1
        sol = self._solve_impl(workload, slack_hat, rates, eligible, stage_pairs)
        sol.solve_seconds = time.perf_counter() - started
        self.stats.solve_seconds += sol.solve_seconds
        if sol.status not in ("optimal",):
            self.stats.fallback_count += 1
        if self.cache_enabled and key is not None:
            self._cache[key] = sol
        return sol

    # ------------------------------------------------------------ LP 构造/求解
    def _solve_impl(self, workload, slack_hat, rates, eligible, stage_pairs) -> FluidSolution:
        if self.mode == "throughput":
            return self._solve_throughput(workload, slack_hat, rates, eligible, stage_pairs)
        tasks = sorted(workload)
        pairs: List[Tuple[int, int]] = []
        for task in tasks:
            for machine in eligible.get(task, []):
                if rates[task, machine] > 0:
                    pairs.append((int(machine), int(task)))
        if not pairs:
            return FluidSolution(status="no_eligible_pair")

        var_index = {p: i for i, p in enumerate(pairs)}
        n_u = len(pairs)
        n_var = n_u + 1                       # 最后一维是 Phi
        phi_col = n_u

        c = np.zeros(n_var)
        c[phi_col] = -1.0                     # linprog 求最小，故 min(-Phi) == max(Phi)

        rows: List[np.ndarray] = []
        rhs: List[float] = []

        # (1) 交期可行比约束：-delta_hat * sum(mu u) + W * Phi <= 0
        workload_only = (self.mode == "workload_only")
        for task in tasks:
            row = np.zeros(n_var)
            scale = 1.0 if workload_only else float(slack_hat.get(task, self.slack_floor))
            for machine in eligible.get(task, []):
                if (machine, task) in var_index:
                    row[var_index[(machine, task)]] = -scale * float(rates[task, machine])
            row[phi_col] = float(workload[task])
            rows.append(row)
            rhs.append(0.0)

        # (2) 机器容量约束：sum_rj u_m,rj <= 1
        for machine in sorted({m for m, _ in pairs}):
            row = np.zeros(n_var)
            for (cand_m, cand_t), idx in var_index.items():
                if cand_m == machine:
                    row[idx] = 1.0
            rows.append(row)
            rhs.append(1.0)

        # (3) 阶段间流平衡：-W_j * sum(mu_{j-1} u_{j-1}) + W_{j-1} * sum(mu_j u_j) <= 0
        if self.use_flow_balance:
            for prev_task, next_task in stage_pairs:
                if prev_task not in workload or next_task not in workload:
                    continue
                row = np.zeros(n_var)
                for machine in eligible.get(prev_task, []):
                    if (machine, prev_task) in var_index:
                        row[var_index[(machine, prev_task)]] -= (
                            float(workload[next_task]) * float(rates[prev_task, machine]))
                for machine in eligible.get(next_task, []):
                    if (machine, next_task) in var_index:
                        row[var_index[(machine, next_task)]] += (
                            float(workload[prev_task]) * float(rates[next_task, machine]))
                rows.append(row)
                rhs.append(0.0)

        a_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)
        bounds = [(0.0, 1.0)] * n_u + [(0.0, None)]

        gp = _try_gurobi()
        if gp is not None:
            sol = self._solve_gurobi(gp, c, a_ub, b_ub, bounds, pairs, phi_col)
            if sol is not None:
                self._finalize(sol, pairs, tasks, rates)
                return sol

        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs",
                      options={"time_limit": self.time_limit} if self.time_limit > 0 else None)
        if not res.success or res.x is None:
            return FluidSolution(status=f"highs_{res.status}", solver="highs")
        sol = FluidSolution(
            alloc={p: float(max(res.x[var_index[p]], 0.0)) for p in pairs},
            phi_star=float(max(res.x[phi_col], 0.0)),
            solver="highs", status="optimal",
        )
        self._finalize(sol, pairs, tasks, rates)
        return sol

    def _solve_throughput(self, workload, slack_hat, rates, eligible, stage_pairs) -> FluidSolution:
        """交期内吞吐最大化：max sum_rj min(delta_hat_rj * lambda_rj, W_rj)。

        为什么换掉 max-min：max-min 交期可行比是**均衡型**目标，它把产能摊平到所有工序
        类型上；而按时完工件数在过载下要的是**集中与选择性放弃**——把产能给还救得回来的
        工作，放弃已经救不回来的。两者方向相反，实测 max-min 引导比 SPT 还差 0.058。
        本目标与 eta 直接同向：它就是"流体意义下能按时交付多少件"。

        分段线性凹函数用辅助变量线性化：y_rj <= delta_hat * lambda_rj 且 y_rj <= W_rj。
        证书 Phi* 定义为 sum(y)/sum(W)，即可按时交付的工作量占比，仍落在 [0, 1]，
        因而可直接用作势函数塑形的势（无需再截断）。
        """
        tasks = sorted(workload)
        pairs: List[Tuple[int, int]] = []
        for task in tasks:
            for machine in eligible.get(task, []):
                if rates[task, machine] > 0:
                    pairs.append((int(machine), int(task)))
        if not pairs:
            return FluidSolution(status="no_eligible_pair")

        var_index = {p: i for i, p in enumerate(pairs)}
        n_u = len(pairs)
        y_index = {task: n_u + k for k, task in enumerate(tasks)}
        n_var = n_u + len(tasks)

        c = np.zeros(n_var)
        for task in tasks:
            c[y_index[task]] = -1.0                    # min(-sum y) == max(sum y)

        rows: List[np.ndarray] = []
        rhs: List[float] = []

        # y_rj - delta_hat * sum_m mu u <= 0
        for task in tasks:
            row = np.zeros(n_var)
            scale = float(slack_hat.get(task, self.slack_floor))
            for machine in eligible.get(task, []):
                if (machine, task) in var_index:
                    row[var_index[(machine, task)]] = -scale * float(rates[task, machine])
            row[y_index[task]] = 1.0
            rows.append(row)
            rhs.append(0.0)

        # 机器容量
        for machine in sorted({m for m, _ in pairs}):
            row = np.zeros(n_var)
            for (cand_m, _), idx in var_index.items():
                if cand_m == machine:
                    row[idx] = 1.0
            rows.append(row)
            rhs.append(1.0)

        # 阶段间流平衡（与 max-min 版本相同）
        if self.use_flow_balance:
            for prev_task, next_task in stage_pairs:
                if prev_task not in workload or next_task not in workload:
                    continue
                row = np.zeros(n_var)
                for machine in eligible.get(prev_task, []):
                    if (machine, prev_task) in var_index:
                        row[var_index[(machine, prev_task)]] -= (
                            float(workload[next_task]) * float(rates[prev_task, machine]))
                for machine in eligible.get(next_task, []):
                    if (machine, next_task) in var_index:
                        row[var_index[(machine, next_task)]] += (
                            float(workload[prev_task]) * float(rates[next_task, machine]))
                rows.append(row)
                rhs.append(0.0)

        a_ub = np.vstack(rows)
        b_ub = np.asarray(rhs, dtype=float)
        # y_rj <= W_rj 直接写进变量上界，不必再占一行约束
        bounds = [(0.0, 1.0)] * n_u + [(0.0, float(workload[t])) for t in tasks]

        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs",
                      options={"time_limit": self.time_limit} if self.time_limit > 0 else None)
        if not res.success or res.x is None:
            return FluidSolution(status=f"highs_{res.status}", solver="highs")

        total_w = sum(float(workload[t]) for t in tasks)
        throughput = float(sum(max(res.x[y_index[t]], 0.0) for t in tasks))
        sol = FluidSolution(
            alloc={p: float(max(res.x[var_index[p]], 0.0)) for p in pairs},
            phi_star=throughput / total_w if total_w > 0 else 0.0,
            solver="highs", status="optimal",
        )
        self._finalize(sol, pairs, tasks, rates)
        return sol

    @staticmethod
    def _solve_gurobi(gp, c, a_ub, b_ub, bounds, pairs, phi_col):
        try:
            model = gp.Model("fluid_lp")
            model.Params.OutputFlag = 0
            xs = [model.addVar(lb=lo, ub=(gp.GRB.INFINITY if hi is None else hi))
                  for lo, hi in bounds]
            model.update()
            for row, rhs in zip(a_ub, b_ub):
                nz = np.nonzero(row)[0]
                if nz.size:
                    model.addConstr(gp.quicksum(float(row[i]) * xs[i] for i in nz) <= float(rhs))
            model.setObjective(gp.quicksum(float(c[i]) * xs[i] for i in np.nonzero(c)[0]),
                               gp.GRB.MINIMIZE)
            model.optimize()
            if int(getattr(model, "SolCount", 0)) <= 0:
                return None
            values = [float(v.X) for v in xs]
            return FluidSolution(
                alloc={p: max(values[i], 0.0) for i, p in enumerate(pairs)},
                phi_star=max(values[phi_col], 0.0), solver="gurobi", status="optimal")
        except Exception:
            return None

    @staticmethod
    def _finalize(sol: FluidSolution, pairs, tasks, rates) -> None:
        eps = 1e-9
        sol.support_size = sum(1 for v in sol.alloc.values() if v > eps)
        n_task = int(rates.shape[0])
        rate_vec = np.zeros(n_task, dtype=np.float32)
        for (machine, task), value in sol.alloc.items():
            rate_vec[task] += float(rates[task, machine]) * value
        sol.rate = rate_vec
