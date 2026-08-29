"""DFFSP-HFOI 离散事件环境（稿件 §4.1–4.6）。

一个决策时点 = 至少一台机器空闲且至少一道工序就绪。动作是 (task, machine, order_slot)。

本环境负责三件在稿件中被形式化的事：
  * 交期感知流体松弛的调用与缓存（Phi* 同时用于势函数塑形）；
  * 带紧急度安全网的剪枝（Prop 2），以及三种同基数对照剪枝（消融用）；
  * 到达不变的计数型奖励（Prop 3），其未折扣回报恒等于 eta - kappa_d * nu。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from data.generator import Instance
from environment.fluid import FluidRelaxation, FluidSolution, effective_slack
from environment.problem import Problem

NOT_ARRIVED, WAITING, IN_PROCESS, COMPLETED, DISCARDED = 0, 1, 2, 3, 4

# 主动空闲（no-op）动作的哨兵三元组。non-delay 调度类一般不含最优解，而全部规则
# 基线按构造都是 non-delay —— 主动空闲是学习策略能表达、规则无法表达的自由度。
NOOP = (-1, -1, -1)


def is_noop(action) -> bool:
    return int(action[0]) < 0


@dataclass
class StepStats:
    """逐决策点的诊断量，供 analysis/pruning.py 汇总（稿件 Table T-NEW-4）。"""
    n_feasible: List[int] = field(default_factory=list)
    n_pruned: List[int] = field(default_factory=list)
    fallback: int = 0
    singleton: int = 0
    critical_epochs: int = 0
    phi_star: List[float] = field(default_factory=list)
    support: List[int] = field(default_factory=list)
    t_fluid: float = 0.0
    t_obs: float = 0.0
    noop_offered: int = 0
    noop_used: int = 0


class SchedulingEnv:
    def __init__(self, inst: Instance, cfg) -> None:
        self.inst = inst
        self.cfg = cfg
        self.problem = Problem(inst)
        self.fluid = FluidRelaxation(cfg)

        self.kappa_d = float(cfg.get("reward.discard_weight", 1.0))
        self.beta_f = float(cfg.get("reward.fluid_align_weight", 0.0))
        self.beta_psi = float(cfg.get("reward.potential_weight", 0.0))
        self.gamma = float(cfg.get("reward.gamma", 1.0))

        self.eps_f = float(cfg.get("action_space.eps_f", 1e-5))
        self.theta_crit = float(cfg.get("action_space.theta_crit", 1.0))
        self.top_k = int(cfg.get("action_space.order_top_k", 5))
        self.pruning_mode = str(cfg.get("action_space.pruning_mode", "fluid"))
        self.slack_floor = float(cfg.get("fluid.slack_floor", 1.0))
        self.max_steps = int(cfg.get("episode.max_decision_steps", 200000))
        self.use_fluid_features = bool(cfg.get("variant.fluid_node_features", True))
        self.fluid_resolve_every = max(int(cfg.get("fluid.resolve_every", 1)), 1)
        self.allow_noop = bool(cfg.get("action_space.allow_noop", False))
        self.max_consecutive_noop = int(cfg.get("action_space.max_consecutive_noop", 3))
        self._rng = np.random.default_rng()

        self.reset()

    # ------------------------------------------------------------------ 生命周期
    def reset(self) -> None:
        p, inst = self.problem, self.inst
        self.now = float(inst.arrival_times.min()) if p.n_order else 0.0
        self.status = np.full(p.n_order, NOT_ARRIVED, dtype=np.int8)
        self.stage = np.zeros(p.n_order, dtype=np.int16)
        self.machine_free_at = np.zeros(p.n_machine, dtype=np.float64)
        self.machine_busy_with = np.full(p.n_machine, -1, dtype=np.int64)
        self.n_completed = 0
        self.n_discarded = 0
        self.step_count = 0
        self.done = False
        self.stats = StepStats()
        self._last_potential = 0.0
        self._fluid: FluidSolution | None = None
        self.machine_busy_time = np.zeros(p.n_machine, dtype=np.float64)
        self._consecutive_noop = 0
        self._fluid_age = 0                       # 距上次重解流体松弛的决策步数
        self._fluid_tasks: tuple = ()             # 上次求解时的活跃工序类型集
        # 无量纲化尺度：时间量除以交期跨度，加工时间除以最大工时
        self._time_scale = max(float(np.median(inst.due_dates - inst.arrival_times)), 1.0)
        self._proc_scale = max(float(inst.proc_times.max()), 1.0)
        self._advance_to_decision()
        self._last_potential = self._potential()

    # ------------------------------------------------------------------ 事件推进
    def _activate_arrivals(self) -> None:
        due = self.status == NOT_ARRIVED
        if due.any():
            arrived = due & (self.inst.arrival_times <= self.now + 1e-9)
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
                else:                                   # 超期完工按未达成计，并计入丢弃
                    self.n_discarded += 1
            else:
                self.status[order] = WAITING

    def _discard_hopeless(self) -> None:
        for order in np.nonzero(self.status == WAITING)[0]:
            if self.problem.is_hopeless(int(order), int(self.stage[order]), self.now):
                self.status[order] = DISCARDED
                self.n_discarded += 1

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
            if self._feasible_actions():
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

    # ------------------------------------------------------------------ 动作空间
    def _waiting_by_task(self) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        for order in np.nonzero(self.status == WAITING)[0]:
            task = self.problem.task_of(int(order), int(self.stage[order]))
            grouped.setdefault(task, []).append(int(order))
        # 每工序类型按有效松弛期升序 -> 紧急度排序的 top-K 槽（稿件 §4.5.1）
        for task, orders in grouped.items():
            orders.sort(key=lambda s: float(self.inst.due_dates[s]) - self.now)
        return grouped

    def _idle_machines(self) -> np.ndarray:
        return np.nonzero((self.machine_busy_with < 0) &
                          (self.machine_free_at <= self.now + 1e-9))[0]

    def _feasible_actions(self) -> List[Tuple[int, int, int]]:
        """A_feas：全部 (task, machine, order) 三元组。"""
        grouped = self._waiting_by_task()
        if not grouped:
            return []
        idle = set(int(m) for m in self._idle_machines())
        if not idle:
            return []
        actions: List[Tuple[int, int, int]] = []
        for task, orders in grouped.items():
            machines = [m for m in self.problem.eligible[task] if m in idle]
            if not machines:
                continue
            for order in orders[: self.top_k]:
                for m in machines:
                    actions.append((task, m, order))
        return actions

    def _is_critical(self, order: int, task: int) -> bool:
        """S_crit：剩余交期已不超过 theta_crit 倍剩余路径下界（稿件 Eq. 42）。"""
        stage = int(self.stage[order])
        lower = self.problem.residual_from(int(order), stage)
        return (float(self.inst.due_dates[order]) - self.now) <= self.theta_crit * lower

    def _solve_fluid(self) -> FluidSolution:
        grouped = self._waiting_by_task()
        # W_rj(t)：等在该阶段的 + 仍在其上游的（稿件 Eq. 16）
        workload: Dict[int, float] = {}
        slack: Dict[int, float] = {}
        for order in np.nonzero((self.status == WAITING) | (self.status == IN_PROCESS))[0]:
            r = int(self.inst.order_product[order])
            cur = int(self.stage[order])
            for j in range(cur, self.problem.n_stage):
                task = self.problem.task_index(r, j)
                workload[task] = workload.get(task, 0.0) + 1.0
                raw = float(self.inst.due_dates[order]) - self.now
                slack[task] = min(slack.get(task, np.inf), raw)
        if not workload:
            return FluidSolution(status="idle")
        slack_hat = {t: effective_slack(slack[t], self.problem.downstream_residual(t),
                                        self.slack_floor) for t in workload}
        # 摊销：流体松弛是**宏观**引导，单次派工不改变总体产能画像。在活跃工序类型集
        # 不变的前提下，最多每 resolve_every 个决策点重解一次（稿件 §4.9）。类型集变化
        # 时必须重解——否则新出现的类型在旧解里份额为 0，其动作会被整体剪掉。
        tasks = tuple(sorted(workload))
        if (self.fluid_resolve_every > 1 and self._fluid is not None
                and self._fluid.status == "optimal" and tasks == self._fluid_tasks
                and self._fluid_age < self.fluid_resolve_every - 1):
            self._fluid_age += 1
            return self._fluid
        key = self.fluid.trigger_key(workload, slack_hat)
        sol = self.fluid.solve(workload=workload, slack_hat=slack_hat,
                               rates=self.problem.rates, eligible=self.problem.eligible,
                               stage_pairs=self.problem.stage_pairs, key=key)
        self._fluid_age = 0
        self._fluid_tasks = tasks
        return sol

    def candidate_actions(self) -> Tuple[List[Tuple[int, int, int]], FluidSolution]:
        """A_f：带紧急度安全网的剪枝动作集（稿件 Eq. 43），并返回本次流体解。"""
        feasible = self._feasible_actions()
        if not feasible:
            return [], FluidSolution(status="empty")

        sol = self._solve_fluid() if self.pruning_mode == "fluid" or self.use_fluid_features \
            else FluidSolution(status="skipped")

        mode = self.pruning_mode
        if mode == "none":
            pruned = feasible
        elif mode == "fluid":
            pruned = [a for a in feasible
                      if sol.u(a[1], a[0]) > self.eps_f or self._is_critical(a[2], a[0])]
        else:
            # 同基数对照：先算流体剪枝会留下多少个，再用无流体信息的规则选同样多的
            target = max(1, sum(1 for a in feasible if sol.u(a[1], a[0]) > self.eps_f
                                or self._is_critical(a[2], a[0]))) if sol.alloc else max(1, len(feasible) // 2)
            target = min(target, len(feasible))
            if mode == "random":
                idx = self._rng.choice(len(feasible), size=target, replace=False)
                pruned = [feasible[i] for i in sorted(idx)]
            elif mode == "heuristic_load":
                order_key = sorted(range(len(feasible)),
                                   key=lambda i: self.machine_busy_time[feasible[i][1]])
                pruned = [feasible[i] for i in sorted(order_key[:target])]
            else:
                raise ValueError(f"unknown pruning_mode: {mode}")
            # 对照变体同样保留安全网，否则违反 Prop 2(c)、比较就不公平
            for a in feasible:
                if self._is_critical(a[2], a[0]) and a not in pruned:
                    pruned.append(a)

        self.stats.n_feasible.append(len(feasible))
        if any(self._is_critical(a[2], a[0]) for a in feasible):
            self.stats.critical_epochs += 1
        if not pruned:                                   # Prop 2(a) 活性回退
            pruned = feasible
            self.stats.fallback += 1
        self.stats.n_pruned.append(len(pruned))
        if len(pruned) == 1:
            self.stats.singleton += 1
        if sol.status == "optimal":
            self.stats.phi_star.append(sol.phi_star)
            self.stats.support.append(sol.support_size)
        self._fluid = sol
        # |A_f| 的统计只计派工动作，不含 no-op —— 保持与 Prop 2 中剪枝率定义一致
        if self._noop_available():
            self.stats.noop_offered += 1
            pruned = pruned + [NOOP]
        return pruned, sol

    def _noop_available(self) -> bool:
        """主动空闲的防死锁条件（稿件 §4.5.3）。

        两条都必须成立，否则 no-op 可能让系统永久停摆：
          (i) 存在未来事件（尚有订单未到达，或有机器在忙），
              否则时间无法推进，状态不会改变；
          (ii) 连续 no-op 次数未超上限，避免策略在同一事件间隔内反复空转。
        条件 (i) 保证 Prop 2(a) 的活性论证在扩展动作集上仍然成立：
        no-op 之后必然抵达一个严格更晚的事件时刻。
        """
        if not self.allow_noop:
            return False
        if self._consecutive_noop >= self.max_consecutive_noop:
            return False
        nxt = self._next_event_time()
        return nxt is not None and nxt > self.now + 1e-9

    # ------------------------------------------------------------------ 奖励
    def _potential(self) -> float:
        """Psi(omega) = min{Phi*, 1}，势函数塑形用（稿件 Eq. 50）。"""
        if self.beta_psi <= 0:
            return 0.0
        sol = self._fluid if self._fluid is not None else self._solve_fluid()
        return float(min(max(sol.phi_star, 0.0), 1.0)) if sol.status == "optimal" else 0.0

    def step(self, action: Tuple[int, int, int]) -> Tuple[float, bool, dict]:
        if is_noop(action):
            return self._step_noop()
        self._consecutive_noop = 0
        task, machine, order = int(action[0]), int(action[1]), int(action[2])
        if self.status[order] != WAITING:
            raise ValueError(f"order {order} is not waiting (status={self.status[order]})")
        if self.problem.rates[task, machine] <= 0:
            raise ValueError(f"machine {machine} cannot process task {task}")

        before_c, before_d = self.n_completed, self.n_discarded
        proc = float(self.inst.proc_times[task, machine])
        self.status[order] = IN_PROCESS
        self.machine_busy_with[machine] = order
        self.machine_free_at[machine] = self.now + proc
        self.machine_busy_time[machine] += proc
        self.step_count += 1

        self._advance_to_decision()

        # 到达不变的计数型奖励（稿件 Eq. 46）：只对完工/丢弃事件可测，与到达无关
        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        reward = (d_c - self.kappa_d * d_d) / max(self.problem.n_order, 1)

        info = {"base_reward": reward, "d_completed": d_c, "d_discarded": d_d}

        if self.beta_psi > 0:                            # 势函数塑形，策略不变
            psi_next = self._potential()
            shaping = self.gamma * psi_next - self._last_potential
            self._last_potential = psi_next
            reward += self.beta_psi * shaping
            info["shaping"] = shaping
        if self.beta_f > 0 and self._fluid is not None:  # 非势函数对齐项（仅敏感性研究用）
            denom = sum(v for (m, _), v in self._fluid.alloc.items() if m == machine) + 1e-9
            reward += self.beta_f * self._fluid.u(machine, task) / denom

        return reward, self.done, info

    def _step_noop(self) -> Tuple[float, bool, dict]:
        """主动空闲：本时刻不派工，直接推进到下一事件。

        奖励用与派工动作**完全相同**的计数式，故 Prop 3 的望远镜恒等式
        sum_t r_t = eta - kappa_d * nu 在扩展动作集上逐项不变：空闲期间若有订单
        变为 hopeless，代价会自动通过 d_d 计入，无需额外惩罚项。
        """
        before_c, before_d = self.n_completed, self.n_discarded
        self._consecutive_noop += 1
        self.stats.noop_used += 1
        self.step_count += 1

        nxt = self._next_event_time()
        if nxt is None or nxt <= self.now + 1e-9:        # 防死锁条件已排除，稳妥起见再兜一层
            self.done = True
        else:
            self.now = nxt
            self._advance_to_decision()

        d_c = self.n_completed - before_c
        d_d = self.n_discarded - before_d
        reward = (d_c - self.kappa_d * d_d) / max(self.problem.n_order, 1)
        info = {"base_reward": reward, "d_completed": d_c, "d_discarded": d_d, "noop": True}

        if self.beta_psi > 0:
            psi_next = self._potential()
            shaping = self.gamma * psi_next - self._last_potential
            self._last_potential = psi_next
            reward += self.beta_psi * shaping
            info["shaping"] = shaping
        return reward, self.done, info

    # ------------------------------------------------------------------ 观测
    def observation(self, actions, sol: FluidSolution) -> dict:
        """异构图状态 + 候选动作特征（稿件 Eqs. 27-29, 38）。

        节点是**工序类型**而非订单级工序，故图规模固定为 |O| x |M|，与订单数无关
        （稿件 §4.3 的尺度不变性）——这正是 OOD 档不重训即可评测的前提。

        全部特征在此处无量纲化：时间量除以交期尺度、计数量除以订单总数。
        否则 slack(~1e2) 与 rate(~7e-3) 相差四个数量级，线性层无法同时利用两者。
        """
        import time as _time
        t0 = _time.perf_counter()
        p = self.problem
        grouped = self._waiting_by_task()
        idle = set(int(m) for m in self._idle_machines())
        use_fluid = self.use_fluid_features and sol.status == "optimal"
        n_order = max(p.n_order, 1)
        t_scale = self._time_scale
        horizon = max(float(self.inst.due_dates.max()), 1.0)

        op = np.zeros((p.n_task, 12), dtype=np.float32)
        active = np.nonzero((self.status == WAITING) | (self.status == IN_PROCESS))[0]
        active_product = self.inst.order_product[active] if active.size else np.empty(0, dtype=int)
        active_stage = self.stage[active] if active.size else np.empty(0, dtype=int)
        for task in range(p.n_task):
            elig = p.eligible[task]
            orders = grouped.get(task, [])
            slacks = np.asarray([(float(self.inst.due_dates[o]) - self.now) / t_scale
                                 for o in orders], dtype=np.float32) if orders \
                else np.zeros(1, dtype=np.float32)
            r_idx, j_idx = divmod(task, p.n_stage)
            w_all = float(np.count_nonzero((active_product == r_idx) & (active_stage <= j_idx))) \
                if active.size else 0.0
            m_fluid = sum(1 for m in elig if use_fluid and sol.u(m, task) > self.eps_f)
            op[task] = [
                len(elig) / max(p.n_machine, 1),
                sum(1 for m in elig if m in idle) / max(len(elig), 1),
                float(slacks.mean()), float(slacks.min()), float(slacks.max()), float(slacks.std()),
                w_all / n_order, len(orders) / n_order,
                m_fluid / max(len(elig), 1),
                float(sol.rate[task]) * t_scale if (use_fluid and sol.rate is not None) else 0.0,
                r_idx / max(p.n_product - 1, 1), j_idx / max(p.n_stage - 1, 1),
            ]

        ma = np.zeros((p.n_machine, 5), dtype=np.float32)
        for m in range(p.n_machine):
            tasks_m = [t for t in range(p.n_task) if p.rates[t, m] > 0]
            o_fluid = sum(1 for t in tasks_m if use_fluid and sol.u(m, t) > self.eps_f)
            xi = sum(sol.u(m, t) for t in tasks_m) if use_fluid else 0.0
            ma[m] = [len(tasks_m) / max(p.n_task, 1),
                     float(self.machine_free_at[m]) / horizon,
                     1.0 if m in idle else 0.0,
                     o_fluid / max(len(tasks_m), 1), float(xi)]

        # 动作级特征：松弛期 / 归一化处理时间 / 流体份额 / 临界比 / no-op 标志。
        # 临界比 = 剩余松弛期 / 剩余路径最小加工时间，是交期目标下判别力最强的单一信号；
        # <1 表示该订单已不可能按时交付。
        # 第 5 列区分主动空闲：其余四列对 no-op 行置零，避免哨兵下标 (-1,-1) 携带
        # 任意节点嵌入引入伪方差；网络侧再据该标志位屏蔽节点特征（见 networks.py）。
        act = np.zeros((len(actions), 5), dtype=np.float32)
        wait_gap = 0.0
        nxt = self._next_event_time()
        if nxt is not None and nxt > self.now:
            wait_gap = min((nxt - self.now) / t_scale, 10.0)
        for i, a in enumerate(actions):
            if is_noop(a):
                # no-op 行只带"要等多久"这一个量，放在处理时间那一列（语义一致：
                # 本动作要消耗多少时间），其余置零、标志位置 1
                act[i] = [0.0, wait_gap, 0.0, 0.0, 1.0]
                continue
            task, machine, order = int(a[0]), int(a[1]), int(a[2])
            slack = float(self.inst.due_dates[order]) - self.now
            need = max(p.residual_from(order, int(self.stage[order])), 1e-9)
            act[i] = [slack / t_scale,
                      float(self.inst.proc_times[task, machine]) / self._proc_scale,
                      float(sol.u(machine, task)) if use_fluid else 0.0,
                      min(slack / need, 10.0), 0.0]

        arrived = float(np.count_nonzero(self.status != NOT_ARRIVED))
        settled = float(self.n_completed + self.n_discarded)
        global_feat = np.asarray([
            self.eta,                                   # 已按时完工比例
            1.0 - arrived / n_order,                    # 未到达订单比例（旧观测完全看不到）
            1.0 - settled / n_order,                    # 尚未结清的订单比例
            min(self.now / horizon, 1.0),               # 归一化时间进度
            self.nu,                                    # 丢弃率
        ], dtype=np.float32)

        self.stats.t_obs += _time.perf_counter() - t0
        return {
            "op": op, "ma": ma, "proc_rate": p.rates,
            "adj": (p.rates > 0),
            "act_feat": act,
            # no-op 用下标 (0,0) 占位；其节点特征在 networks.py 中按标志位屏蔽
            "act_index": np.asarray([(max(int(a[0]), 0), max(int(a[1]), 0)) for a in actions],
                                    dtype=np.int64),
            "eta_t": np.float32(self.eta),
            "global_feat": global_feat,
        }

    # ------------------------------------------------------------------ 指标
    @property
    def eta(self) -> float:
        return self.n_completed / max(self.problem.n_order, 1)

    @property
    def nu(self) -> float:
        return self.n_discarded / max(self.problem.n_order, 1)
