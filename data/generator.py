"""算例生成：训练算例每周期随机构造；评测算例按设计单元逐个物化。

单算例文件采用长表 CSV：`kind` 列区分 machine / task / order 三类记录，
读取时还原成 InstanceSpec。落盘只用 CSV，不用二进制格式。
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 算例数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    instance_id: str
    tier: str
    product_count: int
    stage_count: int
    machines_per_stage: Tuple[int, ...]
    proc_times: np.ndarray          # [task, machine]，不可加工处为 0
    order_product: np.ndarray       # [S]
    arrival_times: np.ndarray       # [S]
    due_dates: np.ndarray           # [S]
    meta: Dict[str, float]

    @property
    def machine_count(self) -> int:
        return int(sum(self.machines_per_stage))

    @property
    def task_count(self) -> int:
        return self.product_count * self.stage_count

    @property
    def order_count(self) -> int:
        return int(len(self.order_product))

    def task_index(self, product: int, stage: int) -> int:
        return product * self.stage_count + stage

    def stage_machine_slice(self, stage: int) -> Tuple[int, int]:
        start = int(sum(self.machines_per_stage[:stage]))
        return start, start + int(self.machines_per_stage[stage])


# --------------------------------------------------------------------------- #
# 负荷指标（稿件 Eqs. 12-14）——每个算例族都必须落盘
# --------------------------------------------------------------------------- #
def load_metrics(inst: Instance) -> Dict[str, float]:
    """返回 Lambda（到达率）、W_bar（单订单期望总工作量）、rho_sys（系统负荷）、iota（插单强度）。"""
    arrivals = np.sort(inst.arrival_times)
    gaps = np.diff(arrivals)
    mean_gap = float(gaps.mean()) if gaps.size else float("inf")
    lam = 1.0 / mean_gap if mean_gap > 0 else 0.0

    # 每型每阶段在合格机器上的平均加工时间
    per_product_work = np.zeros(inst.product_count, dtype=np.float64)
    per_stage_times: List[float] = []
    for r in range(inst.product_count):
        for j in range(inst.stage_count):
            lo, hi = inst.stage_machine_slice(j)
            row = inst.proc_times[inst.task_index(r, j), lo:hi]
            eligible = row[row > 0]
            mean_p = float(eligible.mean()) if eligible.size else 0.0
            per_product_work[r] += mean_p
            per_stage_times.append(mean_p)

    counts = np.bincount(inst.order_product, minlength=inst.product_count).astype(np.float64)
    pi_r = counts / max(counts.sum(), 1.0)
    w_bar = float((pi_r * per_product_work).sum())
    p_bar = float(np.mean(per_stage_times)) if per_stage_times else 0.0

    rho = lam * w_bar / max(inst.machine_count, 1)
    iota = lam * p_bar
    return {
        "E_dt": mean_gap,
        "Lambda": lam,
        "W_bar": w_bar,
        "p_bar": p_bar,
        "rho_sys": rho,
        "iota": iota,
        "regime": "overloaded" if rho >= 1.0 else ("high_frequency" if iota >= 1.0 else "moderate"),
    }


# --------------------------------------------------------------------------- #
# 到达过程（稿件 §5.8：确定性 / Poisson / MMPP 突发，同一平均速率）
# --------------------------------------------------------------------------- #
def sample_arrivals(rng: np.random.Generator, count: int, mean_gap: float,
                    process: str = "poisson") -> np.ndarray:
    process = process.lower()
    if process == "deterministic":
        gaps = np.full(count, mean_gap, dtype=np.float64)
    elif process == "poisson":
        gaps = rng.exponential(mean_gap, size=count)
    elif process == "mmpp":
        # 两相 Markov 调制突发过程：突发相的间隔是安静相的 1/10。
        p_enter, p_stay = 0.30, 0.75          # 稳态突发概率 = p_enter / (p_enter + 1 - p_stay)
        burst_scale = 0.1
        gaps = np.empty(count, dtype=np.float64)
        in_burst = False
        for i in range(count):
            in_burst = (rng.random() < (p_stay if in_burst else p_enter))
            gaps[i] = rng.exponential(burst_scale if in_burst else 1.0)
    elif process == "uniform":
        gaps = rng.uniform(0.0, 2.0, size=count)
    else:
        raise ValueError(f"unknown arrival process: {process}")

    # 三种过程按构造共享同一平均到达率：把间隔整体缩放到样本均值恰为 mean_gap。
    # 这样 arrival 档内不同过程之间的差异只反映突发性，不混入负荷差异
    # （稿件 §5.8 的到达过程对照正是建立在这一点上）。
    gaps = np.maximum(gaps, 1e-9)
    gaps = gaps * (mean_gap / float(gaps.mean()))
    return np.cumsum(gaps)


# --------------------------------------------------------------------------- #
# 算例构造
# --------------------------------------------------------------------------- #
def build_instance(rng: np.random.Generator, *, instance_id: str, tier: str,
                   product_count: int, stage_count: int, machines_per_stage: Sequence[int],
                   order_count: int, proc_time_range: Sequence[float],
                   ddt: float, mean_interarrival: float,
                   arrival_process: str = "poisson",
                   eligibility_prob: float = 1.0,
                   ddt_spread: Sequence[float] = (1.0, 1.0)) -> Instance:
    machines_per_stage = tuple(int(m) for m in machines_per_stage)
    machine_count = int(sum(machines_per_stage))
    lo_p, hi_p = float(proc_time_range[0]), float(proc_time_range[1])

    proc = np.zeros((product_count * stage_count, machine_count), dtype=np.float32)
    for r in range(product_count):
        for j in range(stage_count):
            start = int(sum(machines_per_stage[:j]))
            end = start + machines_per_stage[j]
            row = rng.integers(int(lo_p), int(hi_p) + 1, size=end - start).astype(np.float32)
            if eligibility_prob < 1.0 and (end - start) > 1:
                keep = rng.random(end - start) < eligibility_prob
                if not keep.any():
                    keep[rng.integers(0, end - start)] = True
                row = np.where(keep, row, 0.0).astype(np.float32)
            proc[r * stage_count + j, start:end] = row

    order_product = rng.integers(0, product_count, size=order_count)
    arrivals = sample_arrivals(rng, order_count, mean_interarrival, arrival_process)
    # 逐单独立抽交期宽松度。若所有订单共用同一常数 DDT，则 due - arrival 恒定，
    # EDD 与 FIFO 完全等价、且与环境构造候选集时的排序键重合 —— 订单维度
    # 在决策上不携带任何信息。乘性扰动是让 EDD 与 FIFO 分离的最小改动。
    lo_d, hi_d = float(ddt_spread[0]), float(ddt_spread[1])
    slack = float(ddt) * (rng.uniform(lo_d, hi_d, size=order_count) if hi_d > lo_d else lo_d)
    due = arrivals + slack

    inst = Instance(
        instance_id=instance_id, tier=tier,
        product_count=product_count, stage_count=stage_count,
        machines_per_stage=machines_per_stage, proc_times=proc,
        order_product=order_product, arrival_times=arrivals, due_dates=due,
        meta={"DDT": float(ddt), "mean_interarrival": float(mean_interarrival),
              "arrival_process": arrival_process,
              "ddt_spread_lo": lo_d, "ddt_spread_hi": hi_d},
    )
    inst.meta.update(load_metrics(inst))
    return inst


def sample_training_instance(rng: np.random.Generator, param_table: Dict) -> Instance:
    """训练算例：每个周期按参数表现场随机构造，不预生成、不落盘。

    到达强度按**系统负荷 rho_sys 均匀抽样**再反推到达间隔，而不是对到达间隔均匀
    抽样。后者因 rho ∝ 1/gap 而极度右偏：实测 82% 的训练算力落在评测从不覆盖的
    工况上，且 31% 的训练算例随机策略即 eta=1.0（零梯度）、44% 落在 eta<0.2
    （同样无区分度）。按 rho 分层可把算力集中到有梯度的带内。
    """
    lo_s, hi_s = param_table["order_count_range"]
    lo_ddt, hi_ddt = param_table["ddt_range"]
    stage_count = int(param_table["stage_count"])
    machines_per_stage = int(param_table["machines_per_stage"])
    proc_range = param_table["proc_time_range"]

    rho_range = param_table.get("rho_range")
    if rho_range is not None:
        rho = float(rng.uniform(float(rho_range[0]), float(rho_range[1])))
        mean_p = (float(proc_range[0]) + float(proc_range[1])) / 2.0
        gap = (stage_count * mean_p) / (rho * stage_count * machines_per_stage)
    else:                                   # 兼容旧参数表
        lo_dt, hi_dt = param_table["interarrival_range"]
        gap = float(rng.integers(int(lo_dt), int(hi_dt) + 1))

    return build_instance(
        rng,
        instance_id="train_random", tier="train",
        product_count=int(param_table["product_count"]),
        stage_count=stage_count,
        machines_per_stage=[machines_per_stage] * stage_count,
        order_count=int(rng.integers(lo_s, hi_s + 1)),
        proc_time_range=proc_range,
        ddt=float(rng.integers(int(lo_ddt), int(hi_ddt) + 1)),
        mean_interarrival=gap,
        ddt_spread=param_table.get("ddt_spread", (1.0, 1.0)),
        arrival_process="poisson",
    )


# --------------------------------------------------------------------------- #
# 单算例 CSV 读写（长表）
# --------------------------------------------------------------------------- #
def save_instance_csv(inst: Instance, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "i", "j", "value", "extra"])
        writer.writerow(["header", inst.product_count, inst.stage_count, inst.machine_count, inst.order_count])
        for j, m in enumerate(inst.machines_per_stage):
            writer.writerow(["stage_machines", j, "", m, ""])
        for t in range(inst.task_count):
            for m in range(inst.machine_count):
                value = float(inst.proc_times[t, m])
                if value > 0:
                    writer.writerow(["proc", t, m, value, ""])
        for s in range(inst.order_count):
            writer.writerow(["order", s, int(inst.order_product[s]),
                             float(inst.arrival_times[s]), float(inst.due_dates[s])])
        for key, value in inst.meta.items():
            writer.writerow(["meta", key, "", value, ""])
    return path


def load_instance_csv(path: str | Path, tier: str = "", instance_id: str = "") -> Instance:
    path = Path(path)
    stage_machines: Dict[int, int] = {}
    proc_entries: List[Tuple[int, int, float]] = []
    orders: List[Tuple[int, int, float, float]] = []
    meta: Dict[str, float] = {}
    header = None
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0] == "kind":
                continue
            kind = row[0]
            if kind == "header":
                header = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
            elif kind == "stage_machines":
                stage_machines[int(row[1])] = int(float(row[3]))
            elif kind == "proc":
                proc_entries.append((int(row[1]), int(row[2]), float(row[3])))
            elif kind == "order":
                orders.append((int(row[1]), int(row[2]), float(row[3]), float(row[4])))
            elif kind == "meta":
                try:
                    meta[row[1]] = float(row[3])
                except ValueError:
                    meta[row[1]] = row[3]
    if header is None:
        raise ValueError(f"instance file missing header row: {path}")
    product_count, stage_count, machine_count, order_count = header
    mps = tuple(stage_machines[j] for j in range(stage_count))
    proc = np.zeros((product_count * stage_count, machine_count), dtype=np.float32)
    for t, m, v in proc_entries:
        proc[t, m] = v
    orders.sort(key=lambda r: r[0])
    order_product = np.asarray([o[1] for o in orders], dtype=np.int64)
    arrivals = np.asarray([o[2] for o in orders], dtype=np.float64)
    due = np.asarray([o[3] for o in orders], dtype=np.float64)
    assert len(order_product) == order_count, f"order count mismatch in {path}"
    return Instance(
        instance_id=instance_id or path.stem, tier=tier or path.parent.name,
        product_count=product_count, stage_count=stage_count, machines_per_stage=mps,
        proc_times=proc, order_product=order_product, arrival_times=arrivals,
        due_dates=due, meta=meta,
    )
