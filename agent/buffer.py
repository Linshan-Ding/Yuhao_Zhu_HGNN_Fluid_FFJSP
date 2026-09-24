"""episode 记录、GAE 与 replay。

`EpisodeRecord` 由 worker 在 episode 结束时组装：逐步数组堆叠，候选轴填充到本 episode 的最大
候选数，两类监督标签（承诺 / 等待前景）在这里回填。记录只含 numpy 数组与标量，可直接 pickle。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from environment.env import ACT_DIM


@dataclass
class EpisodeRecord:
    op: np.ndarray            # [T, N, OP]
    ma: np.ndarray            # [T, M, MA]
    act_feat: np.ndarray      # [T, A, ACT]
    act_mask: np.ndarray      # [T, A]
    act_index: np.ndarray     # [T, A, 2]
    global_feat: np.ndarray   # [T, 5]
    gate_feat: np.ndarray     # [T, 8]
    eta: np.ndarray           # [T]
    action: np.ndarray        # [T] 记录用的动作（候选下标，或规则选择基线的规则下标）
    env_action: np.ndarray    # [T] 实际执行的候选下标
    logp: np.ndarray          # [T]
    value: np.ndarray         # [T]
    reward: np.ndarray        # [T]
    done: np.ndarray          # [T]
    commit_label: np.ndarray  # [T] -1/0/1
    hold_label: np.ndarray    # [T] -1/0/1
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    info: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.reward.shape[0])

    @staticmethod
    def from_steps(steps: List[dict], env) -> "EpisodeRecord":
        T = len(steps)
        A = max(int(s["obs"]["act_feat"].shape[0]) for s in steps)
        first = steps[0]["obs"]
        rec = EpisodeRecord(
            op=np.stack([s["obs"]["op"] for s in steps]).astype(np.float32),
            ma=np.stack([s["obs"]["ma"] for s in steps]).astype(np.float32),
            act_feat=np.zeros((T, A, ACT_DIM), np.float32), act_mask=np.zeros((T, A), bool),
            act_index=np.zeros((T, A, 2), np.int64),
            global_feat=np.stack([s["obs"]["global_feat"] for s in steps]).astype(np.float32),
            gate_feat=np.stack([s["obs"]["gate_feat"] for s in steps]).astype(np.float32),
            eta=np.asarray([float(s["obs"]["eta_t"]) for s in steps], np.float32),
            action=np.asarray([s["action"] for s in steps], np.int64),
            env_action=np.asarray([s["env_action"] for s in steps], np.int64),
            logp=np.asarray([s["logp"] for s in steps], np.float32),
            value=np.asarray([s["value"] for s in steps], np.float32),
            reward=np.asarray([s["reward"] for s in steps], np.float32),
            done=np.asarray([s["done"] for s in steps], bool),
            commit_label=np.full(T, -1, np.int8), hold_label=np.full(T, -1, np.int8),
        )
        for t, s in enumerate(steps):
            n = s["obs"]["act_feat"].shape[0]
            rec.act_feat[t, :n] = s["obs"]["act_feat"]
            rec.act_mask[t, :n] = True
            rec.act_index[t, :n] = s["obs"]["act_index"]
        extra_keys = set().union(*(s["extra"].keys() for s in steps))
        for k in extra_keys:
            vals = [s["extra"][k] for s in steps]
            if isinstance(vals[0], np.ndarray) and vals[0].ndim == 1 and vals[0].dtype == bool:
                arr = np.zeros((T, A), bool)                     # 候选轴掩码，填充到 A
                for t, v in enumerate(vals):
                    arr[t, :v.shape[0]] = v
            else:
                arr = np.asarray(vals)
            rec.extras[k] = arr
        # 监督标签：订单结果与等待前景在 episode 结束时全部已知
        orders = np.asarray([s["order"] for s in steps], np.int64)
        hold_ids = np.asarray([s["hold_id"] for s in steps], np.int64)
        outcome = env.order_outcome
        rec.commit_label = np.where(orders >= 0, outcome[np.maximum(orders, 0)], -1).astype(np.int8)
        labels = env.hold_labels()
        rec.hold_label = np.asarray([labels.get(int(h), -1) if h >= 0 else -1 for h in hold_ids], np.int8)
        rec.info = {"eta": env.eta, "nu": env.nu, "held_share": env.held_share, "steps": env.step_count,
                    "noop_offered": env.stats.noop_offered, "noop_used": env.stats.noop_used,
                    "S": env.problem.n_order, "DDT": float(env.inst.meta.get("DDT", 0.0)),
                    "rho": float(env.inst.meta.get("rho_sys", 0.0)),
                    "n_cand_mean": float(np.mean(env.stats.n_candidates)) if env.stats.n_candidates else 0.0}
        return rec

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return d

    @staticmethod
    def from_dict(d: dict) -> "EpisodeRecord":
        return EpisodeRecord(**d)


def gae(rec: EpisodeRecord, gamma: float = 1.0, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """逐 episode 的广义优势估计；记录是完整 episode，末步之后 bootstrap 为 0。"""
    T = len(rec)
    adv = np.zeros(T, np.float64)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nonterminal = 0.0 if (t == T - 1 or rec.done[t]) else 1.0
        next_value = float(rec.value[t + 1]) if nonterminal else 0.0
        delta = float(rec.reward[t]) + gamma * next_value - float(rec.value[t])
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + rec.value.astype(np.float64)


class ReplayStore:
    """价值型基线的 replay：整段 episode 存放，按转移数限容，采样返回 (记录下标, 步下标)。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.records: List[EpisodeRecord] = []
        self.size = 0

    def add(self, episodes: Sequence[EpisodeRecord]) -> None:
        for rec in episodes:
            self.records.append(rec)
            self.size += len(rec)
        while self.size > self.capacity and len(self.records) > 1:
            self.size -= len(self.records.pop(0))

    def sample(self, n: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
        lengths = np.asarray([len(r) for r in self.records], np.int64)
        flat = rng.integers(0, lengths.sum(), size=n)
        bounds = np.cumsum(lengths)
        e = np.searchsorted(bounds, flat, side="right")
        t = flat - (bounds[e] - lengths[e])
        return list(zip(e.tolist(), t.tolist()))
