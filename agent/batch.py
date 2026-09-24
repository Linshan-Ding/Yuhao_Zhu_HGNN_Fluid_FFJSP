"""批量张量：把变长的观测（节点数、候选数各异）填充成带掩码的 [B, ...] 张量。

填充一律在尾部，掩码 True = 有效。`collate_records` 从 episode 记录里取任意一组转移
（PPO 取整个 epoch，DQN 取采样的 (episode, t) 对），`single_batch` 把一条观测包成 B=1 的批。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from environment.env import ACT_DIM, GATE_DIM, GLOBAL_DIM, MA_DIM, OP_DIM

NEG = float("-inf")
_FIELDS = ("op", "op_mask", "ma", "ma_mask", "act_feat", "act_mask", "act_index",
           "global_feat", "gate_feat", "eta")


@dataclass
class Batch:
    op: torch.Tensor            # [B, N, OP_DIM]
    op_mask: torch.Tensor       # [B, N] bool
    ma: torch.Tensor            # [B, M, MA_DIM]
    ma_mask: torch.Tensor       # [B, M] bool
    act_feat: torch.Tensor      # [B, A, ACT_DIM]
    act_mask: torch.Tensor      # [B, A] bool
    act_index: torch.Tensor     # [B, A, 2] long
    global_feat: torch.Tensor   # [B, GLOBAL_DIM]
    gate_feat: torch.Tensor     # [B, GATE_DIM]
    eta: torch.Tensor           # [B]
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.op.shape[0])

    def to(self, device) -> "Batch":
        return Batch(*(getattr(self, f).to(device, non_blocking=True) for f in _FIELDS),
                     extras={k: v.to(device, non_blocking=True) for k, v in self.extras.items()})

    def subset(self, idx: torch.Tensor) -> "Batch":
        """取子批并裁掉多余的尾部填充（有效行总在前面）。"""
        take = {f: getattr(self, f).index_select(0, idx) for f in _FIELDS}
        n = int(take["op_mask"].sum(1).max().item())
        m = int(take["ma_mask"].sum(1).max().item())
        a = int(take["act_mask"].sum(1).max().item())
        extras = {}
        for k, v in self.extras.items():
            v = v.index_select(0, idx)
            if v.dim() >= 2 and v.shape[1] == self.act_mask.shape[1]:
                v = v[:, :a]
            extras[k] = v
        return Batch(take["op"][:, :n], take["op_mask"][:, :n], take["ma"][:, :m], take["ma_mask"][:, :m],
                     take["act_feat"][:, :a], take["act_mask"][:, :a], take["act_index"][:, :a],
                     take["global_feat"], take["gate_feat"], take["eta"], extras=extras)


def collate_records(records: Sequence, pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, np.ndarray]:
    """把若干 episode 记录里的转移拼成填充后的 numpy 数组字典。

    `pairs` 为 (记录下标, 步下标) 列表；None 表示按顺序取全部转移。
    返回的 extras 里带候选轴的数组（例如 window_mask）同样填充到候选数上限。
    """
    if pairs is None:
        pairs = [(e, t) for e, rec in enumerate(records) for t in range(len(rec))]
    B = len(pairs)
    used = sorted({e for e, _ in pairs})
    N = max(records[e].op.shape[1] for e in used)
    M = max(records[e].ma.shape[1] for e in used)
    A = max(records[e].act_feat.shape[1] for e in used)
    out = {
        "op": np.zeros((B, N, OP_DIM), np.float32), "op_mask": np.zeros((B, N), bool),
        "ma": np.zeros((B, M, MA_DIM), np.float32), "ma_mask": np.zeros((B, M), bool),
        "act_feat": np.zeros((B, A, ACT_DIM), np.float32), "act_mask": np.zeros((B, A), bool),
        "act_index": np.zeros((B, A, 2), np.int64),
        "global_feat": np.zeros((B, GLOBAL_DIM), np.float32), "gate_feat": np.zeros((B, GATE_DIM), np.float32),
        "eta": np.zeros(B, np.float32),
    }
    extra_keys = set()
    for e in used:
        extra_keys |= set(records[e].extras)
    extras: Dict[str, np.ndarray] = {}
    # 按记录分组填充，减少 Python 循环次数
    by_rec: Dict[int, List[Tuple[int, int]]] = {}
    for i, (e, t) in enumerate(pairs):
        by_rec.setdefault(e, []).append((i, t))
    for e, items in by_rec.items():
        rec = records[e]
        rows = np.fromiter((i for i, _ in items), dtype=np.int64, count=len(items))
        ts = np.fromiter((t for _, t in items), dtype=np.int64, count=len(items))
        n, m, a = rec.op.shape[1], rec.ma.shape[1], rec.act_feat.shape[1]
        out["op"][rows, :n] = rec.op[ts]
        out["op_mask"][rows, :n] = True
        out["ma"][rows, :m] = rec.ma[ts]
        out["ma_mask"][rows, :m] = True
        out["act_feat"][rows, :a] = rec.act_feat[ts]
        out["act_mask"][rows, :a] = rec.act_mask[ts]
        out["act_index"][rows, :a] = rec.act_index[ts]
        out["global_feat"][rows] = rec.global_feat[ts]
        out["gate_feat"][rows] = rec.gate_feat[ts]
        out["eta"][rows] = rec.eta[ts]
        for k in extra_keys:
            v = rec.extras[k][ts]
            if k not in extras:
                shape = (B, A) + v.shape[2:] if v.ndim >= 2 and v.shape[1] == a else (B,) + v.shape[1:]
                extras[k] = np.zeros(shape, v.dtype)
            if v.ndim >= 2 and v.shape[1] == a:
                extras[k][rows, :a] = v
            else:
                extras[k][rows] = v
    assert out["act_mask"].any(1).all(), "every transition needs at least one valid candidate"
    assert (out["act_index"][..., 0] < N).all() and (out["act_index"][..., 1] < M).all()
    out["extras"] = extras
    return out


def to_batch(arrays: Dict[str, np.ndarray], device=None) -> Batch:
    device = device or torch.device("cpu")
    tensors = {f: torch.from_numpy(np.ascontiguousarray(arrays[f])).to(device) for f in _FIELDS}
    extras = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.get("extras", {}).items()}
    return Batch(**tensors, extras=extras)


def single_batch(obs: dict, device=None) -> Batch:
    """一条观测 -> B=1 的批（无填充）。worker 与评测都走这条路径。"""
    device = device or torch.device("cpu")
    n_act = obs["act_feat"].shape[0]
    return Batch(
        op=torch.from_numpy(obs["op"]).unsqueeze(0).to(device),
        op_mask=torch.ones(1, obs["op"].shape[0], dtype=torch.bool, device=device),
        ma=torch.from_numpy(obs["ma"]).unsqueeze(0).to(device),
        ma_mask=torch.ones(1, obs["ma"].shape[0], dtype=torch.bool, device=device),
        act_feat=torch.from_numpy(obs["act_feat"]).unsqueeze(0).to(device),
        act_mask=torch.ones(1, n_act, dtype=torch.bool, device=device),
        act_index=torch.from_numpy(obs["act_index"]).unsqueeze(0).to(device),
        global_feat=torch.from_numpy(obs["global_feat"]).unsqueeze(0).to(device),
        gate_feat=torch.from_numpy(obs["gate_feat"]).unsqueeze(0).to(device),
        eta=torch.as_tensor([float(obs["eta_t"])], dtype=torch.float32, device=device),
    )
