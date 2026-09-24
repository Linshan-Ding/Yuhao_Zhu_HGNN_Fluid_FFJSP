"""价值型学习基线的网络：三元组 Double DQN（AHP-DQN 风格）与分层 Double DQN（HSDDQN 风格）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from agent.batch import NEG, Batch
from agent.baselines.rule_ppo import HOLD, N_ACTIONS
from agent.networks import Encoder, mlp


@dataclass
class QOut:
    q: torch.Tensor                         # [B, A] 逐候选，填充 -inf
    q_upper: Optional[torch.Tensor] = None  # [B, 7]（分层）
    upper_mask: Optional[torch.Tensor] = None


def priority_features(b: Batch, dispatch: torch.Tensor) -> torch.Tensor:
    """三个优先级代理各自会选中的候选位置（归一化）：最紧急、最短工时、最小临界比。"""
    A = b.act_feat.shape[1]
    big = torch.full_like(b.act_feat[..., 0], float("inf"))
    cols = []
    for c in (0, 1, 2):
        key = torch.where(dispatch, b.act_feat[..., c], big)
        cols.append(key.argmin(-1).to(b.act_feat.dtype) / max(A, 1))
    return torch.stack(cols, dim=-1)                                       # [B, 3]


class TripletQNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.enc = Encoder(cfg)
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        self.q = mlp([self.enc.feat_dim + 3] + hidden + [1])

    def forward(self, b: Batch) -> QOut:
        feats, g_op, g_ma, noop = self.enc(b)
        dispatch = b.act_mask & ~noop
        prio = priority_features(b, dispatch).unsqueeze(1).expand(-1, feats.shape[1], -1)
        q = self.q(torch.cat([feats, prio], dim=-1)).squeeze(-1).masked_fill(~b.act_mask, NEG)
        return QOut(q=q)


class HierQNet(nn.Module):
    """上层 Q 在 6 条规则 + 保留产能上，下层 Q 逐候选（窗口限制由学习器与 worker 用掩码施加）。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.enc = Encoder(cfg)
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        self.upper = mlp([self.enc.state_dim] + hidden + [N_ACTIONS])
        self.lower = mlp([self.enc.feat_dim] + hidden + [1])

    def forward(self, b: Batch) -> QOut:
        feats, g_op, g_ma, noop = self.enc(b)
        has_noop = (b.act_mask & noop).any(-1)
        upper_mask = torch.ones(b.size, N_ACTIONS, dtype=torch.bool, device=feats.device)
        upper_mask[:, HOLD] = has_noop
        q_upper = self.upper(self.enc.state(b, g_op, g_ma)).masked_fill(~upper_mask, NEG)
        q = self.lower(feats).squeeze(-1).masked_fill(~b.act_mask, NEG)
        return QOut(q=q, q_upper=q_upper, upper_mask=upper_mask)
