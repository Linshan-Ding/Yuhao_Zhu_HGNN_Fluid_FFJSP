"""规则选择 PPO（DRLG 风格的学习基线）：在 6 条调度规则 + 保留产能里选一个，规则再定三元组。"""
from __future__ import annotations

import torch
import torch.nn as nn

from agent.batch import Batch
from agent.networks import Encoder, Out, masked_entropy, masked_log_softmax, mlp

RULE_POOL = ["MOR", "FIFO", "MWKR", "SPT", "EDD", "Random"]
HOLD = len(RULE_POOL)                      # 第 7 个动作 = 保留产能
N_ACTIONS = HOLD + 1


class RulePolicyNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.enc = Encoder(cfg)
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        c_hidden = list(cfg.get("network.critic_hidden", [128, 128, 128]))
        self.policy = mlp([self.enc.state_dim] + hidden + [N_ACTIONS])
        self.value = mlp([self.enc.state_dim] + c_hidden + [1])

    def forward(self, b: Batch) -> Out:
        feats, g_op, g_ma, noop = self.enc(b)
        state = self.enc.state(b, g_op, g_ma)
        logits = self.policy(state)                                         # [B, 7]
        has_noop = (b.act_mask & noop).any(-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, HOLD] = has_noop
        logp = masked_log_softmax(logits, mask)
        return Out(logp=logp, value=self.value(state).squeeze(-1), entropy=masked_entropy(logp, mask),
                   logits=logits, mask=mask, dispatch=mask, noop=torch.zeros_like(mask), has_noop=has_noop)
