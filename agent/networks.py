"""网络：分类型 MLP 编码器 + Commit-or-Hold 的 actor / critic / 三个头。全部支持带掩码的批量输入。

编码器对工序类型节点与机器节点各做逐节点 MLP（LayerNorm 逐节点，填充不泄漏），掩码均值
池化得到全局嵌入；候选特征 = [所选节点嵌入 ‖ 全局嵌入 ‖ 动作特征 ‖ η_t]。
Commit-or-Hold 的联合分布：
    P(hold) = sigmoid(beta * (logit ĥ - logit max p̂) + c(s)),
    P(a)    = (1 - P(hold)) * softmax(派工 logits)_a,
无 no-op 的行退化为普通 softmax；关掉门控时 no-op 作为一行候选进 softmax。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.batch import NEG, Batch
from environment.env import ACT_DIM, GATE_DIM, GLOBAL_DIM, MA_DIM, OP_DIM


def mlp(sizes: List[int], out_act: bool = False, norm: bool = True) -> nn.Sequential:
    """隐藏层带 LayerNorm：观测特征量纲不一，无归一化时线性层难以同时利用大小量。"""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def small_head(head: nn.Sequential, bias: float = 0.0) -> None:
    """输出层小初始化：起始输出约等于偏置。"""
    nn.init.normal_(head[-1].weight, std=0.01)
    nn.init.constant_(head[-1].bias, bias)


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


def masked_log_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(x.masked_fill(~mask, NEG), dim=-1)


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = logp.masked_fill(~mask, 0.0)
    return -(lp.exp() * lp).sum(-1)


class Encoder(nn.Module):
    """逐节点 MLP + 掩码均值池化 + 候选特征拼接。主方法与三个学习基线共用。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        d = int(cfg.get("network.embed_dim", 16))
        self.dim = d
        self.op_mlp = mlp([OP_DIM, 4 * d, d], out_act=True)
        self.ma_mlp = mlp([MA_DIM, 4 * d, d], out_act=True)
        self.feat_dim = 4 * d + ACT_DIM + 1
        self.state_dim = 2 * d + GLOBAL_DIM + GATE_DIM

    def forward(self, b: Batch):
        h_op = self.op_mlp(b.op)                                   # [B, N, d]
        h_ma = self.ma_mlp(b.ma)                                   # [B, M, d]
        g_op, g_ma = masked_mean(h_op, b.op_mask), masked_mean(h_ma, b.ma_mask)
        d, A = self.dim, b.act_feat.shape[1]
        h_t = torch.gather(h_op, 1, b.act_index[..., 0].unsqueeze(-1).expand(-1, -1, d))
        h_m = torch.gather(h_ma, 1, b.act_index[..., 1].unsqueeze(-1).expand(-1, -1, d))
        noop = b.act_feat[..., ACT_DIM - 1] > 0.5                  # [B, A]
        # no-op 不对应任何节点对，其占位下标 (0,0) 的嵌入按标志位置零
        node = torch.cat([h_t, h_m], dim=-1) * (~noop).unsqueeze(-1).to(h_t.dtype)
        feats = torch.cat([node, g_op.unsqueeze(1).expand(-1, A, -1), g_ma.unsqueeze(1).expand(-1, A, -1),
                           b.act_feat, b.eta.view(-1, 1, 1).expand(-1, A, 1)], dim=-1)
        return feats, g_op, g_ma, noop

    def state(self, b: Batch, g_op: torch.Tensor, g_ma: torch.Tensor) -> torch.Tensor:
        return torch.cat([g_op, g_ma, b.global_feat, b.gate_feat], dim=-1)


@dataclass
class Out:
    logp: torch.Tensor                       # [B, K]，填充位 -inf
    value: torch.Tensor                      # [B]
    entropy: torch.Tensor                    # [B]
    logits: torch.Tensor                     # [B, K]
    mask: torch.Tensor                       # [B, K] 有效动作
    dispatch: torch.Tensor                   # [B, K] 派工行（K = 候选数时）
    noop: torch.Tensor                       # [B, K] no-op 行
    has_noop: torch.Tensor                   # [B]
    commit_logit: Optional[torch.Tensor] = None   # [B, K]
    hold_logit: Optional[torch.Tensor] = None     # [B]
    gate_logit: Optional[torch.Tensor] = None     # [B]


class ActorCritic(nn.Module):
    """Commit-or-Hold 的策略网络。开关全关时是"no-op 作为一行候选"的普通候选打分 actor-critic。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.enc = Encoder(cfg)
        d, feat = self.enc.dim, self.enc.feat_dim
        self.commit_critic = bool(cfg.get("coh.commit_critic", True))
        self.gate_enabled = bool(cfg.get("coh.gate", True))
        self.hold_critic = bool(cfg.get("coh.hold_critic", True))
        self.gate_context = bool(cfg.get("coh.gate_context", True))
        self.detach_gate_inputs = bool(cfg.get("coh.detach_gate_inputs", True))
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        c_hidden = list(cfg.get("network.critic_hidden", [128, 128, 128]))
        if self.commit_critic:
            self.commit = mlp([feat, 128, 1])
            small_head(self.commit)                       # 起点 p̂ = 0.5
        self.actor = mlp([feat + (1 if self.commit_critic else 0)] + hidden + [1])
        self.critic = mlp([2 * d + GLOBAL_DIM] + c_hidden + [1])
        if self.gate_enabled:
            ctx_dim = (GATE_DIM + GLOBAL_DIM) if self.gate_context else 0
            self.gate = mlp([ctx_dim + 1, 64, 1])
            self.gate_beta = nn.Parameter(torch.ones(1))
            # 起点是 non-delay：门控 logit 起始约等于偏置 -1.5，P(hold) 约 0.18，再由训练学会何时等待
            small_head(self.gate, bias=-1.5)
        if self.hold_critic:
            self.hold_value = mlp([2 * d + GATE_DIM + GLOBAL_DIM, 64, 1])
            small_head(self.hold_value)

    def forward(self, b: Batch) -> Out:
        feats, g_op, g_ma, noop = self.enc(b)
        B = b.size
        commit_logit = None
        if self.commit_critic:
            commit_logit = self.commit(feats).squeeze(-1)                          # [B, A]
            feats = torch.cat([feats, torch.sigmoid(commit_logit).detach().unsqueeze(-1)], dim=-1)
        logits = self.actor(feats).squeeze(-1)                                      # [B, A]
        value = self.critic(torch.cat([g_op, g_ma, b.global_feat], dim=-1)).squeeze(-1)
        dispatch = b.act_mask & ~noop
        has_noop = (b.act_mask & noop).any(-1)
        hold_logit = gate_logit = None
        if self.gate_enabled:
            disp_logp = masked_log_softmax(logits, dispatch)
            ctx = torch.cat([b.gate_feat, b.global_feat], dim=-1)
            if self.commit_critic:
                pmax = commit_logit.masked_fill(~dispatch, NEG).max(-1).values
            else:
                pmax = logits.new_zeros(B)
            if self.hold_critic:
                hold_logit = self.hold_value(torch.cat([g_op, g_ma, ctx], dim=-1)).squeeze(-1)
                hold_in = hold_logit
            else:
                hold_in = logits.new_zeros(B)
            if self.detach_gate_inputs:
                pmax, hold_in = pmax.detach(), hold_in.detach()
            gate_in = torch.sigmoid(pmax).unsqueeze(-1)
            if self.gate_context:
                gate_in = torch.cat([ctx, gate_in], dim=-1)
            c = self.gate(gate_in).squeeze(-1)
            gate_logit = self.gate_beta.reshape(()) * (hold_in - pmax) + c            # [B]
            log_keep = torch.where(has_noop, F.logsigmoid(-gate_logit), torch.zeros_like(gate_logit))
            logp = disp_logp + log_keep.unsqueeze(-1)
            logp = torch.where(noop & b.act_mask, F.logsigmoid(gate_logit).unsqueeze(-1).expand_as(logp), logp)
        else:
            logp = masked_log_softmax(logits, b.act_mask)
        logp = logp.masked_fill(~b.act_mask, NEG)
        return Out(logp=logp, value=value, entropy=masked_entropy(logp, b.act_mask), logits=logits,
                   mask=b.act_mask, dispatch=dispatch, noop=noop & b.act_mask, has_noop=has_noop,
                   commit_logit=commit_logit.masked_fill(~b.act_mask, 0.0) if commit_logit is not None else None,
                   hold_logit=hold_logit, gate_logit=gate_logit)


def greedy_actions(out: Out) -> torch.Tensor:
    """贪心读出：带门控时先判"等不等"（gate logit > 0），再在派工行里取 argmax。"""
    if out.gate_logit is None:
        return out.logp.argmax(-1)
    hold = out.has_noop & (out.gate_logit > 0.0)
    disp = out.logits.masked_fill(~out.dispatch, NEG).argmax(-1)
    noop_idx = out.noop.to(torch.int64).argmax(-1)
    return torch.where(hold, noop_idx, disp)
