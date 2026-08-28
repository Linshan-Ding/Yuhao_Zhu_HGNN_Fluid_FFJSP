"""异构图注意力编码器 + actor/critic（稿件 §4.4、§4.7）。

两阶段更新：机器节点先经边感知注意力聚合邻接工序类型，工序节点再经多分支变换
聚合机器邻域、前驱、后继与自身特征。候选动作经动作级自注意力后打分。
`encoder=mlp` 时整体退化为参数量匹配的 MLP（消融变体 FSHGRL-NoHG）。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_DIM, MA_DIM, ACT_DIM = 12, 5, 3


def mlp(sizes: List[int], out_act: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class HGATLayer(nn.Module):
    """一层两阶段异构图注意力。"""

    def __init__(self, op_in: int, ma_in: int, dim: int) -> None:
        super().__init__()
        self.w_ma = nn.Linear(ma_in, dim)
        self.w_op = nn.Linear(op_in + 1, dim)          # +1 为边特征 mu
        self.attn = nn.Linear(2 * dim, 1)
        self.branch_m = mlp([dim, dim, dim])
        self.branch_prev = mlp([op_in, dim, dim])
        self.branch_next = mlp([op_in, dim, dim])
        self.branch_self = mlp([op_in, dim, dim])
        self.fuse = mlp([4 * dim, dim, dim])

    def forward(self, op, ma, rate, adj, n_stage):
        n_op, n_ma = op.shape[0], ma.shape[0]
        # --- 阶段 1：机器节点聚合邻接工序类型（边感知注意力 + 自注意力项）
        edge_op = torch.cat([op.unsqueeze(1).expand(n_op, n_ma, -1),
                             rate.unsqueeze(-1)], dim=-1)          # [O, M, op_in+1]
        h_op_e = self.w_op(edge_op)                                # [O, M, d]
        h_ma = self.w_ma(ma)                                       # [M, d]
        pair = torch.cat([h_ma.unsqueeze(0).expand(n_op, n_ma, -1), h_op_e], dim=-1)
        score = F.leaky_relu(self.attn(pair).squeeze(-1), 0.2)     # [O, M]
        score = score.masked_fill(~adj, torch.finfo(score.dtype).min)
        self_score = F.leaky_relu(self.attn(torch.cat([h_ma, h_ma], dim=-1)).squeeze(-1), 0.2)
        alpha = torch.softmax(torch.cat([score, self_score.unsqueeze(0)], dim=0), dim=0)
        ma_out = F.elu((alpha[:n_op].unsqueeze(-1) * h_op_e).sum(0) + alpha[n_op].unsqueeze(-1) * h_ma)

        # --- 阶段 2：工序节点多分支聚合
        deg = adj.sum(1, keepdim=True).clamp(min=1).to(ma_out.dtype)
        nbr = (adj.to(ma_out.dtype) @ ma_out) / deg                # [O, d]
        prev_op = torch.roll(op, shifts=1, dims=0)
        next_op = torch.roll(op, shifts=-1, dims=0)
        idx = torch.arange(n_op, device=op.device)
        prev_op = torch.where((idx % n_stage == 0).unsqueeze(-1), torch.zeros_like(prev_op), prev_op)
        next_op = torch.where((idx % n_stage == n_stage - 1).unsqueeze(-1),
                              torch.zeros_like(next_op), next_op)
        op_out = self.fuse(F.elu(torch.cat([
            self.branch_m(nbr), self.branch_prev(prev_op),
            self.branch_next(next_op), self.branch_self(op)], dim=-1)))
        return op_out, ma_out


class ActorCritic(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        dim = int(cfg.get("network.embed_dim", 16))
        n_layers = int(cfg.get("network.gat_layers", 2))
        self.dim = dim
        self.encoder_kind = str(cfg.get("variant.encoder", "hgat"))
        self.use_action_attention = bool(cfg.get("network.action_attention", True))

        if self.encoder_kind == "hgat":
            self.layers = nn.ModuleList()
            op_in, ma_in = OP_DIM, MA_DIM
            for _ in range(n_layers):
                self.layers.append(HGATLayer(op_in, ma_in, dim))
                op_in, ma_in = dim, dim
        else:                                       # 参数量匹配的 MLP 编码器（NoHG 变体）
            self.op_mlp = mlp([OP_DIM, 4 * dim, dim], out_act=True)
            self.ma_mlp = mlp([MA_DIM, 4 * dim, dim], out_act=True)

        feat = 4 * dim + ACT_DIM + 1                # h_op ‖ h_ma ‖ hbar_O ‖ hbar_M ‖ 动作特征 ‖ eta_t
        if self.use_action_attention:
            heads = int(cfg.get("network.action_attention_heads", 1))
            self.q = nn.Linear(feat, feat)
            self.k = nn.Linear(feat, feat)
            self.v = nn.Linear(feat, feat)
            self.heads = heads
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        self.actor = mlp([feat] + hidden + [1])
        c_hidden = list(cfg.get("network.critic_hidden", [128, 128, 128]))
        self.critic = mlp([2 * dim + 1] + c_hidden + [1])

    def encode(self, obs: dict, n_stage: int):
        op = obs["op"]
        ma = obs["ma"]
        if self.encoder_kind == "hgat":
            rate, adj = obs["proc_rate"], obs["adj"]
            for layer in self.layers:
                op, ma = layer(op, ma, rate, adj, n_stage)
            return op, ma
        return self.op_mlp(op), self.ma_mlp(ma)

    def forward(self, obs: dict, n_stage: int):
        h_op, h_ma = self.encode(obs, n_stage)
        g_op, g_ma = h_op.mean(0), h_ma.mean(0)
        idx = obs["act_index"]
        eta = obs["eta_t"].reshape(1, 1)
        n_act = idx.shape[0]
        feats = torch.cat([
            h_op[idx[:, 0]], h_ma[idx[:, 1]],
            g_op.unsqueeze(0).expand(n_act, -1), g_ma.unsqueeze(0).expand(n_act, -1),
            obs["act_feat"], eta.expand(n_act, 1)], dim=-1)
        if self.use_action_attention and n_act > 1:
            q, k, v = self.q(feats), self.k(feats), self.v(feats)
            attn = torch.softmax(q @ k.transpose(0, 1) / (feats.shape[-1] ** 0.5), dim=-1)
            feats = attn @ v
        logits = self.actor(feats).squeeze(-1)
        value = self.critic(torch.cat([g_op, g_ma, eta.squeeze(0)], dim=-1)).squeeze(-1)
        return logits, value


def obs_to_tensors(obs: dict, device: torch.device) -> dict:
    out = {}
    for key, value in obs.items():
        if key == "adj":
            out[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        elif key == "act_index":
            out[key] = torch.as_tensor(value, dtype=torch.long, device=device)
        else:
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return out
