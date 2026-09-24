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

OP_DIM, MA_DIM, ACT_DIM = 12, 5, 5      # 动作特征第 5 列为 no-op 标志位
# critic 的额外全局输入：eta_t / 未到达比例 / 剩余订单比例 / 时间进度 / 丢弃率
CRITIC_EXTRA = 5
# 等待门控的工况特征维数，与 environment.SchedulingEnv._gate_features 一致
GATE_DIM = 8


def mlp(sizes: List[int], out_act: bool = False, norm: bool = True) -> nn.Sequential:
    """隐藏层带 LayerNorm —— 观测特征跨三个数量级，无归一化时线性层难以利用小量纲特征。"""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
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
    """编码器 + 候选打分 actor + 状态 critic，外加 CoH（Commit-or-Hold）的三个可选头。

    三个头默认全关。关着时 `log_policy` 与 `forward` 走完全相同的算子序列，旧 checkpoint
    的评测逐位不变（scripts/_smoke_coh.py 检查这一点）。
      * commit     ：承诺评估器 p̂(o|s)，候选"现在派出后按时完成"的概率，用实际结果做 BCE 监督；
      * gate       ：等待门控 c(s)，输入工况特征、全局特征与 max p̂；
      * hold_value ：等待前景评估器 ĥ(s)，用事后标签做 BCE 监督。
    门控只改变 no-op 那一行概率的参数化：
        P(hold) = sigmoid(beta * (logit ĥ - logit max p̂) + c(s)),
        P(a)    = (1 - P(hold)) * softmax(派工 logits)_a,
    联合分布仍是归一化的离散分布，PPO 的比率、熵与 KL 早停不用改。
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        dim = int(cfg.get("network.embed_dim", 16))
        n_layers = int(cfg.get("network.gat_layers", 2))
        self.dim = dim
        self.encoder_kind = str(cfg.get("variant.encoder", "hgat"))
        self.use_action_attention = bool(cfg.get("network.action_attention", True))
        self.commit_critic = bool(cfg.get("coh.commit_critic", False))
        self.gate_enabled = bool(cfg.get("coh.gate", False))
        self.hold_critic = bool(cfg.get("coh.hold_critic", False))

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
            heads = max(int(cfg.get("network.action_attention_heads", 1)), 1)
            while feat % heads != 0:                # MultiheadAttention 要求 embed_dim 可整除 heads
                heads -= 1
            self.action_attn = nn.MultiheadAttention(feat, heads, batch_first=True)
            self.action_norm = nn.LayerNorm(feat)
        hidden = list(cfg.get("network.actor_hidden", [128, 128, 128]))
        if self.commit_critic:
            self.commit = mlp([feat, 128, 1])
            self._small_head(self.commit)              # 起点 p̂ = 0.5
        # 有承诺评估器时 actor 多看一维 p̂（detach：评估器只由标签训练，不被策略梯度拉偏）
        self.actor = mlp([feat + (1 if self.commit_critic else 0)] + hidden + [1])
        c_hidden = list(cfg.get("network.critic_hidden", [128, 128, 128]))
        self.critic = mlp([2 * dim + CRITIC_EXTRA] + c_hidden + [1])
        if self.gate_enabled:
            self.gate = mlp([GATE_DIM + CRITIC_EXTRA + 1, 64, 1])
            self.gate_beta = nn.Parameter(torch.ones(1))
            # 起点是 non-delay：三个头的输出层都用小初始化，门控 logit 起始约等于偏置 -1.5，
            # P(hold) 约 0.18（与单体 softmax 里 no-op 占 1/|A| 的先验相当），再由训练学会何时等待。
            # 否则随机初始化下门控 logit 的量级约为 1，部分种子从"逢事必等"起步（实测 3 个种子
            # 中 1 个的贪心读出 η=0.02）。
            self._small_head(self.gate, bias=-1.5)
        if self.hold_critic:
            self.hold_value = mlp([2 * dim + GATE_DIM + CRITIC_EXTRA, 64, 1])
            self._small_head(self.hold_value)

    @staticmethod
    def _small_head(head: nn.Sequential, bias: float = 0.0) -> None:
        """输出层小初始化：起始输出约等于偏置。"""
        nn.init.normal_(head[-1].weight, std=0.01)
        nn.init.constant_(head[-1].bias, bias)

    def encode(self, obs: dict, n_stage: int):
        op = obs["op"]
        ma = obs["ma"]
        if self.encoder_kind == "hgat":
            rate, adj = obs["proc_rate"], obs["adj"]
            for layer in self.layers:
                op, ma = layer(op, ma, rate, adj, n_stage)
            return op, ma
        return self.op_mlp(op), self.ma_mlp(ma)

    def _trunk(self, obs: dict, n_stage: int):
        """公共前段：编码 -> 候选特征（含动作级自注意力）-> 状态价值 -> no-op 行掩码。"""
        h_op, h_ma = self.encode(obs, n_stage)
        g_op, g_ma = h_op.mean(0), h_ma.mean(0)
        idx = obs["act_index"]
        eta = obs["eta_t"].reshape(1, 1)
        n_act = idx.shape[0]
        act_feat = obs["act_feat"]
        node_feat = torch.cat([h_op[idx[:, 0]], h_ma[idx[:, 1]]], dim=-1)
        noop = None
        if act_feat.shape[1] >= ACT_DIM:
            # 主动空闲不对应任何 (工序类型, 机器) 节点对，其占位下标 (0,0) 的嵌入是
            # 随状态漂移的噪声。按标志位屏蔽，让 no-op 行只由池化全局量与标志位决定。
            keep = (1.0 - act_feat[:, ACT_DIM - 1]).unsqueeze(-1)
            node_feat = node_feat * keep
            noop = act_feat[:, ACT_DIM - 1] > 0.5
        feats = torch.cat([
            node_feat,
            g_op.unsqueeze(0).expand(n_act, -1), g_ma.unsqueeze(0).expand(n_act, -1),
            act_feat, eta.expand(n_act, 1)], dim=-1)
        if self.use_action_attention and n_act > 1:
            # 残差 + LayerNorm 是必需的，不是装饰：候选集只有 1--5 个元素且特征高度相似，
            # 纯 attn @ v 会把所有候选映射成 mean(v)，logits 全等、策略恒为均匀分布，
            # 梯度比无注意力时小约三个数量级。
            attended, _ = self.action_attn(feats.unsqueeze(0), feats.unsqueeze(0),
                                           feats.unsqueeze(0), need_weights=False)
            feats = self.action_norm(feats + attended.squeeze(0))
        value = self.critic(torch.cat([g_op, g_ma, obs["global_feat"]], dim=-1)).squeeze(-1)
        return feats, g_op, g_ma, value, noop

    def forward(self, obs: dict, n_stage: int):
        """派工 logits 与状态价值（不含门控），供 BC 热启动与旧路径使用。"""
        feats, _, _, value, _ = self._trunk(obs, n_stage)
        if self.commit_critic:
            feats = torch.cat([feats, torch.sigmoid(self.commit(feats)).detach()], dim=-1)
        return self.actor(feats).squeeze(-1), value

    def log_policy(self, obs: dict, n_stage: int):
        """返回 (候选的 log 概率, 状态价值, 辅助头输出)。"""
        feats, g_op, g_ma, value, noop = self._trunk(obs, n_stage)
        aux = {"commit_logit": None, "hold_logit": None, "gate_logit": None}
        if self.commit_critic:
            commit_logit = self.commit(feats).squeeze(-1)
            aux["commit_logit"] = commit_logit
            feats = torch.cat([feats, torch.sigmoid(commit_logit).detach().unsqueeze(-1)], dim=-1)
        logits = self.actor(feats).squeeze(-1)
        has_noop = noop is not None and bool(noop.any()) and logits.shape[0] > 1
        if not (self.gate_enabled and has_noop):
            return F.log_softmax(logits, dim=-1), value, aux

        dispatch = ~noop
        disp_logp = F.log_softmax(logits[dispatch], dim=-1)
        ctx = torch.cat([obs["gate_feat"], obs["global_feat"]], dim=-1)
        zero = logits.new_zeros(())
        pmax_logit = aux["commit_logit"][dispatch].max() if self.commit_critic else zero
        hold_logit = self.hold_value(torch.cat([g_op, g_ma, ctx], dim=-1)).squeeze(-1) \
            if self.hold_critic else zero
        if self.hold_critic:
            aux["hold_logit"] = hold_logit
        c = self.gate(torch.cat([ctx, torch.sigmoid(pmax_logit).reshape(1)], dim=-1)).squeeze(-1)
        gate_logit = self.gate_beta.reshape(()) * (hold_logit - pmax_logit) + c
        aux["gate_logit"] = gate_logit
        log_probs = torch.empty_like(logits)
        log_probs[dispatch] = F.logsigmoid(-gate_logit) + disp_logp
        log_probs[noop] = F.logsigmoid(gate_logit)
        return log_probs, value, aux

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
