"""三个学习基线在 DFFSP-HFOI 上的适配实现（稿件 §5.7.1 与 Table T-NEW-6）。

适配原则（README §6 与论文协议表一致）：
  * 共用同一离散事件环境、同一算例生成器、同一奖励 Eq.(46) 与同一评测种子；
  * 任何基线都拿不到流体解 —— 那正是被检验的机制；
  * 也不削弱基线：价值型方法同时给出"规则选择"与"直接三元组选择"两种动作空间，
    取其强者上报；
  * 预算按环境交互步数对齐，而非按 epoch（各方法 episode 长度不同）。

因此它们是**适配版**而非原文逐字复现，这一点在论文与本文件中都明确写出。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from agent.baselines.rules import RULES, select as rule_select
from agent.networks import ACT_DIM, MA_DIM, OP_DIM, mlp, obs_to_tensors

RULE_POOL = [r for r in RULES if r != "RRC"]


def _global_state(obs: dict) -> torch.Tensor:
    """把异构图观测压成一个定长向量，供向量状态型基线使用。"""
    op, ma = obs["op"], obs["ma"]
    return torch.cat([op.mean(0), op.max(0).values, ma.mean(0), ma.max(0).values,
                      obs["eta_t"].reshape(1)], dim=-1)


STATE_DIM = 2 * OP_DIM + 2 * MA_DIM + 1


class DRLG(nn.Module):
    """GNN + 策略梯度的调度规则生成方法（\\citep{huang2023novel} 的适配版）。

    动作空间：规则池上的分布 —— 与原文一致；状态用图嵌入均值池化。
    """

    name = "DRLG"
    action_space = "rule_selection"

    def __init__(self, cfg) -> None:
        super().__init__()
        dim = int(cfg.get("network.embed_dim", 16))
        self.op_enc = mlp([OP_DIM, 4 * dim, dim], out_act=True)
        self.ma_enc = mlp([MA_DIM, 4 * dim, dim], out_act=True)
        self.policy = mlp([2 * dim + 1, 128, 128, len(RULE_POOL)])
        self.value = mlp([2 * dim + 1, 128, 128, 1])

    def _embed(self, obs):
        return torch.cat([self.op_enc(obs["op"]).mean(0),
                          self.ma_enc(obs["ma"]).mean(0),
                          obs["eta_t"].reshape(1)], dim=-1)

    def forward(self, obs):
        h = self._embed(obs)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, env, actions, obs_np, rng, epsilon=0.0, greedy=False):
        obs = obs_to_tensors(obs_np, torch.device("cpu"))
        logits, value = self(obs)
        probs = torch.softmax(logits, dim=-1)
        r = int(torch.argmax(probs).item()) if greedy else int(torch.multinomial(probs, 1).item())
        return rule_select(RULE_POOL[r], env, actions, rng), float(value.item())


class _DQNBase(nn.Module):
    """价值型基线的公共部分：Q 网络 + epsilon-greedy 行为。"""

    action_space = "triplet"

    def __init__(self, cfg, extra_state: int = 0) -> None:
        super().__init__()
        dim = int(cfg.get("network.embed_dim", 16))
        self.op_enc = mlp([OP_DIM, 4 * dim, dim], out_act=True)
        self.ma_enc = mlp([MA_DIM, 4 * dim, dim], out_act=True)
        self.q = mlp([2 * dim + ACT_DIM + 1 + extra_state, 128, 128, 1])

    def q_values(self, obs, extra: torch.Tensor | None = None):
        h_op, h_ma = self.op_enc(obs["op"]), self.ma_enc(obs["ma"])
        idx = obs["act_index"]
        n = idx.shape[0]
        feats = [h_op[idx[:, 0]], h_ma[idx[:, 1]], obs["act_feat"],
                 obs["eta_t"].reshape(1, 1).expand(n, 1)]
        if extra is not None:
            feats.append(extra.reshape(1, -1).expand(n, -1))
        return self.q(torch.cat(feats, dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, env, actions, obs_np, rng, epsilon=0.0, greedy=False):
        obs = obs_to_tensors(obs_np, torch.device("cpu"))
        qs = self.q_values(obs)
        if not greedy and rng.random() < epsilon:
            return int(rng.integers(len(actions))), float(qs.mean().item())
        return int(torch.argmax(qs).item()), float(qs.max().item())


class AHPDQN(_DQNBase):
    """自适应混合优先级 DQN（\\citep{du2025adaptive} 的适配版）。

    原文在优先级集合上决策；适配后把该优先级集合映射到本问题的三元组动作，
    并保留其"自适应混合"的加权优先级特征作为额外状态维。
    """

    name = "AHP-DQN"
    N_PRIORITY = 3          # 松弛期 / 处理速率 / 流体份额 三个优先级代理

    def __init__(self, cfg) -> None:
        super().__init__(cfg, extra_state=self.N_PRIORITY)

    @staticmethod
    def _priority_features(obs) -> torch.Tensor:
        """自适应混合优先级特征：三个优先级代理各自会选中的动作位置（归一化）。

        全部从观测本身导出，不依赖 env/actions —— 这样训练时的 Q 回归与决策时的
        前向使用同一份特征，不会出现"训练/推断特征不一致"这一类隐蔽缺陷。
        """
        act = obs["act_feat"]                       # [n, 3] = (归一化松弛期, 速率, 流体份额)
        n = max(act.shape[0], 1)
        return torch.stack([
            act[:, 0].argmin().float() / n,         # 最紧急
            act[:, 1].argmax().float() / n,         # 最快
            act[:, 2].argmax().float() / n,         # 流体最支持
        ])

    def q_values(self, obs, extra=None):
        return super().q_values(obs, extra=self._priority_features(obs) if extra is None else extra)

    @torch.no_grad()
    def act(self, env, actions, obs_np, rng, epsilon=0.0, greedy=False):
        obs = obs_to_tensors(obs_np, torch.device("cpu"))
        qs = self.q_values(obs)
        if not greedy and rng.random() < epsilon:
            return int(rng.integers(len(actions))), float(qs.mean().item())
        return int(torch.argmax(qs).item()), float(qs.max().item())


class HSDDQN(_DQNBase):
    """分层策略优化双 DQN（\\citep{ren2025hierarchical} 的适配版）。

    上层在规则族上决策、下层在该规则给出的候选内选具体三元组，与原文的分层结构一致。
    """

    name = "HSDDQN"
    action_space = "hierarchical"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        dim = int(cfg.get("network.embed_dim", 16))
        self.upper = mlp([2 * dim + 1, 128, len(RULE_POOL)])

    @torch.no_grad()
    def act(self, env, actions, obs_np, rng, epsilon=0.0, greedy=False):
        obs = obs_to_tensors(obs_np, torch.device("cpu"))
        h = torch.cat([self.op_enc(obs["op"]).mean(0),
                       self.ma_enc(obs["ma"]).mean(0), obs["eta_t"].reshape(1)], dim=-1)
        upper_q = self.upper(h)
        if not greedy and rng.random() < epsilon:
            rule_idx = int(rng.integers(len(RULE_POOL)))
        else:
            rule_idx = int(torch.argmax(upper_q).item())
        preferred = rule_select(RULE_POOL[rule_idx], env, actions, rng)
        qs = self.q_values(obs)
        # 下层只在上层规则的邻域内微调（分层约束）
        window = 3
        lo, hi = max(0, preferred - window), min(len(actions), preferred + window + 1)
        local = int(torch.argmax(qs[lo:hi]).item()) + lo
        return local, float(upper_q.max().item())


BASELINES = {"DRLG": DRLG, "AHP-DQN": AHPDQN, "HSDDQN": HSDDQN}
