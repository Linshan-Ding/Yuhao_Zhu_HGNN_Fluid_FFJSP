"""PPO 更新，重要性比率对行为策略修正（稿件 Eq. 47、Prop 4）。

    rho_t = pi_theta(a|omega) / b_k(a|omega),   b_k = (1-eps_k) pi_old + eps_k/|A_f|

Prop 4(b) 给出 rho_t <= |A_f| / eps_k —— 该上界与实测最大比率一并落进 log.csv，
既是诊断量，也是"剪枝同时收紧优化方差"这一论断的直接证据。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from agent.buffer import RolloutBuffer
from agent.networks import obs_to_tensors


class PPOAgent:
    def __init__(self, net, cfg, device: torch.device) -> None:
        self.net = net.to(device)
        self.device = device
        self.cfg = cfg
        p = cfg.get("ppo")
        self.gamma = float(cfg.get("reward.gamma", 1.0))
        self.lam = float(p["gae_lambda"])
        self.clip = float(p["clip_eps"])
        self.epochs = int(p["update_epochs"])
        self.c1 = float(p["policy_coeff"])
        self.c2 = float(p["value_coeff"])
        self.c3 = float(p["entropy_coeff"])
        self.max_grad_norm = float(p["max_grad_norm"])
        self.target_kl = float(p["target_kl"])
        self.minibatch = int(p["minibatch_size"])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(p["lr"]))

        e = cfg.get("exploration")
        self.correct_behaviour = bool(e["behaviour_correction"])
        self.eps0 = float(e["epsilon0"])
        self.eps_min = float(e["epsilon_min"])
        self.anneal = max(int(e["anneal_epochs"]), 1)
        # CoH 辅助监督（承诺评估器 / 等待前景评估器）的权重；两个头都关着时没有任何作用
        self.critic_coeff = float(cfg.get("coh.critic_coeff", 0.5))

    def epsilon(self, epoch: int) -> float:
        """eps_k = max(eps0 (1 - k/K_tot), eps_min)，稿件 Eq. (53)。"""
        if self.eps0 <= 0:
            return 0.0
        return max(self.eps0 * (1.0 - epoch / self.anneal), self.eps_min)

    @torch.no_grad()
    def act(self, obs_np: dict, n_stage: int, epsilon: float, greedy: bool = False):
        """返回 (动作下标, log b_k, value, 诊断)。

        行为策略 b_k 是"目标策略"与"均匀分布"的显式混合，可解析计算，
        因此修正是精确的而非近似的。
        """
        obs = obs_to_tensors(obs_np, self.device)
        log_probs, value, aux = self.net.log_policy(obs, n_stage)
        probs = log_probs.exp()
        n = probs.shape[0]

        if greedy or epsilon <= 0.0:
            behaviour = probs
        else:
            behaviour = (1.0 - epsilon) * probs + epsilon / n

        if greedy:
            idx = self._greedy_index(obs, probs, aux)
        else:
            idx = int(torch.multinomial(behaviour, 1).item())

        logp = float(torch.log(behaviour[idx].clamp_min(1e-12)).item())
        logp_target = float(torch.log(probs[idx].clamp_min(1e-12)).item())
        ratio_bound = (n / epsilon) if epsilon > 0 else float("inf")
        return idx, logp, float(value.item()), {"n_candidates": n, "ratio_bound": ratio_bound,
                                                "logp_target": logp_target}

    @staticmethod
    def _greedy_index(obs: dict, probs: torch.Tensor, aux: dict) -> int:
        """贪心读出。单体 softmax 取整体 argmax；带等待门控的策略先判"等不等"再在派工行里
        取 argmax——门控是一个二元决策，把 P(hold) 与被 softmax 摊薄的单行派工概率直接比大小
        会系统性偏向等待。"""
        gate_logit = aux.get("gate_logit") if aux else None
        if gate_logit is None:
            return int(torch.argmax(probs).item())
        noop = obs["act_feat"][:, -1] > 0.5
        if float(gate_logit) > 0.0:
            return int(torch.nonzero(noop)[0].item())
        return int(torch.argmax(probs.masked_fill(noop, -1.0)).item())

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {}
        buffer.compute_gae(self.gamma, self.lam)
        adv = torch.as_tensor(buffer.normalized_advantages(), dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(buffer.returns, dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor([t.logp_behaviour for t in buffer.data],
                                   dtype=torch.float32, device=self.device)
        # KL 早停必须以目标策略为基准。用 log b_k 算出的"KL"在第一次梯度步之前就非零
        # （它衡量的是行为混合分布与目标策略的距离），会把 update_epochs 误削成 1。
        logp_target_old = torch.as_tensor([t.logp_target for t in buffer.data],
                                          dtype=torch.float32, device=self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "ratio_max": 0.0, "n_updates": 0.0,
                 "commit_bce": 0.0, "commit_brier": 0.0, "hold_bce": 0.0}
        aux_count = {"commit": 0, "hold": 0}
        # 只数网络真正会用到的标签，P0 这类没有辅助头的配置记 0
        stats["n_labels"] = float(
            (sum(1 for t in buffer.data if t.commit_label >= 0)
             if getattr(self.net, "commit_critic", False) else 0)
            + (sum(1 for t in buffer.data if t.hold_label >= 0)
               if getattr(self.net, "hold_critic", False) else 0))
        n = len(buffer)
        for _ in range(self.epochs):
            order = np.random.permutation(n)
            for start in range(0, n, self.minibatch):
                batch = order[start:start + self.minibatch]
                logps, values, entropies = [], [], []
                commit_terms, hold_terms = [], []
                for i in batch:
                    tr = buffer.data[i]
                    obs = obs_to_tensors(tr.obs, self.device)
                    logp_all, value, aux = self.net.log_policy(obs, tr.n_stage)
                    probs = logp_all.exp()
                    logps.append(logp_all[tr.action_index])
                    values.append(value)
                    entropies.append(-(probs * logp_all).sum())
                    # CoH 辅助监督：只有拿到标签的转移进损失
                    if aux["commit_logit"] is not None and tr.commit_label >= 0:
                        commit_terms.append((aux["commit_logit"][tr.action_index],
                                             float(tr.commit_label)))
                    if aux["hold_logit"] is not None and tr.hold_label >= 0:
                        hold_terms.append((aux["hold_logit"], float(tr.hold_label)))
                logp_new = torch.stack(logps)
                value_new = torch.stack(values)
                entropy = torch.stack(entropies).mean()
                aux_loss, aux_stats = self._aux_losses(commit_terms, hold_terms)

                ratio = torch.exp(logp_new - logp_old[batch])
                surr1 = ratio * adv[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[batch]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_new, ret[batch])
                loss = (self.c1 * policy_loss + self.c2 * value_loss - self.c3 * entropy
                        + self.critic_coeff * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float((logp_target_old[batch] - logp_new).mean().item())
                    stats["policy_loss"] += float(policy_loss.item())
                    stats["value_loss"] += float(value_loss.item())
                    stats["entropy"] += float(entropy.item())
                    stats["approx_kl"] += approx_kl
                    stats["clip_frac"] += float(((ratio - 1).abs() > self.clip).float().mean().item())
                    stats["ratio_max"] = max(stats["ratio_max"], float(ratio.max().item()))
                    stats["n_updates"] += 1.0
                    for key, cnt in (("commit_bce", "commit"), ("commit_brier", "commit"),
                                     ("hold_bce", "hold")):
                        if aux_stats.get(key) is not None:
                            stats[key] += aux_stats[key]
                    aux_count["commit"] += int(aux_stats.get("commit_bce") is not None)
                    aux_count["hold"] += int(aux_stats.get("hold_bce") is not None)
            if stats["n_updates"] and stats["approx_kl"] / stats["n_updates"] > self.target_kl:
                break                                   # 早停，防止越出信任域

        k = max(stats.pop("n_updates"), 1.0)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[key] /= k
        for key, cnt in (("commit_bce", "commit"), ("commit_brier", "commit"), ("hold_bce", "hold")):
            stats[key] = stats[key] / aux_count[cnt] if aux_count[cnt] else float("nan")
        return stats

    def _aux_losses(self, commit_terms, hold_terms):
        """CoH 的两个 BCE 辅助损失；没有标签的 minibatch 返回 0 与空统计。"""
        loss = torch.zeros((), device=self.device)
        out = {}
        if commit_terms:
            logit = torch.stack([t for t, _ in commit_terms])
            y = torch.tensor([y for _, y in commit_terms], dtype=torch.float32, device=self.device)
            bce = F.binary_cross_entropy_with_logits(logit, y)
            loss = loss + bce
            out["commit_bce"] = float(bce.item())
            out["commit_brier"] = float(((torch.sigmoid(logit) - y) ** 2).mean().item())
        if hold_terms:
            logit = torch.stack([t for t, _ in hold_terms])
            y = torch.tensor([y for _, y in hold_terms], dtype=torch.float32, device=self.device)
            bce = F.binary_cross_entropy_with_logits(logit, y)
            loss = loss + bce
            out["hold_bce"] = float(bce.item())
        return loss, out
