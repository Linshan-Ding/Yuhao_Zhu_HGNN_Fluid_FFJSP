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
        logits, value = self.net(obs, n_stage)
        probs = torch.softmax(logits, dim=-1)
        n = probs.shape[0]

        if greedy or epsilon <= 0.0:
            behaviour = probs
        else:
            behaviour = (1.0 - epsilon) * probs + epsilon / n

        if greedy:
            idx = int(torch.argmax(probs).item())
        else:
            idx = int(torch.multinomial(behaviour, 1).item())

        logp = float(torch.log(behaviour[idx].clamp_min(1e-12)).item())
        logp_target = float(torch.log(probs[idx].clamp_min(1e-12)).item())
        ratio_bound = (n / epsilon) if epsilon > 0 else float("inf")
        return idx, logp, float(value.item()), {"n_candidates": n, "ratio_bound": ratio_bound,
                                                "logp_target": logp_target}

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
                 "approx_kl": 0.0, "clip_frac": 0.0, "ratio_max": 0.0, "n_updates": 0.0}
        n = len(buffer)
        for _ in range(self.epochs):
            order = np.random.permutation(n)
            for start in range(0, n, self.minibatch):
                batch = order[start:start + self.minibatch]
                logps, values, entropies = [], [], []
                for i in batch:
                    tr = buffer.data[i]
                    obs = obs_to_tensors(tr.obs, self.device)
                    logits, value = self.net(obs, tr.n_stage)
                    logp_all = torch.log_softmax(logits, dim=-1)
                    probs = logp_all.exp()
                    logps.append(logp_all[tr.action_index])
                    values.append(value)
                    entropies.append(-(probs * logp_all).sum())
                logp_new = torch.stack(logps)
                value_new = torch.stack(values)
                entropy = torch.stack(entropies).mean()

                ratio = torch.exp(logp_new - logp_old[batch])
                surr1 = ratio * adv[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[batch]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_new, ret[batch])
                loss = self.c1 * policy_loss + self.c2 * value_loss - self.c3 * entropy

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
            if stats["n_updates"] and stats["approx_kl"] / stats["n_updates"] > self.target_kl:
                break                                   # 早停，防止越出信任域

        k = max(stats.pop("n_updates"), 1.0)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[key] /= k
        return stats
