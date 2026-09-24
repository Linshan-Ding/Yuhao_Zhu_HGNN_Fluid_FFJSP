"""价值型基线的学习器：三元组 Double DQN 与分层 Double DQN，共用 replay、n 步回报与目标网络。

每个 epoch 把新采的 episode 放进 replay，做 replay_ratio x 新转移数 步梯度；目标为
R^(n) + Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))，episode 在 t+n 前结束则 bootstrap 为 0。
分层版本上层在规则 + 保留产能上、下层在 t+n 时实际选中的窗口内取 argmax（SARSA 式近似）。
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from agent.batch import NEG, collate_records, to_batch
from agent.buffer import EpisodeRecord, ReplayStore


class DQNLearner:
    hier = False

    def __init__(self, net, cfg, device: torch.device, total_steps: int) -> None:
        self.net = net.to(device)
        self.target = copy.deepcopy(self.net).to(device).eval()
        self.device = device
        self.total_steps = int(total_steps)
        d = cfg.get("dqn")
        self.lr = float(d["lr"])
        self.minibatch = int(d["minibatch_size"])
        self.learning_starts = int(d["learning_starts"])
        self.replay_ratio = float(d["replay_ratio"])
        self.n_step = int(d["n_step"])
        self.target_update = int(d["target_update_steps"])
        self.eps_start, self.eps_end = float(d["epsilon_start"]), float(d["epsilon_end"])
        self.eps_fraction = float(d["epsilon_fraction"])
        self.max_grad_norm = float(d["max_grad_norm"])
        self.replay = ReplayStore(int(d["replay_capacity"]))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.rng = np.random.default_rng(0)
        self.grad_steps = 0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def policy_state_numpy(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def behaviour_params(self, step: int) -> dict:
        frac = min(step / max(self.eps_fraction * self.total_steps, 1.0), 1.0)
        return {"epsilon": self.eps_start + (self.eps_end - self.eps_start) * frac}

    def state_dict(self) -> dict:
        return {"model": self.net.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.opt.state_dict(), "grad_steps": self.grad_steps}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])
        self.opt.load_state_dict(state["optimizer"])
        self.grad_steps = int(state["grad_steps"])

    def lr_at(self, step: int) -> float:
        return self.lr

    # ---- 更新
    def _targets(self, records, pairs) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
        """n 步回报、bootstrap 的 (e, t+n) 对与是否终止。"""
        rewards = np.zeros(len(pairs), np.float32)
        boot_pairs, terminal = [], np.zeros(len(pairs), bool)
        for i, (e, t) in enumerate(pairs):
            rec = records[e]
            T = len(rec)
            end = min(t + self.n_step, T)
            rewards[i] = rec.reward[t:end].sum()
            if t + self.n_step < T:
                boot_pairs.append((e, t + self.n_step))
            else:
                boot_pairs.append((e, T - 1))
                terminal[i] = True
        return rewards, boot_pairs, terminal

    def update(self, episodes: Sequence[EpisodeRecord], step: int) -> Dict[str, float]:
        self.replay.add(episodes)
        new_steps = sum(len(r) for r in episodes)
        stats: Dict[str, List[float]] = {}
        if step < self.learning_starts:
            return {"lr": self.lr, "n_updates": 0.0, "replay_size": float(self.replay.size)}
        n_updates = max(1, int(round(self.replay_ratio * new_steps)))
        records = self.replay.records
        for _ in range(n_updates):
            pairs = self.replay.sample(self.minibatch, self.rng)
            rewards, boot_pairs, terminal = self._targets(records, pairs)
            b = to_batch(collate_records(records, pairs), self.device)
            b_next = to_batch(collate_records(records, boot_pairs), self.device)
            loss, st = self._loss(records, pairs, b, b_next, rewards, terminal)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.opt.step()
            self.grad_steps += 1
            if self.grad_steps % self.target_update == 0:
                self.target.load_state_dict(self.net.state_dict())
            for k, v in st.items():
                stats.setdefault(k, []).append(v)
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out.update({"lr": self.lr, "n_updates": float(n_updates), "replay_size": float(self.replay.size)})
        return out

    def _loss(self, records, pairs, b, b_next, rewards, terminal):
        dev = self.device
        action = torch.as_tensor([int(records[e].env_action[t]) for e, t in pairs], device=dev)
        r = torch.from_numpy(rewards).to(dev)
        not_done = torch.from_numpy(~terminal).float().to(dev)
        q_sa = self.net(b).q.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.net(b_next).q.argmax(-1)
            q_next = self.target(b_next).q.gather(1, a_star.unsqueeze(1)).squeeze(1)
            q_next = torch.where(torch.isfinite(q_next), q_next, torch.zeros_like(q_next))
            target = r + not_done * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        return loss, {"q_loss": loss.item(), "q_mean": q_sa.mean().item()}


class HierDQNLearner(DQNLearner):
    hier = True

    def _loss(self, records, pairs, b, b_next, rewards, terminal):
        dev = self.device
        action = torch.as_tensor([int(records[e].env_action[t]) for e, t in pairs], device=dev)
        rule = torch.as_tensor([int(records[e].extras["rule_action"][t]) for e, t in pairs], device=dev)
        r = torch.from_numpy(rewards).to(dev)
        not_done = torch.from_numpy(~terminal).float().to(dev)
        out = self.net(b)
        q_u = out.q_upper.gather(1, rule.unsqueeze(1)).squeeze(1)
        q_l = out.q.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt_online, nxt_target = self.net(b_next), self.target(b_next)
            r_star = nxt_online.q_upper.argmax(-1)
            qu_next = nxt_target.q_upper.gather(1, r_star.unsqueeze(1)).squeeze(1)
            window = b_next.extras["window_mask"]
            ql_online = nxt_online.q.masked_fill(~window, NEG)
            a_star = ql_online.argmax(-1)
            ql_next = nxt_target.q.gather(1, a_star.unsqueeze(1)).squeeze(1)
            qu_next = torch.where(torch.isfinite(qu_next), qu_next, torch.zeros_like(qu_next))
            ql_next = torch.where(torch.isfinite(ql_next), ql_next, torch.zeros_like(ql_next))
            t_u = r + not_done * qu_next
            t_l = r + not_done * ql_next
        loss = F.smooth_l1_loss(q_u, t_u) + F.smooth_l1_loss(q_l, t_l)
        return loss, {"q_loss": loss.item(), "q_mean": q_l.mean().item(), "q_upper_mean": q_u.mean().item()}
