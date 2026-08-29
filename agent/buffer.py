"""Rollout 缓存与 GAE。

关键点（稿件 §4.8.2）：存进缓存的是**行为策略** log b_k(a|omega)，不是目标策略
log pi_old。epsilon-贪婪探索下二者不同，用后者会让 PPO 的重要性比率算错、
裁剪区间失去它本该表示的信任域含义。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class Transition:
    obs: dict
    action_index: int
    logp_behaviour: float          # log b_k(a_t | omega_t)，用于重要性比率
    logp_target: float             # log pi_theta_old(a_t | omega_t)，用于 KL 早停判据
    reward: float
    value: float
    done: bool
    n_candidates: int
    n_stage: int


@dataclass
class RolloutBuffer:
    data: List[Transition] = field(default_factory=list)
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(self, **kwargs) -> None:
        self.data.append(Transition(**kwargs))

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()
        self.advantages = None
        self.returns = None

    def compute_gae(self, gamma: float, lam: float) -> None:
        n = len(self.data)
        adv = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for i in reversed(range(n)):
            nonterminal = 0.0 if self.data[i].done else 1.0
            next_value = 0.0 if i + 1 >= n or self.data[i].done else self.data[i + 1].value
            delta = self.data[i].reward + gamma * next_value * nonterminal - self.data[i].value
            gae = delta + gamma * lam * nonterminal * gae
            adv[i] = gae
        self.advantages = adv
        self.returns = adv + np.asarray([t.value for t in self.data], dtype=np.float64)

    def normalized_advantages(self) -> np.ndarray:
        adv = self.advantages
        if adv is None or adv.size <= 1:
            return np.zeros_like(adv) if adv is not None else np.zeros(0)
        return (adv - adv.mean()) / (adv.std() + 1e-8)
