"""批量 PPO（主方法与规则选择基线共用）。

每个 epoch 把全部 episode 填充成一个大批一次搬上设备，minibatch 用切片取子批；GAE 逐 episode
在 numpy 里算；KL 用 k3 估计 (r-1) - log r（非负、低方差），每个 update-epoch 的均值超过
target_kl 即停，单 minibatch 超 4 倍硬停。承诺评估器与等待前景评估器的 BCE 只对有标签的转移计算。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from agent.batch import collate_records, to_batch
from agent.buffer import EpisodeRecord, gae


class PPOLearner:
    def __init__(self, net, cfg, device: torch.device, total_steps: int) -> None:
        self.net = net.to(device)
        self.device = device
        self.total_steps = int(total_steps)
        p = cfg.get("ppo")
        self.lr0 = float(p["lr"])
        self.lr_final_ratio = float(p.get("lr_final_ratio", 1.0))
        self.lam = float(p["gae_lambda"])
        self.clip = float(p["clip_eps"])
        self.update_epochs = int(p["update_epochs"])
        self.c_ent = float(p["entropy_coeff"])
        self.c_val = float(p["value_coeff"])
        self.max_grad_norm = float(p["max_grad_norm"])
        self.target_kl = float(p["target_kl"])
        self.minibatch = int(p["minibatch_size"])
        self.minibatch_large = int(p.get("minibatch_size_large", self.minibatch))
        self.critic_coeff = float(cfg.get("coh.critic_coeff", 0.5))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr0)
        self.gen = torch.Generator(device="cpu")

    # ---- 与 worker / checkpoint 的接口
    def seed(self, seed: int) -> None:
        self.gen.manual_seed(int(seed))

    def policy_state_numpy(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def behaviour_params(self, step: int) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {"model": self.net.state_dict(), "optimizer": self.opt.state_dict(),
                "generator": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.gen.set_state(state["generator"])

    def lr_at(self, step: int) -> float:
        frac = min(max(step / max(self.total_steps, 1), 0.0), 1.0)
        return self.lr0 * (1.0 - (1.0 - self.lr_final_ratio) * frac)

    # ---- 更新
    def update(self, episodes: Sequence[EpisodeRecord], step: int) -> Dict[str, float]:
        adv_list, ret_list = zip(*(gae(rec, 1.0, self.lam) for rec in episodes))
        adv = np.concatenate(adv_list)
        ret = np.concatenate(ret_list)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        arrays = collate_records(episodes)
        batch = to_batch(arrays, self.device)
        n = batch.size
        dev = self.device
        action = torch.from_numpy(np.concatenate([r.action for r in episodes])).to(dev)
        logp_old = torch.from_numpy(np.concatenate([r.logp for r in episodes])).to(dev)
        commit_label = torch.from_numpy(np.concatenate([r.commit_label for r in episodes]).astype(np.int64)).to(dev)
        hold_label = torch.from_numpy(np.concatenate([r.hold_label for r in episodes]).astype(np.int64)).to(dev)
        adv_t = torch.from_numpy(adv.astype(np.float32)).to(dev)
        ret_t = torch.from_numpy(ret.astype(np.float32)).to(dev)

        lr = self.lr_at(step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        mb = self.minibatch_large if (dev.type == "cuda" and n >= 64000) else self.minibatch
        acc: Dict[str, List[float]] = {}
        ratio_max = 0.0
        stop = False
        epochs_done = 0
        for k in range(self.update_epochs):
            perm = torch.randperm(n, generator=self.gen).to(dev)
            kls = []
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                sub = batch.subset(idx)
                out = self.net(sub)
                logp_new = out.logp.gather(1, action[idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(logp_new - logp_old[idx])
                a_mb = adv_t[idx]
                pg = -torch.min(ratio * a_mb, ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * a_mb).mean()
                vl = F.mse_loss(out.value, ret_t[idx])
                ent = out.entropy.mean()
                aux, aux_stats = self._aux_losses(out, action[idx], commit_label[idx], hold_label[idx])
                loss = pg + self.c_val * vl - self.c_ent * ent + self.critic_coeff * aux
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-12))).mean()
                    stats = {"policy_loss": pg.item(), "value_loss": vl.item(), "entropy": ent.item(),
                             "approx_kl": kl.item(),
                             "clip_frac": ((ratio - 1.0).abs() > self.clip).float().mean().item()}
                    if out.gate_logit is not None and bool(out.has_noop.any()):
                        stats["p_hold"] = torch.sigmoid(out.gate_logit[out.has_noop]).mean().item()
                    stats.update(aux_stats)
                    ratio_max = max(ratio_max, ratio.max().item())
                    for key, val in stats.items():
                        acc.setdefault(key, []).append(float(val))
                    kls.append(kl.item())
                    if kl.item() > 4.0 * self.target_kl:
                        stop = True
                        break
            epochs_done = k + 1
            if stop or (kls and float(np.mean(kls)) > self.target_kl):
                break
        out_stats = {key: float(np.mean(v)) for key, v in acc.items()}
        out_stats.update({"ratio_max": ratio_max, "update_epochs_done": epochs_done, "lr": lr,
                          "n_labels": float(int((commit_label >= 0).sum()) if self._has("commit_critic") else 0)
                          + float(int((hold_label >= 0).sum()) if self._has("hold_critic") else 0)})
        return out_stats

    def _has(self, flag: str) -> bool:
        return bool(getattr(self.net, flag, False))

    def _aux_losses(self, out, action, commit_label, hold_label):
        loss = torch.zeros((), device=self.device)
        stats = {}
        if out.commit_logit is not None:
            lg = out.commit_logit.gather(1, action.unsqueeze(1)).squeeze(1)
            m = commit_label >= 0
            if bool(m.any()):
                y = commit_label[m].float()
                bce = F.binary_cross_entropy_with_logits(lg[m], y)
                loss = loss + bce
                stats["commit_bce"] = bce.item()
                stats["commit_brier"] = ((torch.sigmoid(lg[m]) - y) ** 2).mean().item()
        if out.hold_logit is not None:
            m = hold_label >= 0
            if bool(m.any()):
                bce = F.binary_cross_entropy_with_logits(out.hold_logit[m], hold_label[m].float())
                loss = loss + bce
                stats["hold_bce"] = bce.item()
        return loss, stats
