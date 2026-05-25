"""
PPO代理类
"""
try:
    from agent.HGHH_model import HGNNScheduler
except ImportError:
    from HGHH_model import HGNNScheduler
import copy
import numpy as np
import torch
import torch.nn.functional as F

class PPO:
    def __init__(self, model_paras, train_paras):
        self.lr = train_paras["lr"]  # 学习率
        self.current_lr = float(self.lr)
        self.betas = train_paras["betas"]  # Adam 参数
        self.gamma = train_paras["gamma"]  # 折扣因子
        self.eps_clip = train_paras["eps_clip"]  # PPO 截断比例
        self.K_epochs = train_paras["K_epochs"]  # 更新轮次
        self.A_coeff = train_paras["A_coeff"]  # 策略损失系数
        self.vf_coeff = train_paras["vf_coeff"]  # 状态价值损失系数
        self.entropy_coeff = train_paras["entropy_coeff"]  # 熵正则系数
        self.max_grad_norm = float(train_paras.get("max_grad_norm", 0.5))
        self.target_kl = train_paras.get("target_kl", None)
        self.target_kl = None if self.target_kl in (None, 0) else float(self.target_kl)
        self.device = model_paras["device"]

        # 初始化策略网络
        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

    def set_learning_rate(self, lr: float):
        self.current_lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

    def update_rollout(self, buffer: object) -> dict:
        device = self.device
        if len(buffer) == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }

        returns = torch.tensor(buffer.returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-5)
        old_logprobs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device).detach()
        actions = torch.tensor(buffer.action_indexes, dtype=torch.long, device=device)
        env_actions = getattr(buffer, "env_actions", None)
        minibatch_size = max(int(getattr(buffer, "minibatch_size", len(buffer))), 1)
        buffer.minibatch_size = minibatch_size
        loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_fraction_sum = 0.0
        update_count = 0
        stop_early = False

        for _ in range(self.K_epochs):
            minibatches = (buffer.iter_minibatches(shuffle=True)
                           if hasattr(buffer, "iter_minibatches")
                           else [np.arange(len(buffer))])
            for batch_indices in minibatches:
                obs_batch = [buffer.obs[i] for i in batch_indices]
                batch_actions = ([env_actions[i] for i in batch_indices]
                                 if env_actions is not None else actions[batch_indices])
                logprobs, state_values, entropies = self.policy.evaluate_batch(obs_batch, batch_actions)
                log_ratio = logprobs - old_logprobs[batch_indices]
                ratios = torch.exp(log_ratio)
                batch_advantages = advantages[batch_indices]
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(state_values, returns[batch_indices])
                entropy = entropies.mean()
                approx_kl = ((ratios - 1.0) - log_ratio).mean()
                clip_fraction = ((ratios - 1.0).abs() > self.eps_clip).float().mean()
                loss = (
                    self.A_coeff * policy_loss
                    + self.vf_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                loss_sum += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()
                approx_kl_sum += approx_kl.item()
                clip_fraction_sum += clip_fraction.item()
                update_count += 1
                if self.target_kl is not None and approx_kl.item() > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.policy_old.load_state_dict(self.policy.state_dict())
        denom = max(update_count, 1)
        return {
            "loss": loss_sum / denom,
            "policy_loss": policy_loss_sum / denom,
            "value_loss": value_loss_sum / denom,
            "entropy": entropy_sum / denom,
            "approx_kl": approx_kl_sum / denom,
            "clip_fraction": clip_fraction_sum / denom,
        }
