"""
PPO代理类
"""
from HGHH_model import HGNNScheduler
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Any

class PPO:
    def __init__(self, model_paras, train_paras):
        self.lr = train_paras["lr"]  # 学习率
        self.betas = train_paras["betas"]  # Adam 参数
        self.gamma = train_paras["gamma"]  # 折扣因子
        self.eps_clip = train_paras["eps_clip"]  # PPO 截断比例
        self.K_epochs = train_paras["K_epochs"]  # 更新轮次
        self.A_coeff = train_paras["A_coeff"]  # 策略损失系数
        self.vf_coeff = train_paras["vf_coeff"]  # 状态价值损失系数
        self.entropy_coeff = train_paras["entropy_coeff"]  # 熵正则系数
        self.device = model_paras["device"]

        # 初始化策略网络
        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory: object) -> Tuple[float, float]:
        """
        更新策略模型。
        memory 中保存单实例收集的以下数据：
         - ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
           ope_step, eligible, logprobs, rewards, is_terminals, action_indexes
        本函数将所有内存数据拉平成单个序列进行训练（无需拆分批量）。
        返回平均更新 loss 与平均折扣奖励（仅作参考）。
        """
        device = self.device

        # 提取 memory 中各数据（均已为单实例，不含 batch 维度）
        old_states = memory.states
        old_ope_ma_adj = memory.ope_ma_adj
        old_ope_pre_adj = memory.ope_pre_adj
        old_ope_sub_adj = memory.ope_sub_adj
        old_raw_opes = memory.raw_opes
        old_raw_mas = memory.raw_mas
        old_proc_time = memory.proc_time
        old_eligible = memory.eligible
        old_logprobs = torch.tensor(memory.log_probs, device=device).detach()
        old_action_env = memory.action_indexes

        # 计算折扣奖励（Backward rolled rewards）
        discounted_rewards = []
        discounted_reward = 0.0
        for reward, terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        rewards_tensor = (rewards_tensor - rewards_tensor.min())/(rewards_tensor.max() - rewards_tensor.min() + 1e-5)  #
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-5)

        loss_sum = 0.0

        # PPO 更新：对所有数据进行 K_epochs 轮更新（单实例模式下无需 mini-batch切分）
        for _ in range(self.K_epochs):
            # 重新评估所有状态-动作对
            action_logprobs = []
            state_values = []
            dist_entropys = []
            for i in range(len(old_states)):
                action_logprob, state_value, dist_entropy = self.policy.evaluate(old_states[i],
                                                                                old_ope_ma_adj[i],
                                                                                old_ope_pre_adj[i],
                                                                                old_ope_sub_adj[i],
                                                                                old_raw_opes[i],
                                                                                old_raw_mas[i],
                                                                                old_proc_time[i],
                                                                                old_eligible[i],
                                                                                old_action_env[i])
                action_logprobs.append(action_logprob)
                state_values.append(state_value)
                dist_entropys.append(dist_entropy)
            # 转换为张量
            action_logprobs_tensor = torch.stack(action_logprobs).squeeze()
            state_values_tensor = torch.stack(state_values)
            dist_entropys_tensor = torch.stack(dist_entropys).squeeze()
            # 计算概率比
            ratios = torch.exp(action_logprobs_tensor - old_logprobs)
            advantages = rewards_tensor - state_values_tensor.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = (- self.A_coeff * torch.min(surr1, surr2)
                    + self.vf_coeff * self.MseLoss(state_values_tensor, rewards_tensor)
                    - self.entropy_coeff * dist_entropys_tensor).mean()
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 将更新后的 policy 同步到 policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_sum