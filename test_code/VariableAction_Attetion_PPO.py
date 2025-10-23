import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random

# ===========================================================
# 1. 简易环境定义（可替换为自己的任务）
# ===========================================================
class VariableActionEnv:
    """变动作数量环境"""
    def __init__(self, state_dim=16, cand_dim=8, max_candidates=10, max_steps=20):
        self.state_dim = state_dim
        self.cand_dim = cand_dim
        self.max_candidates = max_candidates
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        state = np.random.randn(self.state_dim).astype(np.float32)
        return state

    def get_candidates(self):
        n = random.randint(3, self.max_candidates)  # 候选数量变化
        cands = np.random.randn(n, self.cand_dim).astype(np.float32)
        mask = np.ones(n, dtype=np.float32)
        return cands, mask

    def step(self, action_idx):
        """根据动作返回奖励"""
        reward = random.random() * 2 - 1  # [-1, 1]
        self.step_count += 1
        done = self.step_count >= self.max_steps
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        return next_state, reward, done


# ===========================================================
# 2. Self-Attention 策略网络（可变动作数量）
# ===========================================================
class AttentionPolicy(nn.Module):
    def __init__(self, state_dim, cand_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.cand_proj = nn.Linear(cand_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, cands, mask):
        """
        state: [B, state_dim]
        cands: [B, K, cand_dim]
        mask:  [B, K]  1=valid, 0=pad
        """
        B, K, _ = cands.shape
        state_embed = self.state_proj(state).unsqueeze(1)  # [B,1,H]
        cand_embed = self.cand_proj(cands)  # [B,K,H]

        # TransformerEncoder 处理候选间关系
        attn_mask = mask == 0
        encoded = self.encoder(cand_embed, src_key_padding_mask=attn_mask)  # [B,K,H]

        logits = self.scorer(encoded).squeeze(-1)  # [B,K]
        very_neg = -1e9
        masked_logits = logits + (1.0 - mask) * very_neg

        probs = F.softmax(masked_logits, dim=-1)  # [B,K]
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # 值函数使用 pooled 表示
        global_repr = torch.cat([state, state_embed.squeeze(1)], dim=-1)
        value = self.value_net(global_repr).squeeze(-1)
        return action, log_prob, entropy, value, probs

    def evaluate_actions(self, state, cands, mask, actions):
        """PPO 更新时重新计算 log_prob、熵、值"""
        B, K, _ = cands.shape
        state_embed = self.state_proj(state).unsqueeze(1)
        cand_embed = self.cand_proj(cands)
        attn_mask = mask == 0
        encoded = self.encoder(cand_embed, src_key_padding_mask=attn_mask)
        logits = self.scorer(encoded).squeeze(-1)
        masked_logits = logits + (1.0 - mask) * -1e9
        probs = F.softmax(masked_logits, dim=-1)
        probs = probs * mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value_net(torch.cat([state, state_embed.squeeze(1)], dim=-1)).squeeze(-1)
        return log_probs, entropy, value


# ===========================================================
# 3. PPO Agent 实现
# ===========================================================
class PPOAgent:
    def __init__(self, state_dim, cand_dim, device='cpu'):
        self.policy = AttentionPolicy(state_dim, cand_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.device = device
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01

    def select_action(self, state, cands, mask):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        cands = torch.tensor(cands, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value, _ = self.policy(state, cands, mask)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, dones, values):
        advantages = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        returns = [a + v for a, v in zip(advantages, values)]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, batch):
        states, cands, masks, actions, old_logps, returns, advantages = batch
        states = torch.stack(states).to(self.device)
        cands = torch.nn.utils.rnn.pad_sequence(cands, batch_first=True).to(self.device)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        logps, entropies, values = self.policy.evaluate_actions(states, cands, masks, actions)
        ratios = torch.exp(logps - old_logps)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = entropies.mean()

        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


# ===========================================================
# 4. 训练主程序
# ===========================================================
def train():
    env = VariableActionEnv()
    agent = PPOAgent(state_dim=16, cand_dim=8)
    max_episodes = 200

    for ep in range(max_episodes):
        state = env.reset()
        logps, rewards, values, dones = [], [], [], []
        states, cands_list, masks, actions = [], [], [], []

        done = False
        while not done:
            cands, mask = env.get_candidates()
            action, logp, value = agent.select_action(state, cands, mask)
            next_state, reward, done = env.step(action)

            states.append(torch.tensor(state, dtype=torch.float32))
            cands_list.append(torch.tensor(cands, dtype=torch.float32))
            masks.append(torch.tensor(mask, dtype=torch.float32))
            actions.append(action)
            logps.append(logp)
            rewards.append(reward)
            values.append(value)
            dones.append(float(done))
            state = next_state

        advantages, returns = agent.compute_gae(rewards, dones, values)
        batch = (states, cands_list, masks, actions, logps, returns, advantages)
        loss, al, cl, el = agent.update(batch)
        print(f"Episode {ep:03d} | Loss={loss:.3f} | Actor={al:.3f} | Critic={cl:.3f} | Entropy={el:.3f}")


if __name__ == "__main__":
    train()
