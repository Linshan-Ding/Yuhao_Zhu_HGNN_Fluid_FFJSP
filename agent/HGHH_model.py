"""
训练算法：异质神经网络定义、策略网络定义、值网络定义、PPO算法
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
try:
    from agent.hgnn import GATedge, MLPsim
    from agent.mlp import MLPCritic, MLPActor
except ImportError:
    from hgnn import GATedge, MLPsim
    from mlp import MLPCritic, MLPActor
from typing import Tuple, List
import random


class Memory:
    def __init__(self):
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.eligible = []

    def clear_memory(self):
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.eligible[:]

class MLPs(nn.Module):
    """
    工序类型节点嵌入
    单个多层感知机模块，用于计算操作节点嵌入。
    输入特征经过 MLP 滤波后，沿着边的邻接矩阵聚合信息。
    """
    def __init__(self,
                 W_sizes_ope: List[int],
                 hidden_size_ope: int,
                 out_size_ope: int,
                 num_head: int,
                 dropout: float) -> None:
        """
        参数：
        - W_sizes_ope: 输入向量尺寸的列表，每个元素对应不同输入类型，
        例如 [machine, operation(pre), operation(sub), operation(self)]
        - hidden_size_ope: MLP 隐藏层维度
        - out_size_ope: 工序节点最终嵌入的维度
        - num_head: 注意力头个数（此处仅作为模块参数传入）
        - dropout: dropout 率
        """
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout

        # 构造多个 MLPsim 模块，每个针对不同输入
        self.gnn_layers = nn.ModuleList([
            MLPsim(in_size, self.out_size_ope, self.hidden_size_ope, self.num_head, self.dropout, self.dropout)
            for in_size in self.in_sizes_ope
        ])

        # 聚合层，将各个分支拼接后映射到目标空间
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self,
                ope_ma_adj: torch.Tensor,
                ope_pre_adj: torch.Tensor,
                ope_sub_adj: torch.Tensor,
                feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        参数：
         - ope_ma_adj: 工序-机器邻接矩阵，形状 [num_opes, num_mas]
         - ope_pre_adj: 工序-前序邻接矩阵，形状 [num_opes, num_opes]
         - ope_sub_adj: 工序-后序邻接矩阵，形状 [num_opes, num_opes]
         - feats: 包含工序类型、机器和边特征，格式为 (feat_ope, feat_ma, feat_edge)
        其中，feat_ope 的形状为 [num_opes, in_size_ope]，
           feat_ma  为 [num_mas, *]（具体尺寸取决于上游模块）；
           feat_edge 用于边上信息（例如机器对工序的指标）。
        返回：
         - 工序节点嵌入，形状 [num_opes, out_size_ope]
        """
        # 构造不同分支的输入，按照顺序：
        #  branch0: 来自机器节点嵌入（注意：尺寸需要与模块输入相匹配）
        #  branch1/2/3: 均使用工序节点原始特征
        h: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (feats[1], feats[0], feats[0], feats[0])
        # 构造自环邻接矩阵：对工序节点构造单位阵
        num_opes = feats[0].size(-2)
        self_adj: torch.Tensor = torch.eye(num_opes, dtype=torch.int64, device=feats[0].device).expand_as(ope_pre_adj)

        # 汇总各个邻接矩阵，构造元组：此处采用顺序 [ope_ma_adj, ope_pre_adj, ope_sub_adj, self_adj]
        adj_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
            ope_ma_adj, ope_pre_adj, ope_sub_adj, self_adj
        )
        mlp_embeddings: List[torch.Tensor] = []
        for i, adj in enumerate(adj_tuple):
            mlp_embeddings.append(self.gnn_layers[i](h[i], adj))
        # 拼接多分支输出
        mlp_embedding_in = torch.cat(mlp_embeddings, dim=-1)
        # 聚合后输出最终工序嵌入
        mu_ij_prime = self.project(mlp_embedding_in)
        return mu_ij_prime


class HGNNScheduler(nn.Module):
    """
    HGNN 调度器，构造机器节点与工序节点嵌入模块。
    并利用 actor-critic 网络对 O-M 对（操作与机器）进行决策。
    注意：本实现为单算例版，每次处理一个实例。
    """
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # 机器节点原始特征尺寸
        self.out_size_ma = model_paras["out_size_ma"]  # 机器节点嵌入尺寸
        self.in_size_ope = model_paras["in_size_ope"]  # 工序节点原始特征尺寸
        self.out_size_ope = model_paras["out_size_ope"]  # 工序节点嵌入尺寸
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # 工序 MLP 隐藏层尺寸
        self.actor_dim = model_paras["actor_in_dim"]  # Actor 输入尺寸
        self.critic_dim = model_paras["critic_in_dim"]  # Critic 输入尺寸
        self.n_latent_actor = model_paras["n_latent_actor"]  # Actor 隐藏维度
        self.n_latent_critic = model_paras["n_latent_critic"]  # Critic 隐藏维度
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Actor 层数
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Critic 层数
        self.action_dim = model_paras["action_dim"]  # Actor 输出尺寸
        self.num_heads: List[int] = model_paras["num_heads"]  # 注意力头数量（列表，各阶段使用）
        self.dropout = model_paras["dropout"]
        self.device = model_paras["device"]
        self.epsilon = model_paras["epsilon"]
        self.current_action_dict = None # 添加订单维度后的动作字典

        # 构造机器节点嵌入模块（多层图注意力网络，此处直接使用前文已有的 GATedge 模块）
        # 由于本示例为单算例，输入数据尺寸不再包含 batch 维度
        # 第一阶段：输入尺寸为 (in_size_ope, in_size_ma)
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma),
                                         self.out_size_ma, self.num_heads[0],
                                         feat_drop=self.dropout,
                                         attn_drop=self.dropout,
                                         activation=F.elu))
        # 后续阶段采用前阶段输出更新
        for i in range(1, len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma),
                                             self.out_size_ma, self.num_heads[i],
                                             feat_drop=self.dropout,
                                             attn_drop=self.dropout,
                                             activation=F.elu))
        # 构造工序类型节点嵌入模块
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))
        # 构造 actor 与 critic 网络
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim + 2, self.n_latent_actor, self.action_dim).to(self.device)
        self.order_actor = MLPActor(self.n_hidden_actor, self.actor_dim + 2, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim + 1, self.n_latent_critic, 1).to(self.device)
        self.use_sparse_attention = bool(model_paras.get("use_sparse_attention", False))
        self.sparse_attention_scope = str(model_paras.get("sparse_attention_scope", "pair")).lower()
        if self.sparse_attention_scope not in {"pair", "pair_order"}:
            self.sparse_attention_scope = "pair"
        sparse_feature_dim = self.actor_dim + 2
        sparse_heads = max(int(model_paras.get("sparse_attention_heads", 1)), 1)
        if sparse_feature_dim % sparse_heads != 0:
            sparse_heads = 1
        sparse_dropout = float(model_paras.get("sparse_attention_dropout", 0.0))
        self.sparse_pair_attention = None
        self.sparse_pair_norm = None
        self.sparse_order_attention = None
        self.sparse_order_norm = None
        if self.use_sparse_attention:
            self.sparse_pair_attention = nn.MultiheadAttention(
                sparse_feature_dim,
                sparse_heads,
                dropout=sparse_dropout,
                batch_first=True,
            ).to(self.device)
            self.sparse_pair_norm = nn.LayerNorm(sparse_feature_dim).to(self.device)
            if self.sparse_attention_scope == "pair_order":
                self.sparse_order_attention = nn.MultiheadAttention(
                    sparse_feature_dim,
                    sparse_heads,
                    dropout=sparse_dropout,
                    batch_first=True,
                ).to(self.device)
                self.sparse_order_norm = nn.LayerNorm(sparse_feature_dim).to(self.device)
        self.last_forward_stats = {}

    def forward(self) -> None:
        """
        本模块仅提供 act 与 evaluate 函数，forward 接口不使用。
        """
        raise NotImplementedError("Use act() or evaluate() functions instead.")

    def feature_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """对单个实例特征归一化"""
        return (data - torch.mean(data)) / (torch.std(data, unbiased=False) + 1e-5)

    def get_normalized(self,
                       raw_opes: torch.Tensor,
                       raw_mas: torch.Tensor,
                       proc_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对原始工序节点、机器节点和加工时长特征值进行标准化。
        假设尺寸分别为：
          - raw_opes: [batch, num_opes, in_size_ope]
          - raw_mas: [batch, num_mas, in_size_ma]
          - proc_time: [batch, num_opes, num_mas]
        返回标准化后的特征元组。
        """
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # 计算工序特征均值
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True, unbiased=False)  # 计算工序特征标准差
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # 计算机器特征均值
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True, unbiased=False)  # 计算机器特征标准差
        # 对加工时长做归一化（按所有元素）
        proc_time_norm = self.feature_normalize(proc_time)  # 标准化加工时长
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-5)  # 标准化工序特征
        norm_mas = (raw_mas - mean_mas) / (std_mas + 1e-5)  # 标准化机器特征
        return norm_opes, norm_mas, proc_time_norm

    def get_action_prob(self, state, memory, flag_train=True):
        """
        根据状态数据计算各候选 O-M 对的选择概率。
        state 应包含以下字段：
          - feat_opes: [num_opes, in_size_ope]
          - feat_mas: [num_mas, in_size_ma]
          - proc_times: [num_opes, num_mas]
          - nums_opes: 工序节点个数（int）
          - ope_step: 当前待处理的工序索引 [num_opes]（或其他标识信息）
          - ope_ma_adj: 工序-机器邻接矩阵 [num_opes, num_mas]
          - ope_pre_adj: 工序-前序邻接矩阵 [num_opes, num_opes]
          - ope_sub_adj: 工序-后序邻接矩阵 [num_opes, num_opes]
          - mask_ma_procing: 机器处理状态掩码 [num_mas]（False 表示空闲）
          - mask_job_procing & mask_job_finish: 作业是否处于处理中或已经完成 [num_opes]
        由于本版本为单实例，所有数据均为二维张量或向量，不含 batch 维度。
        返回：
          - action_probs: 每个候选动作的选择概率（已将不合法动作屏蔽为 -inf 后 softmax 归一化）
          - ope_step: 当前待处理工序索引（用于后续计算动作）
          - h_pooled: 拼接后的全局状态特征，用于辅助决策（可选）
        """
        # 提取原始特征
        raw_opes: torch.Tensor = state.feat_opes.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, in_size_ope]
        raw_mas: torch.Tensor = state.feat_mas.unsqueeze(0).to(self.device)  # shape: [batch, num_mas, in_size_ma]
        proc_time: torch.Tensor = state.proc_times.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, num_mas]

        # 是否标准化
        norm_opes, norm_mas, norm_proc = self.get_normalized(raw_opes, raw_mas, proc_time)
        # norm_opes, norm_mas, norm_proc = raw_opes, raw_mas, proc_time

        # 邻接矩阵
        ope_ma_adj: torch.Tensor = state.ope_ma_adj.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, num_mas]
        ope_pre_adj: torch.Tensor = state.ope_pre_adj.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, num_opes]
        ope_sub_adj: torch.Tensor = state.ope_sub_adj.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, num_opes]
        eligible: torch.Tensor = state.eligible_fluid.unsqueeze(0).to(self.device)  # shape: [batch, num_opes, num_mas]

        # features用于后续 HGNN 传播，结构为 (ope_features, mas_features, proc_time)
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (norm_opes, norm_mas, norm_proc)

        # L 次 HGNN 传播（这里 L 等于 self.num_heads 的长度）
        for i in range(len(self.num_heads)):
            # 第一阶段：机器节点嵌入，得到 h_mas: [batch, num_mas, out_size_ma]
            h_mas: torch.Tensor = self.get_machines[i](ope_ma_adj, features)  # 传入 features 整体作为额外信息
            # 更新 features 第二分量为机器嵌入
            features = (features[0], h_mas, features[2])
            # 第二阶段：工序节点嵌入，得到 h_opes: [batch, num_opes, out_size_ope]
            h_opes: torch.Tensor = self.get_operations[i](ope_ma_adj,
                                                          ope_pre_adj,
                                                          ope_sub_adj,
                                                          features)
            # 更新 features 第一分量为工序嵌入
            features = (h_opes, features[1], features[2])

        # 对机器节点及工序节点分别做 pooling（取均值）
        h_mas_pooled: torch.Tensor = torch.mean(h_mas, dim=-2)  # shape: [batch, out_size_ma]
        h_opes_pooled: torch.Tensor = torch.mean(h_opes, dim=-2)  # shape: [batch, out_size_ope]

        # 提取每种工序中的订单交期形成二维列表
        max_long_due_date_list = max(state.kind_task_idle_id_due_date_list.values(), key=len)
        max_long = len(max_long_due_date_list)  # 每个工序类型的最大订单数
        eligibles = eligible.unsqueeze(-1).expand(-1, -1, -1, max_long).clone()
        h_ords_tensor = torch.full((len(state.kind_task_tuple), len(max_long_due_date_list)), 0, device=self.device, dtype=torch.float32)
        for key, values in state.kind_task_idle_id_due_date_list.items():
            kind_task_index = state.kind_task_tuple.index(key)
            for index, due_date in enumerate(values):
                h_ords_tensor[kind_task_index, index] = due_date
            # 从eligible中隐去不可选订单索引
            if 0 < len(values) < max_long:
                eligibles[:, kind_task_index, :, len(values):] = False

        # 计算不同订单的交期均值和标准差
        mean_ords_tensor = torch.mean(h_ords_tensor)
        std_ords_tensor = torch.std(h_ords_tensor, unbiased=False)
        h_ords_tensor = (h_ords_tensor - mean_ords_tensor) / (std_ords_tensor + 1e-5)
        h_ords_padding = h_ords_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-3).expand(-1, -1, ope_ma_adj.size(-1), -1, -1)

        # 扩展 h_opes 与 h_mas 信息以匹配动作计算所需尺寸
        h_opes_padding = h_opes.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, ope_ma_adj.size(-1), max_long, -1)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        h_mas_padding = h_mas.unsqueeze(-3).unsqueeze(-2).expand(-1, ope_ma_adj.size(-2), -1, max_long, -1)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        # 同理，对全局 pooling信息扩展到相同尺寸
        h_mas_pooled_padding = h_mas_pooled.expand_as(h_opes_padding)  # [batch, num_opes, num_mas, num_ords, out_size_ma]
        h_opes_pooled_padding = h_opes_pooled.expand_as(h_opes_padding)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        h_rate_padding = torch.tensor(state.current_order_completed_rate, device=self.device).expand_as(h_ords_padding)
        """拼接形成 actor 网络的输入：每个动作候选的特征组合"""
        # 形状：[batch, num_mas, num_opes, num_ords, out_size_ma + out_size_ope + out_size_ope + out_size_ma+1]
        h_actions = torch.cat((h_opes_padding, h_mas_padding,
                               h_opes_pooled_padding, h_mas_pooled_padding, h_ords_padding, h_rate_padding), dim=-1).transpose(1, 2)
        if torch.isnan(h_actions).any():
            print("组合动作存咋Nan")
        # 更新新的动作字典
        self.current_action_dict = {action_index: (action1, action2) for action_index, (action1, action2) in
                                    enumerate([(i, j) for i in state.action_dict.keys() for j in range(max_long)])}
        # 候选动作合法性掩码：结合ope_ma_adj与作业/机器状态（这里简化为直接使用 ope_ma_adj 的值）
        mask = eligibles.transpose(1, 2).flatten(1)  # 转置后展平，得到 [batch, num_mas * num_opes]

        # 经过 actor 网络计算动作得分
        scores: torch.Tensor = self.actor(h_actions.float()).flatten(1)  # [batch, num_mas * num_opes]
        # 对不合法动作赋予 -inf （注意：eligible 为 True 标示合法）
        scores[~mask] = float('-inf')
        action_probs: torch.Tensor = F.softmax(scores, dim=1)  # 计算动作概率分布

        # 如果action_probs 中有 NaN，输出state中的各属性值
        if torch.isnan(action_probs).any():
            print("kind_task_available_list:", state.kind_task_available_list)
            print("h_ords_tensor:", h_ords_tensor)
            print("达成率", state.current_order_completed_rate)
            print("最大得分:", torch.max(scores))

        if flag_train:
            memory.ope_ma_adj.append(copy.deepcopy(ope_ma_adj))
            memory.ope_pre_adj.append(copy.deepcopy(ope_pre_adj))
            memory.ope_sub_adj.append(copy.deepcopy(ope_sub_adj))
            memory.raw_opes.append(copy.deepcopy(raw_opes))
            memory.raw_mas.append(copy.deepcopy(raw_mas))
            memory.proc_time.append(copy.deepcopy(proc_time))
            memory.eligible.append(copy.deepcopy(eligible))

        h_opes_mas = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        return action_probs, h_opes_mas

    def act(self, state: object, memory, epoch, flag_sample: bool = True):
        """
        根据当前状态 sample 得到动作。返回的动作包含三个部分：
          - opes: 当前操作索引
          - mas: 所选机器索引
        """
        if hasattr(state, "to_policy_obs"):
            obs = state.to_policy_obs()
            action_indexes, log_probs, _, env_actions = self.act_batch([obs], sample=flag_sample)
            env_action = env_actions[0]
            return action_indexes[0], log_probs[0], env_action[0], env_action[1]

        action_probs, h_opes_mas = self.get_action_prob(state, memory)
        train_flag = True  # 前500周期动作随机选择开关

        epsilon = max(self.epsilon * (1000- epoch)/1000, 0.001)

        if random.random() < epsilon and train_flag:
            valid_actions = torch.nonzero(action_probs > 0, as_tuple=False)
            # 随机选择动作
            idx = random.choice(valid_actions.tolist())
            action_index = torch.tensor(idx[1], device=action_probs.device)
            log_prob = torch.log(action_probs[idx[0], action_index])
        else:
            dist = Categorical(action_probs)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)  # 记录动作的 log 概率

        # 选择具体订单动作
        action = action_index.item()  # 工序类型选择策略网络选择的动作
        # 选择工序类型-机器、订单的动作组合
        action1, action2 = self.current_action_dict[action]

        return action_index, log_prob, action1, action2

    def _pad_obs_batch(self, obs_batch):
        batch_size = len(obs_batch)
        max_opes = max(obs["raw_opes"].size(0) for obs in obs_batch)
        max_mas = max(obs["raw_mas"].size(0) for obs in obs_batch)
        max_orders = max(obs["due_dates"].size(1) for obs in obs_batch)
        ope_dim = obs_batch[0]["raw_opes"].size(-1)
        ma_dim = obs_batch[0]["raw_mas"].size(-1)
        padded = {
            "raw_opes": torch.zeros((batch_size, max_opes, ope_dim), dtype=torch.float32, device=self.device),
            "raw_mas": torch.zeros((batch_size, max_mas, ma_dim), dtype=torch.float32, device=self.device),
            "proc_time": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.float32, device=self.device),
            "ope_ma_adj": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.bool, device=self.device),
            "ope_pre_adj": torch.zeros((batch_size, max_opes, max_opes), dtype=torch.bool, device=self.device),
            "ope_sub_adj": torch.zeros((batch_size, max_opes, max_opes), dtype=torch.bool, device=self.device),
            "eligible": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.bool, device=self.device),
            "due_dates": torch.zeros((batch_size, max_opes, max_orders), dtype=torch.float32, device=self.device),
            "order_mask": torch.zeros((batch_size, max_opes, max_orders), dtype=torch.bool, device=self.device),
            "current_rate": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
        }
        for batch_index, obs in enumerate(obs_batch):
            num_opes = obs["raw_opes"].size(0)
            num_mas = obs["raw_mas"].size(0)
            num_orders = obs["due_dates"].size(1)
            padded["raw_opes"][batch_index, :num_opes] = obs["raw_opes"].to(self.device).float()
            padded["raw_mas"][batch_index, :num_mas] = obs["raw_mas"].to(self.device).float()
            padded["proc_time"][batch_index, :num_opes, :num_mas] = obs["proc_time"].to(self.device).float()
            padded["ope_ma_adj"][batch_index, :num_opes, :num_mas] = obs["ope_ma_adj"].to(self.device).bool()
            padded["ope_pre_adj"][batch_index, :num_opes, :num_opes] = obs["ope_pre_adj"].to(self.device).bool()
            padded["ope_sub_adj"][batch_index, :num_opes, :num_opes] = obs["ope_sub_adj"].to(self.device).bool()
            padded["eligible"][batch_index, :num_opes, :num_mas] = obs["eligible"].to(self.device).bool()
            padded["due_dates"][batch_index, :num_opes, :num_orders] = obs["due_dates"].to(self.device).float()
            padded["order_mask"][batch_index, :num_opes, :num_orders] = obs["order_mask"].to(self.device).bool()
            padded["current_rate"][batch_index] = float(obs["current_order_completed_rate"])
        return padded

    def _policy_forward_batch(self, obs_batch):
        start_time = time.perf_counter()
        batch = self._pad_obs_batch(obs_batch)
        raw_opes, raw_mas, proc_time = batch["raw_opes"], batch["raw_mas"], batch["proc_time"]
        ope_ma_adj, ope_pre_adj, ope_sub_adj = batch["ope_ma_adj"], batch["ope_pre_adj"], batch["ope_sub_adj"]
        eligible = batch["eligible"]
        features = self.get_normalized(raw_opes, raw_mas, proc_time)
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, features)
            features = (h_opes, features[1], features[2])
        batch_size, max_opes, _ = h_opes.shape
        max_mas = h_mas.size(1)
        max_orders = batch["due_dates"].size(-1)
        h_mas_pooled = torch.mean(h_mas, dim=-2)
        h_opes_pooled = torch.mean(h_opes, dim=-2)
        due_dates = batch["due_dates"]
        valid_due_dates = due_dates[batch["order_mask"]]
        if valid_due_dates.numel() > 0:
            due_dates = (due_dates - valid_due_dates.mean()) / (valid_due_dates.std(unbiased=False) + 1e-5)

        pair_due = due_dates[:, :, :1].expand(-1, -1, max_mas).transpose(1, 2).unsqueeze(-1)
        pair_rate = batch["current_rate"][:, None, None, None].expand(batch_size, max_mas, max_opes, 1)
        pair_features = torch.cat((
            h_opes[:, None, :, :].expand(-1, max_mas, -1, -1),
            h_mas[:, :, None, :].expand(-1, -1, max_opes, -1),
            h_opes_pooled[:, None, None, :].expand(-1, max_mas, max_opes, -1),
            h_mas_pooled[:, None, None, :].expand(-1, max_mas, max_opes, -1),
            pair_due,
            pair_rate,
        ), dim=-1)

        pair_mask = eligible.transpose(1, 2).flatten(1)
        no_valid = ~pair_mask.any(dim=1)
        if no_valid.any():
            pair_mask[no_valid, 0] = True
        pair_scores = self.actor(pair_features.float()).squeeze(-1).flatten(1).masked_fill(~pair_mask, -1e9)
        pair_probs = F.softmax(pair_scores, dim=1)

        h_ords_padding = due_dates.unsqueeze(1).unsqueeze(-1).expand(-1, max_mas, -1, -1, -1)
        order_features = torch.cat((
            h_opes[:, None, :, None, :].expand(-1, max_mas, -1, max_orders, -1),
            h_mas[:, :, None, None, :].expand(-1, -1, max_opes, max_orders, -1),
            h_opes_pooled[:, None, None, None, :].expand(-1, max_mas, max_opes, max_orders, -1),
            h_mas_pooled[:, None, None, None, :].expand(-1, max_mas, max_opes, max_orders, -1),
            h_ords_padding,
            batch["current_rate"][:, None, None, None, None].expand(batch_size, max_mas, max_opes, max_orders, 1),
        ), dim=-1)
        order_scores = self.order_actor(order_features.float()).squeeze(-1)
        order_mask = eligible.transpose(1, 2).unsqueeze(-1) & batch["order_mask"].unsqueeze(1)
        if no_valid.any():
            order_mask[no_valid, 0, 0, 0] = True

        state_values = self.critic(torch.cat((h_opes_pooled, h_mas_pooled,
                                              batch["current_rate"].view(batch_size, 1)), dim=-1)).squeeze(-1)
        valid_tokens = order_mask.flatten(1).sum(dim=1).float()
        total_tokens = max(order_mask[0].numel(), 1)
        self.last_forward_stats = {
            "batch_size": int(batch_size),
            "avg_action_tokens": float(valid_tokens.mean().detach().cpu().item()),
            "max_action_tokens": float(valid_tokens.max().detach().cpu().item()),
            "padding_ratio": float(1.0 - valid_tokens.sum().detach().cpu().item() / max(batch_size * total_tokens, 1)),
            "forward_seconds": float(time.perf_counter() - start_time),
        }
        return pair_probs, order_scores, order_mask, state_values, batch

    def _has_sparse_candidates(self, obs_batch):
        return bool(obs_batch) and all("valid_pairs" in obs and "order_counts" in obs for obs in obs_batch)

    def _apply_sparse_attention(self, candidate_features, attention_layer, norm_layer):
        if attention_layer is None or norm_layer is None or candidate_features.size(0) <= 1:
            return candidate_features
        attended, _ = attention_layer(
            candidate_features.unsqueeze(0),
            candidate_features.unsqueeze(0),
            candidate_features.unsqueeze(0),
            need_weights=False,
        )
        return norm_layer(candidate_features + attended.squeeze(0))

    def _sparse_embedding_batch(self, obs_batch):
        start_time = time.perf_counter()
        batch = self._pad_obs_batch(obs_batch)
        raw_opes, raw_mas, proc_time = batch["raw_opes"], batch["raw_mas"], batch["proc_time"]
        ope_ma_adj, ope_pre_adj, ope_sub_adj = batch["ope_ma_adj"], batch["ope_pre_adj"], batch["ope_sub_adj"]
        features = self.get_normalized(raw_opes, raw_mas, proc_time)
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, features)
            features = (h_opes, features[1], features[2])
        h_mas_pooled = torch.mean(h_mas, dim=-2)
        h_opes_pooled = torch.mean(h_opes, dim=-2)
        due_dates = batch["due_dates"]
        valid_due_dates = due_dates[batch["order_mask"]]
        if valid_due_dates.numel() > 0:
            due_dates = (due_dates - valid_due_dates.mean()) / (valid_due_dates.std(unbiased=False) + 1e-5)
        state_values = self.critic(torch.cat((h_opes_pooled, h_mas_pooled,
                                              batch["current_rate"].view(len(obs_batch), 1)), dim=-1)).squeeze(-1)
        return batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, start_time

    def _sparse_pair_distribution(self, obs, batch_index, h_opes, h_mas,
                                  h_opes_pooled, h_mas_pooled, due_dates, current_rate):
        valid_pairs = obs["valid_pairs"].to(self.device).long()
        if valid_pairs.numel() == 0:
            valid_pairs = torch.zeros((1, 2), dtype=torch.long, device=self.device)
        machine_indexes = valid_pairs[:, 0].clamp(0, h_mas.size(1) - 1)
        task_indexes = valid_pairs[:, 1].clamp(0, h_opes.size(1) - 1)
        pair_due = due_dates[batch_index, task_indexes, 0:1]
        pair_rate = current_rate.view(1, 1).expand(valid_pairs.size(0), 1)
        pair_features = torch.cat((
            h_opes[batch_index, task_indexes],
            h_mas[batch_index, machine_indexes],
            h_opes_pooled[batch_index].view(1, -1).expand(valid_pairs.size(0), -1),
            h_mas_pooled[batch_index].view(1, -1).expand(valid_pairs.size(0), -1),
            pair_due,
            pair_rate,
        ), dim=-1)
        if self.use_sparse_attention:
            pair_features = self._apply_sparse_attention(
                pair_features,
                self.sparse_pair_attention,
                self.sparse_pair_norm,
            )
        pair_scores = self.actor(pair_features.float()).squeeze(-1)
        return Categorical(F.softmax(pair_scores, dim=0)), valid_pairs

    def _sparse_order_distribution(self, obs, batch_index, machine_index, task_index,
                                   h_opes, h_mas, h_opes_pooled, h_mas_pooled,
                                   due_dates, current_rate):
        order_counts = obs["order_counts"].to(self.device).long()
        max_orders = due_dates.size(-1)
        order_count = int(order_counts[task_index].detach().cpu().item()) if task_index < order_counts.numel() else 0
        order_count = max(1, min(order_count, max_orders))
        order_due = due_dates[batch_index, task_index, :order_count].view(order_count, 1)
        order_rate = current_rate.view(1, 1).expand(order_count, 1)
        order_features = torch.cat((
            h_opes[batch_index, task_index].view(1, -1).expand(order_count, -1),
            h_mas[batch_index, machine_index].view(1, -1).expand(order_count, -1),
            h_opes_pooled[batch_index].view(1, -1).expand(order_count, -1),
            h_mas_pooled[batch_index].view(1, -1).expand(order_count, -1),
            order_due,
            order_rate,
        ), dim=-1)
        if self.use_sparse_attention and self.sparse_attention_scope == "pair_order":
            order_features = self._apply_sparse_attention(
                order_features,
                self.sparse_order_attention,
                self.sparse_order_norm,
            )
        order_scores = self.order_actor(order_features.float()).squeeze(-1)
        return Categorical(F.softmax(order_scores, dim=0)), order_count

    def _act_batch_sparse(self, obs_batch, sample=True):
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, start_time = self._sparse_embedding_batch(obs_batch)
        action_indexes = []
        log_probs = []
        env_actions = []
        valid_action_tokens = []
        for batch_index, obs in enumerate(obs_batch):
            current_rate = batch["current_rate"][batch_index]
            pair_dist, valid_pairs = self._sparse_pair_distribution(
                obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, current_rate
            )
            pair_local_index = pair_dist.sample() if sample else pair_dist.probs.argmax(dim=0)
            machine_index = int(valid_pairs[pair_local_index, 0].detach().cpu().item())
            task_index = int(valid_pairs[pair_local_index, 1].detach().cpu().item())
            order_dist, order_count = self._sparse_order_distribution(
                obs, batch_index, machine_index, task_index, h_opes, h_mas,
                h_opes_pooled, h_mas_pooled, due_dates, current_rate
            )
            order_index = order_dist.sample() if sample else order_dist.probs.argmax(dim=0)
            log_probs.append(pair_dist.log_prob(pair_local_index) + order_dist.log_prob(order_index))
            action_indexes.append(pair_local_index * max(int(obs.get("max_orders", 1)), 1) + order_index)
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            env_actions.append((machine_index * local_tasks + task_index, int(order_index.detach().cpu().item())))
            valid_action_tokens.append(int(valid_pairs.size(0)) * int(order_count))

        action_indexes = torch.stack([index if torch.is_tensor(index) else torch.tensor(index, device=self.device)
                                      for index in action_indexes]).long()
        log_probs = torch.stack(log_probs)
        valid_tokens = torch.tensor(valid_action_tokens, dtype=torch.float32, device=self.device)
        self.last_forward_stats = {
            "batch_size": int(len(obs_batch)),
            "avg_action_tokens": float(valid_tokens.mean().detach().cpu().item()) if valid_tokens.numel() else 0.0,
            "max_action_tokens": float(valid_tokens.max().detach().cpu().item()) if valid_tokens.numel() else 0.0,
            "padding_ratio": 0.0,
            "forward_seconds": float(time.perf_counter() - start_time),
        }
        return action_indexes, log_probs, state_values, env_actions

    def _evaluate_batch_sparse(self, obs_batch, actions):
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, _ = self._sparse_embedding_batch(obs_batch)
        log_probs = []
        entropies = []
        for batch_index, (obs, env_action) in enumerate(zip(obs_batch, actions)):
            action1, order_index = env_action
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            task_index = int(action1) % local_tasks
            machine_index = int(action1) // local_tasks
            current_rate = batch["current_rate"][batch_index]
            pair_dist, valid_pairs = self._sparse_pair_distribution(
                obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, current_rate
            )
            matches = torch.nonzero(
                (valid_pairs[:, 0] == machine_index) & (valid_pairs[:, 1] == task_index),
                as_tuple=False,
            )
            pair_local_index = matches[0, 0] if matches.numel() else torch.tensor(0, dtype=torch.long, device=self.device)
            order_dist, order_count = self._sparse_order_distribution(
                obs, batch_index, machine_index, task_index, h_opes, h_mas,
                h_opes_pooled, h_mas_pooled, due_dates, current_rate
            )
            order_tensor = torch.tensor(min(max(int(order_index), 0), order_count - 1),
                                        dtype=torch.long, device=self.device)
            log_probs.append(pair_dist.log_prob(pair_local_index) + order_dist.log_prob(order_tensor))
            entropies.append(pair_dist.entropy() + order_dist.entropy())
        return torch.stack(log_probs), state_values, torch.stack(entropies)

    def act_batch(self, obs_batch, sample=True):
        if self._has_sparse_candidates(obs_batch):
            return self._act_batch_sparse(obs_batch, sample=sample)
        pair_probs, order_scores, order_mask, state_values, batch = self._policy_forward_batch(obs_batch)
        pair_dist = Categorical(pair_probs)
        pair_indexes = pair_dist.sample() if sample else pair_probs.argmax(dim=1)
        batch_indexes = torch.arange(len(obs_batch), device=self.device)
        max_opes = batch["raw_opes"].size(1)
        max_orders = batch["due_dates"].size(2)
        flat_order_scores = order_scores.reshape(len(obs_batch), -1, max_orders)
        flat_order_mask = order_mask.reshape(len(obs_batch), -1, max_orders)
        selected_order_scores = flat_order_scores[batch_indexes, pair_indexes]
        selected_order_mask = flat_order_mask[batch_indexes, pair_indexes]
        no_order = ~selected_order_mask.any(dim=1)
        if no_order.any():
            selected_order_mask[no_order, 0] = True
        selected_order_scores = selected_order_scores.masked_fill(~selected_order_mask, -1e9)
        order_probs = F.softmax(selected_order_scores, dim=1)
        order_dist = Categorical(order_probs)
        order_indexes = order_dist.sample() if sample else order_probs.argmax(dim=1)
        log_probs = pair_dist.log_prob(pair_indexes) + order_dist.log_prob(order_indexes)
        action_indexes = pair_indexes * max_orders + order_indexes
        env_actions = []
        for batch_index, pair_index in enumerate(pair_indexes.tolist()):
            obs = obs_batch[batch_index]
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            machine_index = int(pair_index) // max_opes
            task_index = int(pair_index) % max_opes
            env_actions.append((machine_index * local_tasks + task_index, int(order_indexes[batch_index].item())))
        return action_indexes, log_probs, state_values, env_actions

    def _env_actions_to_pair_order_indexes(self, obs_batch, env_actions, max_opes: int, max_orders: int):
        pair_indexes = []
        order_indexes = []
        for obs, env_action in zip(obs_batch, env_actions):
            action1, order_index = env_action
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            task_index = int(action1) % local_tasks
            machine_index = int(action1) // local_tasks
            pair_indexes.append(machine_index * max_opes + task_index)
            order_indexes.append(min(max(int(order_index), 0), max_orders - 1))
        return (
            torch.tensor(pair_indexes, dtype=torch.long, device=self.device),
            torch.tensor(order_indexes, dtype=torch.long, device=self.device),
        )

    def _env_actions_to_padded_indexes(self, obs_batch, env_actions):
        max_opes = max(obs["raw_opes"].size(0) for obs in obs_batch)
        max_orders = max(obs["due_dates"].size(1) for obs in obs_batch)
        pair_indexes, order_indexes = self._env_actions_to_pair_order_indexes(obs_batch, env_actions, max_opes, max_orders)
        return pair_indexes * max_orders + order_indexes

    def evaluate_batch(self, obs_batch, actions):
        if self._has_sparse_candidates(obs_batch) and not torch.is_tensor(actions):
            return self._evaluate_batch_sparse(obs_batch, actions)
        pair_probs, order_scores, order_mask, state_values, batch = self._policy_forward_batch(obs_batch)
        pair_dist = Categorical(pair_probs)
        batch_size = len(obs_batch)
        max_orders = batch["due_dates"].size(2)
        max_opes = batch["raw_opes"].size(1)
        if torch.is_tensor(actions):
            action_indexes = actions.to(self.device).long()
            pair_indexes = torch.div(action_indexes, max_orders, rounding_mode="floor")
            order_indexes = action_indexes % max_orders
        else:
            pair_indexes, order_indexes = self._env_actions_to_pair_order_indexes(obs_batch, actions, max_opes, max_orders)

        batch_indexes = torch.arange(batch_size, device=self.device)
        flat_order_scores = order_scores.reshape(batch_size, -1, max_orders)
        flat_order_mask = order_mask.reshape(batch_size, -1, max_orders)
        selected_order_scores = flat_order_scores[batch_indexes, pair_indexes]
        selected_order_mask = flat_order_mask[batch_indexes, pair_indexes]
        no_order = ~selected_order_mask.any(dim=1)
        if no_order.any():
            selected_order_mask[no_order, 0] = True
        selected_order_scores = selected_order_scores.masked_fill(~selected_order_mask, -1e9)
        order_dist = Categorical(F.softmax(selected_order_scores, dim=1))
        log_probs = pair_dist.log_prob(pair_indexes) + order_dist.log_prob(order_indexes)
        entropies = pair_dist.entropy() + order_dist.entropy()
        return log_probs, state_values, entropies


    def evaluate(self,
                 state: object,
                 ope_ma_adj: torch.Tensor,
                 ope_pre_adj: torch.Tensor,
                 ope_sub_adj: torch.Tensor,
                 raw_opes: torch.Tensor,
                 raw_mas: torch.Tensor,
                 proc_time: torch.Tensor,
                 eligible: torch.Tensor,
                 action_env: torch.Tensor,
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据当前状态和动作对 actor 与 critic 网络输出进行评估，用于 PPO 更新。
        参数均为单实例数据，不再包含 batch 维度。
        返回：动作 log 概率、状态价值和策略熵。
        """

        features = self.get_normalized(raw_opes, raw_mas, proc_time)
        # features = raw_opes, raw_mas, proc_time

        # L 次 HGNN 传播（这里 L 等于 self.num_heads 的长度）
        for i in range(len(self.num_heads)):
            # 第一阶段：机器节点嵌入，得到 h_mas: [batch, num_mas, out_size_ma]
            h_mas: torch.Tensor = self.get_machines[i](ope_ma_adj, features)  # 传入 features 整体作为额外信息
            # 更新 features 第二分量为机器嵌入
            features = (features[0], h_mas, features[2])
            # 第二阶段：工序节点嵌入，得到 h_opes: [batch, num_opes, out_size_ope]
            h_opes: torch.Tensor = self.get_operations[i](ope_ma_adj,
                                                          ope_pre_adj,
                                                          ope_sub_adj,
                                                          features)
            # 更新 features 第一分量为工序嵌入
            features = (h_opes, features[1], features[2])

        # 对机器节点及工序节点分别做 pooling（取均值）
        h_mas_pooled: torch.Tensor = torch.mean(h_mas, dim=-2)  # shape: [batch, 1, out_size_ma]
        h_opes_pooled: torch.Tensor = torch.mean(h_opes, dim=-2)  # shape: [batch, 1, out_size_ope]

        # 提取每种工序中的订单交期形成二维列表
        max_long_due_date_list = max(state.kind_task_idle_id_due_date_list.values(), key=len)
        max_long = len(max_long_due_date_list)  # 每个工序类型的最大订单数
        eligible = eligible.unsqueeze(-1).expand(-1, -1, -1, max_long).clone()
        h_ords_tensor = torch.full((len(state.kind_task_tuple), len(max_long_due_date_list)), 0, device=self.device,
                                   dtype=torch.float32)
        for key, values in state.kind_task_idle_id_due_date_list.items():
            kind_task_index = state.kind_task_tuple.index(key)
            for index, due_date in enumerate(values):
                h_ords_tensor[kind_task_index, index] = due_date
            # 从eligible中隐去不可选订单索引
            if 0 < len(values) < max_long:
                eligible[:, kind_task_index, :, len(values):] = False
        mean_ords_tensor = torch.mean(h_ords_tensor)
        std_ords_tensor = torch.std(h_ords_tensor, unbiased=False)
        h_ords_tensor = (h_ords_tensor - mean_ords_tensor) / (std_ords_tensor + 1e-5)
        h_ords_padding = h_ords_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-3).expand(-1, -1, ope_ma_adj.size(-1), -1,
                                                                                       -1)

        # 扩展 h_opes 与 h_mas 信息以匹配动作计算所需尺寸
        h_opes_padding = h_opes.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, ope_ma_adj.size(-1), max_long,
                                                                   -1)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        h_mas_padding = h_mas.unsqueeze(-3).unsqueeze(-2).expand(-1, ope_ma_adj.size(-2), -1, max_long,
                                                                 -1)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        # 同理，对全局 pooling信息扩展到相同尺寸
        h_mas_pooled_padding = h_mas_pooled.expand_as(
            h_opes_padding)  # [batch, num_opes, num_mas, num_ords, out_size_ma]
        h_opes_pooled_padding = h_opes_pooled.expand_as(
            h_opes_padding)  # [batch, num_opes, num_mas, num_ords, out_size_ope]
        h_rate_padding = torch.tensor(state.current_order_completed_rate, device=self.device).expand_as(h_ords_padding)
        """拼接形成 actor 网络的输入：每个动作候选的特征组合"""
        # 形状：[batch, num_mas, num_opes, num_ords, out_size_ma + out_size_ope + out_size_ope + out_size_ma+1]
        h_actions = torch.cat((h_opes_padding, h_mas_padding,
                               h_opes_pooled_padding, h_mas_pooled_padding, h_ords_padding, h_rate_padding),
                              dim=-1).transpose(1, 2)
        # 更新新的动作字典
        self.current_action_dict = {action_index: (action1, action2) for action_index, (action1, action2) in
                                    enumerate([(i, j) for i in state.action_dict.keys() for j in range(max_long)])}
        # 候选动作合法性掩码：结合ope_ma_adj与作业/机器状态（这里简化为直接使用 ope_ma_adj 的值）
        mask = eligible.transpose(1, 2).flatten(1)  # 转置后展平，得到 [batch, num_mas * num_opes]

        # 经过 actor 网络计算动作得分
        scores: torch.Tensor = self.actor(h_actions.float()).flatten(1)  # [batch, num_mas * num_opes]
        # 对不合法动作赋予 -inf （注意：eligible 为 True 标示合法）
        scores[~mask] = float('-inf')
        action_probs: torch.Tensor = F.softmax(scores, dim=1)  # 计算动作概率分布

        # 计算动作的 log 概率和熵
        dist = Categorical(action_probs)  # 使用 Categorical 分布处理动作概率
        action_log_prob = dist.log_prob(action_env)  # action_env 为采样动作索引
        dist_entropy = dist.entropy()  # 计算策略熵
        # 评价状态价值（将全局 pooling特征拼接后输入 critic）
        h_rate_tensor = torch.tensor(state.current_order_completed_rate, device=self.device).unsqueeze(-1).unsqueeze(-1).float()
        state_value = self.critic(torch.cat((h_opes_pooled, h_mas_pooled, h_rate_tensor), dim=-1))

        return action_log_prob, state_value.squeeze(), dist_entropy
