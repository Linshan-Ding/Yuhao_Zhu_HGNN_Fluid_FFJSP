import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class GATedge(nn.Module):
    """
    机器节点嵌入
    Args:
        in_feats (Tuple[int, int]): 输入特征维度，格式为 (src_feature_dim, dst_feature_dim)。
        out_feats (int): 输出特征维度。
        num_head (int): 注意力头个数。
        feat_drop (float, optional): 节点特征丢弃率，默认为 0.0。
        attn_drop (float, optional): 注意力权重丢弃率，默认为 0.0。
        negative_slope (float, optional): LeakyReLU 的负斜率，默认为 0.2。
        residual (bool, optional): 是否使用残差连接，默认为 False。
        activation (Optional[torch.nn.Module], optional): 激活函数，可选。
    """

    def __init__(self,
                 in_feats: Tuple[int, int],
                 out_feats: int,
                 num_head: int,
                 feat_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        self._num_heads = num_head
        self._in_src_feats, self._in_dst_feats = in_feats
        self._out_feats = out_feats

        # 分别对源节点、目标节点和边进行线性变换
        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_head, bias=False)
        self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_head, bias=False)
        self.fc_edge = nn.Linear(1, out_feats * num_head, bias=False)

        # 注意力参数，分别对应源、目标与边
        self.attn_l = nn.Parameter(torch.rand((1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand((1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand((1, num_head, out_feats), dtype=torch.float))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        # 如果 residual 为 True，则设置残差连接
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', torch.tensor(0.0))  # 使用空缓冲区 placeholder

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化所有权重参数"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self,
                ope_ma_adj: torch.Tensor,
                feat: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
        """
        前向传播。

        Args:
            ope_ma_adj (torch.Tensor): 邻接矩阵
            feat (Union[Tuple, torch.Tensor]): 节点和边的特征张量，
                      当为 tuple 时格式为 (src_feat, dst_feat, edge_feat)。

        Returns:
             torch.Tensor: 生成的机器节点嵌入。
        """
        # 当输入特征为元组时，对源节点和目标节点分别做变换
        if isinstance(feat, tuple):
            src_feat, dst_feat, edge_feat = feat
            src_feat = self.feat_drop(src_feat)
            dst_feat = self.feat_drop(dst_feat)
            feat_src = self.fc_src(src_feat)
            feat_dst = self.fc_dst(dst_feat)
        else:
            # 若非元组情况（较少使用），统一视为相同输入
            feat = self.feat_drop(feat)
            feat_src = feat_dst = self.fc_src(feat)

        # 对边特征做线性变换
        feat_edge = self.fc_edge(edge_feat.unsqueeze(-1))

        # 计算源、目标以及边上的注意力评分
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # shape: (batch, num_heads, 1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)  # shape: (num_heads, 1)
        ee = (feat_edge * self.attn_e).sum(dim=-1).unsqueeze(-1)  # shape: (num_heads, 1)

        # 组合注意力分量，注意力计算同时考虑边连接关系 (ope_ma_adj_batch)
        el_add_ee = ope_ma_adj.unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj.unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)

        # 计算机器节点自注意力评分
        ekk = self.leaky_relu(er + er)

        # 构造 mask，用于排除无效位置
        mask = torch.cat((ope_ma_adj.unsqueeze(-1)==1,torch.full(size=(ope_ma_adj.size(0), 1, ope_ma_adj.size(2), 1),
                                                                 dtype=torch.bool, fill_value=True, device=ope_ma_adj.device)), dim=-3)
        e_cat = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e_cat[~mask] = float('-inf')
        alpha = F.softmax(e_cat.squeeze(-1), dim=-2)

        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # 融合边和节点特征，用加权求和方式
        weighted_feat = feat_edge + feat_src.unsqueeze(-2)  # broadcast 源节点信息到边上
        out1 = torch.sum(weighted_feat * alpha_ijk.unsqueeze(-1), dim=-3)
        out2 = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(out1 + out2)
        if self.activation is not None:
            nu_k_prime = self.activation(nu_k_prime)
        return nu_k_prime


class MLPsim(nn.Module):
    """
    工序类型节点嵌入部分代码
    Args:
        in_feats (int): 输入特征维度。
        out_feats (int): 输出特征维度。
        hidden_dim (int): 隐藏层维度。
        num_head (int): 注意力头数量（部分实现中保留）。
        feat_drop (float, optional): 特征丢弃率，默认为 0.0。
        attn_drop (float, optional): 注意力丢弃率，默认为 0.0。
        negative_slope (float, optional): LeakyReLU 的负斜率，默认为 0.2。
        residual (bool, optional): 是否使用残差连接，默认为 False。
        """

    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 hidden_dim: int,
                 num_head: int,
                 feat_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 negative_slope: float = 0.2,
                 residual: bool = False) -> None:
        super().__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats)
        )

        if residual:
            # 当残差连接时，如输入与输出维度不一致，则通过线性层调整
            self.res_fc = nn.Linear(self._in_feats, self._num_heads * out_feats, bias=False) \
                if self._in_feats != out_feats else Identity()
        else:
            self.register_buffer('res_fc', torch.tensor(0.0))

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            feat (torch.Tensor): 输入特征，形状为 (..., in_feats)。
            adj (torch.Tensor): 邻接矩阵，形状为 (..., )，用于指导消息传递。

        Returns:
            torch.Tensor: 输出特征，形状与输入经过 MLP 处理后的维度一致。
        """
        # 将邻接矩阵扩展一个维度，与特征进行逐元素相乘，模拟消息传递
        message = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        aggregated = torch.sum(message, dim=-2)
        out = self.project(aggregated)
        return out
