
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn import metrics
import numpy as np
import pandas as pd
import warnings

import random
class GATMechanism(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        GAT 机制，支持特征拼接
        :param input_dim: 输入特征维度 (F)
        :param output_dim: 输出特征维度 (F')
        """
        super(GATMechanism, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)  # [F, F']
        self.a = nn.Parameter(torch.randn(2 * output_dim + input_dim, 1))  # [2F' + F, 1]

    def forward(self, V, E, time_enc):
        """
        :param V: 节点特征 [N, F]
        :param E: 邻接矩阵 [N, N]
        :param time_enc: 时间编码 [N, N, F]
        :return: 加权消息 [N, N, F']
        """
        N = V.size(0)
        V_i = V.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        V_j = V.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        M = self.W(V_j)  # [N, N, F']，仅变换 v_j^t
        attention_scores = self.compute_attention(V_i, V_j, time_enc)  # [N, N]
        H = (attention_scores.unsqueeze(-1) * M) * E.unsqueeze(-1)  # [N, N, F']，用 E 限制邻居
        return H

    def compute_attention(self, V_i, V_j, time_enc):
        """
        计算节点对之间的注意力权重
        """
        V_i_transformed = self.W(V_i)  # [N, N, F']
        V_j_transformed = self.W(V_j)  # [N, N, F']
        feature_concat = torch.cat([V_i_transformed, V_j_transformed, time_enc], dim=-1).float()  # [N, N, 2F' + F]
        attention_score = torch.matmul(feature_concat, self.a).squeeze(-1)  # [N, N]
        attention_score = F.leaky_relu(attention_score)
        attention_score = F.softmax(attention_score, dim=-1)  # 对邻居归一化
        return attention_score


class ConditionGraphRoutingNetwork(nn.Module):
    def __init__(self, rnn_unit, n_hid, K, Top_K):
        """
        初始化 CGRN
        :param rnn_unit: 输入特征维度 (F)
        :param n_hid: 输出特征维度 (F')
        :param K: 机制数量 (L)
        :param Top_K: Top-K 策略
        """
        super(ConditionGraphRoutingNetwork, self).__init__()
        self.n_hid = n_hid  # F'
        self.K = K  # L
        self.Top_K = Top_K
        self.rnn_unit = rnn_unit  # F

        # 单一 W_a 参数，用于一次性计算 L 个机制的分数
        self.W_a = nn.Parameter(torch.randn(K, 3 * rnn_unit))  # [L, 3F]
        # GAT 机制列表
        self.gat_layers = nn.ModuleList([GATMechanism(rnn_unit, n_hid) for _ in range(K)])
        # BatchNorm 层
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_hid) for _ in range(K)])

    def time_encoding(self, delta_t):
        """
        时间编码
        :param delta_t: 滞后矩阵 [N, N]
        :return: 时间编码 [N, N, F]
        """
        dim_t = torch.arange(0, self.rnn_unit // 2, device=delta_t.device).float()
        div_term = 10000 ** (2 * dim_t / self.rnn_unit).float()
        sin_term = torch.sin(delta_t.unsqueeze(-1) / div_term)  # [N, N, F/2]
        cos_term = torch.cos(delta_t.unsqueeze(-1) / div_term)  # [N, N, F/2]
        return torch.cat([sin_term, cos_term], dim=-1)  # [N, N, F]

    def top_k_selection(self, scores):
        """
        Top-K 策略
        :param scores: 匹配分数 [N, N, L]
        :return: 经过 Top-K 筛选的分数 [N, N, L] 和路径掩码
        """
        _, indices = torch.topk(scores, self.Top_K, dim=-1)  # [N, N, Top_K]
        mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)  # [N, N, L]
        return scores * mask, mask

    def forward(self, V, Delta_t, E):
        """
        :param V: 节点特征 [N, F]
        :param Delta_t: 滞后矩阵 [N, N]
        :param E: 邻接矩阵 [N, N]
        :return: 更新后的节点特征 [N, F']
        """
        N = V.size(0)
        time_enc = self.time_encoding(Delta_t)  # [N, N, F]

        # 计算匹配分数（一次性计算 L 个机制）
        V_i = V.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        V_j = V.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        feature_concat = torch.cat([V_i, V_j, time_enc], dim=-1).float()  # [N, N, 3F]
        scores = F.leaky_relu(torch.matmul(feature_concat, self.W_a.T))  # [N, N, L]

        # Top-K 筛选和归一化
        scores, mask = self.top_k_selection(scores)  # [N, N, L]
        scores_exp = torch.exp(scores) * mask  # [N, N, L]
        scores_sum = scores_exp.sum(dim=-1, keepdim=True) + 1e-8  # [N, N, 1]
        p = scores_exp / scores_sum  # [N, N, L]，对应 p_{i,j}^l

        # 消息传递和聚合
        messages = torch.zeros(N, N, self.n_hid, device=V.device)  # [N, N, F']
        for k in range(self.K):
            if mask[..., k].any():  # 仅对显著路径计算
                H_k = self.gat_layers[k](V, E, time_enc)  # [N, N, F']
                messages += (E.unsqueeze(-1) * mask[..., k].unsqueeze(-1)) * (p[..., k].unsqueeze(-1) * H_k)

        # 最终节点特征聚合
        V_out = messages.sum(dim=1)  # [N, F']
        return V_out

# 噪声感知关系推断模块
class NoiseAwareRelationInference(nn.Module):
    def __init__(self, rnn_unit, n_dim, epsilon=0.5):
        super(NoiseAwareRelationInference, self).__init__()
        # 可学习参数
        self.W_r = nn.Parameter(torch.randn(2 * rnn_unit, n_dim))
        self.a_r = nn.Parameter(torch.randn(n_dim))
        self.epsilon = epsilon  # 噪声过滤因子

    def forward(self, X):
        """
        输入:
        X: [N, F] 所有股票的特征矩阵
        输出:
        E: [N, N] 股票之间的关系矩阵
        """
        N, Fea = X.size()

        X_i = X.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        X_j = X.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        X_combined = torch.cat((X_i, X_j), dim=-1)  # [N, N, 2F]

        transformed = torch.matmul(X_combined, self.W_r)  # [N, N, F']

        R = torch.matmul(transformed, self.a_r)  # [N, N]

        R = F.leaky_relu(R)  # [N, N]

        R = (R - R.mean()) / (R.std() + 1e-5)  # 归一化 R 矩阵

        E = F.softmax(R, dim=1)  # 对列进行归一化 [N, N]

        E_mean = E.mean()  # 标量

        threshold = (1 / (self.epsilon)) * (E_mean)

        filtered_E = torch.where(E >= threshold, E, torch.zeros_like(E))
        return filtered_E




















# 超前滞后效应计算模块 (AutoCorrelation 和 Lead Estimation)
class AutoCorrelationLeadLagEffect(nn.Module):
    def __init__(self, max_lag_step=0):
        super(AutoCorrelationLeadLagEffect, self).__init__()
        self.max_lag_step = max_lag_step  # 最大滞后步范围

    def forward(self, v_i, v_j):
        """
        输入:
        v_i, v_j: [T, F] - 股票i和股票j的特征矩阵
        输出:
        lag_time: [1] - 计算出的滞后时间差
        optimal_lag_step: [1] - 计算出的最优滞后步
        """
        T, F = v_i.shape

        # 对每个滞后步长计算 AutoCorrelation
        auto_corr = torch.zeros(self.max_lag_step + 1)  # 存储各滞后步长的相关性
        for tau in range(self.max_lag_step + 1):
            # 滞后对齐 v_j 并计算点积相关性
            shifted_v_j = torch.roll(v_j, shifts=-tau, dims=0)
            valid_length = T - tau
            corr = torch.sum(v_i[:valid_length] * shifted_v_j[:valid_length]) / (valid_length * F)
            auto_corr[tau] = corr

        # 找到最大相关性对应的滞后步长
        optimal_lag_step = torch.argmax(auto_corr).item()
        lag_time = auto_corr[optimal_lag_step].item()

        return lag_time, optimal_lag_step


# 动态超前滞后检测模型
class DynamicLeadLagModel(nn.Module):
    def __init__(self, F, epsilon=0.5, max_lag_step=3):
        super(DynamicLeadLagModel, self).__init__()
        # 初始化模块
        self.relation_inference = NoiseAwareRelationInference(F=F, epsilon=epsilon)
        self.auto_corr_module = AutoCorrelationLeadLagEffect(max_lag_step=max_lag_step)

    def forward(self, X):
        """
        输入:
        X: [T, N, F] - 股票的特征矩阵
        输出:
        lead_lag_times: [N, N] - 每对股票的滞后时间差矩阵
        lead_steps: [N, N] - 每对股票的最优滞后步
        """
        T, N, F = X.size()

        # 计算关系矩阵 E：[N, N]
        E = self.relation_inference(X)

        # 初始化滞后时间差矩阵和最优滞后步矩阵
        lead_lag_times = torch.zeros(N, N)
        lead_steps = torch.zeros(N, N)

        # 对所有股票对进行处理，使用批量操作减少计算量
        for i in range(N):
            for j in range(i + 1, N):  # 只计算上三角部分，避免重复计算
                # 获取股票i和股票j的特征
                v_i = X[:, i, :]  # [T, F]
                v_j = X[:, j, :]  # [T, F]

                # 如果两只股票有显著关系
                if E[i, j] > 0 and lead_lag_times[i, j] == 0:  # 只计算未计算的部分:
                    # 使用 AutoCorrelation 计算滞后效应
                    lag_time, optimal_lag_step = self.auto_corr_module(v_i, v_j)

                    # 记录最优滞后步
                    lead_steps[i, j] = optimal_lag_step
                    lead_steps[j, i] = -optimal_lag_step  # 对称赋值，反向步长

                    # 记录滞后时间差
                    lead_lag_times[i, j] = lag_time
                    lead_lag_times[j, i] = -lag_time  # 对称赋值

        return lead_lag_times, E  # 返回滞后时间差和最优滞后步





import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalChannelInteractionFusionModule(nn.Module):
    def __init__(self, feature_dim, rnn_unit, time_steps, n_hid):
        """
        初始化市场感知时间-通道交互模块
        :param feature_dim: 特征维度 C
        :param rnn_unit: GRU 单元数 F (输出维度)
        :param time_steps: 时间步长 T
        :param n_hid: 隐藏层维度 F' (中间表示维度)
        """
        super(TemporalChannelInteractionFusionModule, self).__init__()

        self.feature_dim = feature_dim  # C
        self.rnn_unit = rnn_unit       # F
        self.time_steps = time_steps   # T
        self.n_hid = n_hid             # F'

        # 时间交互动态权重参数
        self.W_tem1 = nn.Parameter(torch.randn(n_hid, time_steps))  # F' × T
        self.W_tem2 = nn.Parameter(torch.randn(time_steps, n_hid))  # T × F'

        # 特征交互动态权重参数
        self.W_cha1 = nn.Parameter(torch.randn(feature_dim, n_hid))  # C × F'
        self.W_cha2 = nn.Parameter(torch.randn(n_hid, feature_dim))  # F' × C

        # 重要性权重生成参数
        self.W_tem3 = nn.Parameter(torch.randn(n_hid, feature_dim))  # F' × C
        self.b_tem3 = nn.Parameter(torch.randn(time_steps))               # F'
        self.W_cha3 = nn.Parameter(torch.randn(n_hid, time_steps))   # F' × T
        self.b_cha3 = nn.Parameter(torch.randn(feature_dim))               # F'

        # 双路径门控机制参数
        self.W_t = nn.Parameter(torch.randn(n_hid, feature_dim))  # (F', F)
        self.G_t = nn.Parameter(torch.randn(n_hid, feature_dim))  # (F', F)

        # GRU 用于全局时间依赖性建模
        self.gru = nn.GRU(input_size=feature_dim + n_hid, hidden_size=rnn_unit,
                         num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (T, N, F) - T: 时间步长, N: 股票数量, F: 特征维度
        :return: 输出交互特征，形状为 (N, rnn_unit)
        """
        T, N, F = x.size()

        # Step 1: 计算市场状态向量 z_t
        norm_x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)  # (T, N, F)
        z_t = norm_x.mean(dim=1)  # (T, F)

        # Step 2: 动态权重生成
        gamma_tem = torch.softmax(torch.matmul(self.W_tem3, z_t.T) + self.b_tem3, dim=-1).mean(dim=0)  # (T,)
        gamma_cha = torch.softmax(torch.matmul(self.W_cha3, z_t) + self.b_cha3, dim=-1).mean(dim=0)    # (F,)

        # Step 3: 时间交互检测器
        S_tem = self.time_interaction(norm_x, gamma_tem)  # (T, N, F)

        # Step 4: 特征交互检测器
        S_cha = self.channel_interaction(norm_x, gamma_cha)  # (T, N, F)

        # Step 5: 时间交互和特征交互融合
        S_fused = self.feature_fusion(S_tem, S_cha)  # (T, N, F)

        # Step 6: 双路径门控机制
        S_int = self.dual_gate(S_fused)  # (T, N, F)

        # Step 7: 拼接输入和交互特征并通过 GRU
        S_fused = torch.cat((x, S_int), dim=2)  # (T, N, F + F')
        S_fused = S_fused.permute(1, 0, 2)      # (N, T, F + F')
        _, final_state = self.gru(S_fused)       # final_state: (1, N, rnn_unit)
        final_representation = final_state.squeeze(0)  # (N, rnn_unit)

        return final_representation

    def time_interaction(self, x, gamma_tem):
        """
        时间交互检测器
        :param x: 输入数据，形状为 (T, N, F)
        :param gamma_tem: 时间动态权重，形状为 (T,)
        :return: 时间交互特征，形状为 (T, N, F)
        """
        T, N, F = x.size()
        # 对于每个特征 f，计算时间交互
        S_tem = torch.zeros(T, N, F, device=x.device)
        for f in range(F):
            temp = x[:, :, f]  # (T, N)
            temp = torch.matmul(self.W_tem1, temp)  # (F', N)
            temp = torch.matmul(self.W_tem2, temp)  # (T, N)
            S_tem[:, :, f] = torch.sigmoid(temp) * gamma_tem.view(T, 1)  # (T, N)
        return S_tem

    def channel_interaction(self, x, gamma_cha):
        """
        特征交互检测器
        :param x: 输入数据，形状为 (T, N, F)
        :param gamma_cha: 特征动态权重，形状为 (F,)
        :return: 特征交互特征，形状为 (T, N, F)
        """
        T, N, F = x.size()
        # 对于每个时间步 t，计算特征交互
        S_cha = torch.zeros(T, N, F, device=x.device)
        for t in range(T):
            temp = x[t, :, :]  # (N, F)
            temp = torch.matmul(temp, self.W_cha1)  # (N, F')
            temp = torch.matmul(temp, self.W_cha2)  # (N, F)
            S_cha[t, :, :] = torch.sigmoid(temp) * gamma_cha.view(1, F)  # (N, F)
        return S_cha

    def feature_fusion(self, S_tem, S_cha):
        """
        时间交互和特征交互的融合
        :param S_tem: 时间交互特征，形状为 (T, N, F)
        :param S_cha: 特征交互特征，形状为 (T, N, F)
        :return: 融合后的特征，形状为 (T, N, F)
        """
        return S_tem * S_cha

    def dual_gate(self, S_fused):
        T, N, F = S_fused.size()
        S_int = torch.zeros(T, N, self.n_hid, device=S_fused.device)
        for t in range(T):
            temp = S_fused[t, :, :]  # (N, F)
            gate = torch.sigmoid(torch.matmul(temp, self.W_t.T))  # (N, F)
            filter = torch.tanh(torch.matmul(temp, self.G_t.T))  # (N, F)
            S_int[t, :, :] = gate * filter  # (N, F)
        return S_int




