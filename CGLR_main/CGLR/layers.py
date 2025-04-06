
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
        :param input_dim:  (F)
        :param output_dim:  (F')
        """
        super(GATMechanism, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)  # [F, F']
        self.a = nn.Parameter(torch.randn(2 * output_dim + input_dim, 1))  # [2F' + F, 1]

    def forward(self, V, E, time_enc):
        """
        :param V:  [N, F]
        :param E:  [N, N]
        :param time_enc:  [N, N, F]
        :return:  [N, N, F']
        """
        N = V.size(0)
        V_i = V.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        V_j = V.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        M = self.W(V_j)  # [N, N, F']，
        attention_scores = self.compute_attention(V_i, V_j, time_enc)  # [N, N]
        H = (attention_scores.unsqueeze(-1) * M) * E.unsqueeze(-1)  # [N, N, F']，
        return H

    def compute_attention(self, V_i, V_j, time_enc):
        
        V_i_transformed = self.W(V_i)  # [N, N, F']
        V_j_transformed = self.W(V_j)  # [N, N, F']
        feature_concat = torch.cat([V_i_transformed, V_j_transformed, time_enc], dim=-1).float()  # [N, N, 2F' + F]
        attention_score = torch.matmul(feature_concat, self.a).squeeze(-1)  # [N, N]
        attention_score = F.leaky_relu(attention_score)
        attention_score = F.softmax(attention_score, dim=-1)  
        return attention_score


class ConditionGraphRoutingNetwork(nn.Module):
    def __init__(self, rnn_unit, n_hid, K, Top_K):
        """
         CGRN
        :param rnn_unit:  (F)
        :param n_hid:  (F')
        :param K:  (L)
        :param Top_K
        """
        super(ConditionGraphRoutingNetwork, self).__init__()
        self.n_hid = n_hid  # F'
        self.K = K  # L
        self.Top_K = Top_K
        self.rnn_unit = rnn_unit  # F

       
        self.W_a = nn.Parameter(torch.randn(K, 3 * rnn_unit))  # [L, 3F]
   
        self.gat_layers = nn.ModuleList([GATMechanism(rnn_unit, n_hid) for _ in range(K)])
    
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_hid) for _ in range(K)])

    def time_encoding(self, delta_t):
        """
        :param delta_t:  [N, N]
        :return:  [N, N, F]
        """
        dim_t = torch.arange(0, self.rnn_unit // 2, device=delta_t.device).float()
        div_term = 10000 ** (2 * dim_t / self.rnn_unit).float()
        sin_term = torch.sin(delta_t.unsqueeze(-1) / div_term)  # [N, N, F/2]
        cos_term = torch.cos(delta_t.unsqueeze(-1) / div_term)  # [N, N, F/2]
        return torch.cat([sin_term, cos_term], dim=-1)  # [N, N, F]

    def top_k_selection(self, scores):
        """
        Top-K 
        :param scores:  [N, N, L]
        :return:  [N, N, L] 
        """
        _, indices = torch.topk(scores, self.Top_K, dim=-1)  # [N, N, Top_K]
        mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)  # [N, N, L]
        return scores * mask, mask

    def forward(self, V, Delta_t, E):
        """
        :param V:  [N, F]
        :param Delta_t:  [N, N]
        :param E:  [N, N]
        :return:  [N, F']
        """
        N = V.size(0)
        time_enc = self.time_encoding(Delta_t)  # [N, N, F]

        
        V_i = V.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        V_j = V.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        feature_concat = torch.cat([V_i, V_j, time_enc], dim=-1).float()  # [N, N, 3F]
        scores = F.leaky_relu(torch.matmul(feature_concat, self.W_a.T))  # [N, N, L]

        
        scores, mask = self.top_k_selection(scores)  # [N, N, L]
        scores_exp = torch.exp(scores) * mask  # [N, N, L]
        scores_sum = scores_exp.sum(dim=-1, keepdim=True) + 1e-8  # [N, N, 1]
        p = scores_exp / scores_sum  # [N, N, L]，

      
        messages = torch.zeros(N, N, self.n_hid, device=V.device)  # [N, N, F']
        for k in range(self.K):
            if mask[..., k].any():  
                H_k = self.gat_layers[k](V, E, time_enc)  # [N, N, F']
                messages += (E.unsqueeze(-1) * mask[..., k].unsqueeze(-1)) * (p[..., k].unsqueeze(-1) * H_k)

    
        V_out = messages.sum(dim=1)  # [N, F']
        return V_out

# 噪声感知关系推断模块
class NoiseAwareRelationInference(nn.Module):
    def __init__(self, rnn_unit, n_dim, epsilon=0.5):
        super(NoiseAwareRelationInference, self).__init__()
        
        self.W_r = nn.Parameter(torch.randn(2 * rnn_unit, n_dim))
        self.a_r = nn.Parameter(torch.randn(n_dim))
        self.epsilon = epsilon  # 噪声过滤因子

    def forward(self, X):
        """
        Input:
        X: [N, F] 
        Output:
        E: [N, N] 
        """
        N, Fea = X.size()

        X_i = X.unsqueeze(1).repeat(1, N, 1)  # [N, N, F]
        X_j = X.unsqueeze(0).repeat(N, 1, 1)  # [N, N, F]
        X_combined = torch.cat((X_i, X_j), dim=-1)  # [N, N, 2F]

        transformed = torch.matmul(X_combined, self.W_r)  # [N, N, F']

        R = torch.matmul(transformed, self.a_r)  # [N, N]

        R = F.leaky_relu(R)  # [N, N]

        R = (R - R.mean()) / (R.std() + 1e-5)  

        E = F.softmax(R, dim=1)  #  [N, N]

        E_mean = E.mean()  # 

        threshold = (1 / (self.epsilon)) * (E_mean)

        filtered_E = torch.where(E >= threshold, E, torch.zeros_like(E))
        return filtered_E























# 动态超前滞后检测模型
class DynamicLeadLagModel(nn.Module):
    def __init__(self, F, epsilon=0.5, max_lag_step=3):
        super(DynamicLeadLagModel, self).__init__()
        self.relation_inference = NoiseAwareRelationInference(F=F, epsilon=epsilon)
        self.auto_corr_module = AutoCorrelationLeadLagEffect(max_lag_step=max_lag_step)

    def forward(self, X):
        """
        Input:
        X: [T, N, F] 
        Output:
        lead_lag_times: [N, N] 
        lead_steps: [N, N] 
        """
        T, N, F = X.size()

        
        E = self.relation_inference(X)

       
        lead_lag_times = torch.zeros(N, N)
        lead_steps = torch.zeros(N, N)

        
        for i in range(N):
            for j in range(i + 1, N):  
               
                v_i = X[:, i, :]  # [T, F]
                v_j = X[:, j, :]  # [T, F]

                
                if E[i, j] > 0 and lead_lag_times[i, j] == 0:  
                    
                    lag_time, optimal_lag_step = self.auto_corr_module(v_i, v_j)

                    
                    lead_steps[i, j] = optimal_lag_step
                    lead_steps[j, i] = -optimal_lag_step  

                    
                    lead_lag_times[i, j] = lag_time
                    lead_lag_times[j, i] = -lag_time  

        return lead_lag_times, E  





import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalChannelInteractionFusionModule(nn.Module):
    def __init__(self, feature_dim, rnn_unit, time_steps, n_hid):
        """
        
        :param feature_dim:  C
        :param rnn_unit: GRU  F 
        :param time_steps:  T
        :param n_hid:  F' 
        """
        super(TemporalChannelInteractionFusionModule, self).__init__()

        self.feature_dim = feature_dim  # C
        self.rnn_unit = rnn_unit       # F
        self.time_steps = time_steps   # T
        self.n_hid = n_hid             # F'

      
        self.W_tem1 = nn.Parameter(torch.randn(n_hid, time_steps))  # F' × T
        self.W_tem2 = nn.Parameter(torch.randn(time_steps, n_hid))  # T × F'

   
        self.W_cha1 = nn.Parameter(torch.randn(feature_dim, n_hid))  # C × F'
        self.W_cha2 = nn.Parameter(torch.randn(n_hid, feature_dim))  # F' × C

      
        self.W_tem3 = nn.Parameter(torch.randn(n_hid, feature_dim))  # F' × C
        self.b_tem3 = nn.Parameter(torch.randn(time_steps))               # F'
        self.W_cha3 = nn.Parameter(torch.randn(n_hid, time_steps))   # F' × T
        self.b_cha3 = nn.Parameter(torch.randn(feature_dim))               # F'

       
        self.W_t = nn.Parameter(torch.randn(n_hid, feature_dim))  # (F', F)
        self.G_t = nn.Parameter(torch.randn(n_hid, feature_dim))  # (F', F)

       
        self.gru = nn.GRU(input_size=feature_dim + n_hid, hidden_size=rnn_unit,
                         num_layers=1, batch_first=True, bidirectional=False)

    def forward(self, x):
        """
        :param x:  (T, N, F) 
        :return:  (N, rnn_unit)
        """
        T, N, F = x.size()

       
        norm_x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)  # (T, N, F)
        z_t = norm_x.mean(dim=1)  # (T, F)

       
        gamma_tem = torch.softmax(torch.matmul(self.W_tem3, z_t.T) + self.b_tem3, dim=-1).mean(dim=0)  # (T,)
        gamma_cha = torch.softmax(torch.matmul(self.W_cha3, z_t) + self.b_cha3, dim=-1).mean(dim=0)    # (F,)

        
        S_tem = self.time_interaction(norm_x, gamma_tem)  # (T, N, F)

      
        S_cha = self.channel_interaction(norm_x, gamma_cha)  # (T, N, F)

       
        S_fused = self.feature_fusion(S_tem, S_cha)  # (T, N, F)

        
        S_int = self.dual_gate(S_fused)  # (T, N, F)

      
        S_fused = torch.cat((x, S_int), dim=2)  # (T, N, F + F')
        S_fused = S_fused.permute(1, 0, 2)      # (N, T, F + F')
        _, final_state = self.gru(S_fused)       # final_state: (1, N, rnn_unit)
        final_representation = final_state.squeeze(0)  # (N, rnn_unit)

        return final_representation

    def time_interaction(self, x, gamma_tem):
        """
        :param x:  (T, N, F)
        :param gamma_tem:  (T,)
        :return:  (T, N, F)
        """
        T, N, F = x.size()
        S_tem = torch.zeros(T, N, F, device=x.device)
        for f in range(F):
            temp = x[:, :, f] * gamma_tem.view(T, 1)  # (T, N)，apply gamma_tem
            temp = torch.matmul(self.W_tem1, temp)  # (F', T) @ (T, N) -> (F', N)
            temp = torch.matmul(self.W_tem2, temp)  # (T, F') @ (F', N) -> (T, N)
            S_tem[:, :, f] = torch.sigmoid(temp)  # (T, N)
        return S_tem

    def channel_interaction(self, x, gamma_cha):
        """
        :param x:  (T, N, F)
        :param gamma_cha:  (F,)
        :return:  (T, N, F)
        """
        T, N, F = x.size()
        S_cha = torch.zeros(T, N, F, device=x.device)
        for t in range(T):
            temp = x[t, :, :] * gamma_cha.view(1, F)  # (N, F)，apply gamma_cha
            temp = torch.matmul(temp, self.W_cha1)  # (N, F) @ (F, F') -> (N, F')
            temp = torch.matmul(temp, self.W_cha2)  # (N, F') @ (F', F) -> (N, F)
            S_cha[t, :, :] = torch.sigmoid(temp)  # (N, F)
        return S_cha

    def feature_fusion(self, S_tem, S_cha):
        """
        
        :param S_tem:  (T, N, F)
        :param S_cha:  (T, N, F)
        :return:  (T, N, F)
        """
        return S_tem * S_cha

    def dual_gate(self, S_fused):
        T, N, F = S_fused.size()
        S_fused = S_fused.permute(0, 2, 1)  # (T, F, N)
        gate = torch.sigmoid(torch.matmul(self.W_t, S_fused))  # (F', F) @ (T, F, N) -> (T, F', N)
        filter = torch.tanh(torch.matmul(self.G_t, S_fused))  # (F', F) @ (T, F, N) -> (T, F', N)
        S_int = gate * filter  # (F', N, T)
        S_int = S_int.permute(0, 2, 1)  # (T, N, F')
        return S_int




