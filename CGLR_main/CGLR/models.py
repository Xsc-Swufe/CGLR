''' Define the HGTAN model '''
import torch

import torch.nn as nn
#from training.mytools import *
import torch.nn.functional as F
from CGLR.layers import ConditionGraphRoutingNetwork,NoiseAwareRelationInference,AutoCorrelationLeadLagEffect,DynamicLeadLagModel,TemporalChannelInteractionFusionModule





class CGLR(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            num_stock, rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model,dropout,
            tgt_emb_prj_weight_sharing,use_hidden_rel,window_size):

        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(feature, d_word_vec)
        self.use_hidden_rel = use_hidden_rel

        self.n_hid=n_hid

        self.tgt_word_prj = nn.Linear(rnn_unit+n_hid, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(2*rnn_unit, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        self.rnn2 = nn.GRU(feature,
                           rnn_unit,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)
        #nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1

        self.bn = nn.BatchNorm1d(rnn_unit)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid+rnn_unit)
        self.bn_in = nn.BatchNorm1d(feature)
        self.First_module = TemporalChannelInteractionFusionModule(feature, rnn_unit,window_size, n_hid)
        self.second_module = NoiseAwareRelationInference(rnn_unit,n_hid, epsilon=1)
        self.cgrn = ConditionGraphRoutingNetwork(rnn_unit,n_hid, 15, 6)
        self.ln = nn.LayerNorm(rnn_unit)
        self.ln2 = nn.LayerNorm(n_hid)
        self.ln_in = nn.LayerNorm(feature)
        self.ln3 = nn.LayerNorm(n_hid+rnn_unit)
        self.ln5 = nn.LayerNorm(rnn_unit)
        self.out_1 = nn.Linear(n_hid+rnn_unit, n_hid)
        self.out_2 = nn.Linear(n_hid, n_class)
    def forward(self,src_seq1,matrix,matrix2):
        stock_num = src_seq1.size(1)
        seq_len = src_seq1.size(0)
        dim = src_seq1.size(2)

        # src_seq1_flat = src_seq1.view(-1, dim)
        # src_seq = self.ln_in(src_seq1_flat)
        # src_seq = src_seq.view(seq_len, stock_num, -1)
        if torch.isnan(src_seq1).any():
            print("src_seq1 中存在 NaN 值！")

        # src_seq1rnn = src_seq1.permute(1, 0, 2)
        # _, rnn_output = self.rnn2(src_seq1rnn)
        # rnn_output = self.ln5(rnn_output.squeeze(0))
        # rnn_output = F.dropout(rnn_output, self.dropout, training=self.training)


        #src_seq = self.linear(src_seq_x)
        src_seq=src_seq1
        new_src_seq = self.First_module(src_seq1)
        new_src_seq = self.ln(new_src_seq)

        new_new_src_seq = self.second_module(new_src_seq)




        new_src_seq = F.dropout(new_src_seq, self.dropout, training=self.training)

        H = self.cgrn(new_src_seq, matrix, new_new_src_seq)
        #H = self.ln2(H)
        combined_output = torch.cat((new_src_seq, H), dim=1)
        #combined_output = self.ln3(combined_output)

        seq_logit = F.elu(self.out_1(combined_output))
        seq_logit = F.dropout( seq_logit, self.dropout, training=self.training)

        output =self.out_2(seq_logit)
        #print(sum(output))

        return output





