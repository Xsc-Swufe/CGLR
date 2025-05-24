''' Define the HGTAN model '''
import torch

import torch.nn as nn
#from training.mytools import *
import torch.nn.functional as F
from CGLR.layers import ConditionGraphRoutingNetwork,NoiseAwareRelationInference,TemporalChannelInteractionFusionModule





class CGLR(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            num_stock, rnn_unit, n_hid, n_class,
            feature,
            d_word_vec, d_model,dropout,
            tgt_emb_prj_weight_sharing, use_hidden_rel, window_size, max_step, num_path, top_k):

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


        self.First_module = TemporalChannelInteractionFusionModule(feature, rnn_unit,window_size, n_hid)
        self.second_module = NoiseAwareRelationInference(rnn_unit,n_hid, epsilon=1)
        self.cgrn = ConditionGraphRoutingNetwork(rnn_unit,n_hid, num_path, top_k, max_step)

        self.out_1 = nn.Linear(n_hid+rnn_unit, n_hid)
        self.out_2 = nn.Linear(n_hid, n_class)
    def forward(self,src_seq1,matrix,matrix2):
        stock_num = src_seq1.size(1)
        seq_len = src_seq1.size(0)
        dim = src_seq1.size(2)

        if torch.isnan(src_seq1).any():
            print("src_seq1 中存在 NaN 值！")

        src_seq=src_seq1
        new_src_seq = self.First_module(src_seq1)

        new_new_src_seq = self.second_module(new_src_seq)




        new_src_seq = F.dropout(new_src_seq, self.dropout, training=self.training)

        H = self.cgrn(new_src_seq, matrix, new_new_src_seq)

        combined_output = torch.cat((new_src_seq, H), dim=1)

        seq_logit = F.elu(self.out_1(combined_output))
        seq_logit = F.dropout( seq_logit, self.dropout, training=self.training)

        output =self.out_2(seq_logit)
        #print(sum(output))

        return output





