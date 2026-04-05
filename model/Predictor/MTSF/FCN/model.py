import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .block import FreqConv, Indepent_Linear, TimeEmbedding, ChannelsShare

class Model(nn.Module):
    def __init__(self, configs, **args):
        nn.Module.__init__(self)
        self.params(configs)
        self.fconv1 = FreqConv(6, self.inp_len, self.inp_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fconv2 = FreqConv(6, self.pred_len, self.pred_len, kernel_size=self.kernel_size, dilation=self.dilation, order=self.order)
        self.fc_idp = Indepent_Linear(self.inp_len, self.pred_len, self.channels, True, self.dp_rate)
        # self.fc_idp = nn.Linear(self.inp_len, self.pred_len)
        #self.time_emb = TimeEmbedding(self.inp_len, self.pred_len, self.c_date, self.channels, configs['time_emb'], self.dp_rate)
        self.time_emb = TimeEmbedding(self.inp_len, self.pred_len, self.c_date, self.channels, configs['time_emb'], dp_rate=0)
        self.chan_share_x = ChannelsShare(self.inp_len, self.channels, layers=self.c_share_layers, 
                                          hid_dim=self.c_emb_dim,device=configs['device'], dp_rate=self.dp_rate)
        self.chan_share_y = ChannelsShare(self.pred_len, self.channels, device=configs['device'], dp_rate=self.dp_rate)
        
    def params(self, configs):
        self.c_in = configs['c_in']
        self.order = configs['order']
        self.c_out = configs['c_out']
        self.channels = configs['c_in']
        self.c_date = configs['c_date']
        self.dp_rate = configs['dropout']
        self.n_nodes = configs['n_nodes']
        self.inp_len = configs['inp_len']
        self.pred_len = configs['pred_len']
        self.dilation = configs['dilation']
        self.c_emb_dim = configs['c_emb_dim']
        self.kernel_size = configs['kernel_size']
        self.c_share_layers = configs['c_share_layers']
        self.dname = configs['d_name']

    def forward(self, x, x_mark, y_mark, **args):
        x_t, y_t = self.time_emb(x_mark, y_mark)
        x_c = self.chan_share_x(x+x_t)
        h_x = self.fconv1(x, x_t, x_c)
        # 
        h_y = self.fc_idp(h_x)
        # 
        y_c = self.chan_share_y(h_y+y_t)
        y = self.fconv2(h_y, y_t, y_c)
        loss = 0.0
        return y, loss, x_t, y_t