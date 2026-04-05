# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)
        return S_normalized

class cheb_conv_withSAt(nn.Module):
    def __init__(self, configs, K, cheb_polynomials, in_channels, out_channels, with_GraphGen=False):
        super(cheb_conv_withSAt, self).__init__()
        self.with_GraphGen = with_GraphGen
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = configs['device']
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, spatial_attention, adjs):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)
            for k in range(self.K):
                if self.with_GraphGen:
                    T_k = adjs[:,k,:,:]
                else:
                    T_k = self.cheb_polynomials[k]  # (N,N)
                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)
    
class Spatial_Conv(nn.Module):
    def __init__(self,configs, in_channels, nb_chev_filter):
        super(Spatial_Conv,self).__init__()
        self.configs = configs
        support_len = configs['n_prob']
        num_adjmat = support_len+1
        # functional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(configs['dropout'])
        # projection
        self.mlp = nn.Linear(num_adjmat*in_channels, nb_chev_filter)
    
    def forward(self,x,adjs):
        B,N,D,S = x.size()
        out = [x.unsqueeze(1)]
        # Graph Convolution
        x1 = torch.einsum('BHMN,BNDS->BHMDS',(adjs,x))
        out.append(x1)
        h = torch.cat(out,dim=1)
        h = h.permute(0,2,4,1,3).reshape(B,N,S,-1)
        h = self.dropout(self.mlp(h))
        # (b, N, F_out, T)
        return self.relu(h).transpose(2,3)

class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, T, F_in) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class ASTGCN_block(nn.Module):
    def __init__(self, configs, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, 
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps, with_GraphGen=False):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)

        self.with_GraphGen = with_GraphGen

        self.cheb_conv_SAt = cheb_conv_withSAt(configs, K, cheb_polynomials, in_channels, nb_chev_filter, with_GraphGen=with_GraphGen)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上

    def forward(self, x, adjs=None):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        
        # SAt
        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At, adjs)  # (b,N,F,T)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        return x_residual


class ASTGCN_submodule(nn.Module):
    def __init__(self, configs, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, with_GraphGen=False):
        super(ASTGCN_submodule, self).__init__()
        self.BlockList = nn.ModuleList([ASTGCN_block(configs, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, 
                                                     time_strides, cheb_polynomials, num_of_vertices, len_input, with_GraphGen=with_GraphGen)])
        self.BlockList.extend([ASTGCN_block(configs, DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 
                                            1, cheb_polynomials, num_of_vertices, len_input//time_strides, with_GraphGen=with_GraphGen) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, inputs, adjs=None):
        B,C,N,S = inputs.size()
        x = inputs.permute(0,2,1,3)
        for block in self.BlockList:
            x = block(x,adjs)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return output.reshape(B,C,S,N).transpose(2,3)
