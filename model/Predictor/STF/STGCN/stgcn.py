import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes, num_adjmat):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        B,N,S,D = t.size()
        lfs = torch.einsum("kij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)

class STGCN(nn.Module):
    def __init__(self, configs):
        super(STGCN, self).__init__()
        num_nodes, num_features, num_timesteps_input, num_timesteps_output = configs['n_nodes'], configs['data_channels'], configs['seq_len'], configs['seq_len']
        num_adjmat = configs['n_adjs']
        self.blocks = nn.ModuleList()
        block = STGCNBlock(in_channels=num_features, out_channels=configs['n_channels'],
                               spatial_channels=configs['spatial_channels'], num_nodes=num_nodes, num_adjmat=num_adjmat)
        self.blocks.append(block)
        for _ in range(1,configs['blocks']):
            block = STGCNBlock(in_channels=configs['n_channels'], out_channels=configs['n_channels'],
                               spatial_channels=configs['spatial_channels'], num_nodes=num_nodes, num_adjmat=num_adjmat)
            self.blocks.append(block)
        
        self.last_temporal = TimeBlock(in_channels=configs['n_channels'], out_channels=configs['n_channels'])
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output*configs['data_channels'])

    def forward(self, X, A_hat):
        B,C,N,S = X.size()
        h = X.permute(0,2,3,1)
        for block in self.blocks:
            h = block(h, A_hat)
        h = self.last_temporal(h)
        out = self.fully(h.reshape((h.shape[0], h.shape[1], -1)))
        return out.reshape(B,N,S,C).permute(0,3,1,2)