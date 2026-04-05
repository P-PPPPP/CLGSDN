import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, ModuleList
import torch.nn.functional as F
from .transformer_model import Transformer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class MyTransformer(nn.Module):
    def __init__(self,hid_dim, layers, heads=8):
        super().__init__()
        self.heads = heads
        self.layers = layers
        self.hid_dim = hid_dim
        self.trans = Transformer(hid_dim, heads, layers)

    def forward(self, x, mask):
        x = self.trans(x, mask)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ttnet(nn.Module):
    def __init__(self, configs, supports):
        super(ttnet, self).__init__()
        dropout=0.1
        in_dim=configs['in_dim']
        out_dim=12
        hid_dim=32
        layers=8
        cnn_layers=4
        group=4
        self.start_conv = Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        self.cnn_layers = cnn_layers
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        self.group=group
        D = [1, 2, 4, 8]
        additional_scope = 1
        receptive_field = 1
        for i in range(self.cnn_layers):
            self.filter_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            self.gate_convs.append(Conv2d(hid_dim, hid_dim, (1, 2), dilation=D[i], groups=group))
            receptive_field += additional_scope
            additional_scope *= 2
        self.receptive_field=receptive_field
        depth = list(range(self.cnn_layers))
        self.bn = ModuleList([BatchNorm2d(hid_dim) for _ in depth])

        self.end_conv1 = nn.Linear(hid_dim, hid_dim*4)
        self.end_conv2 = nn.Linear(hid_dim*4, out_dim)

        mask0 = supports[0].detach()
        mask1 = supports[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0
        # self.spatial_conv = Spatial_Conv(hid_dim,hid_dim,configs['num_adjs'],dropout)
        self.network = MyTransformer(hid_dim, layers=layers, heads=8)
        self.se=SELayer(hid_dim)
        self.se2=SELayer(hid_dim)


    def forward(self, input, epoch=0):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(input)
        skip = 0
        for i in range(self.cnn_layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            if self.group !=1:
                x = channel_shuffle(x,self.group)
            try:
                skip += x[:, :, :, -1:]
            except:
                skip = 0
            if i == self.cnn_layers-1:
                break
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = torch.squeeze(skip, dim=-1)
        x = self.se(x)
        x = x.transpose(1, 2)
        # else:
        x_residual = x
        x = self.network(x, self.mask)
        x = x + x_residual
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        return x.transpose(1, 2).unsqueeze(-1)

class Spatial_Conv(nn.Module):
    # '''The graph convolution speed of Graph wavenet is too slow, use this for instead'''
    def __init__(self, c_in, c_out, num_adjmat, dropout):
        super(Spatial_Conv,self).__init__()
        # functional
        self.relu = nn.ReLU()
        # projection
        self.mlp = nn.Linear(c_in*(num_adjmat+1),c_out)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(num_adjmat)
    
    def forward(self,x,adjs):
        B,N,D = x.size()
        out = [x.unsqueeze(1)]
        # Graph Convolution
        x1 = torch.einsum('BHNM,BMD->BHND',(adjs,x))
        x1 = self.bn(x1)
        out.append(x1)
        h = torch.cat(out,dim=1)
        h = h.reshape(B,N,-1)
        h = self.mlp(h)
        y = self.dropout(h)
        y = x + y
        return y
