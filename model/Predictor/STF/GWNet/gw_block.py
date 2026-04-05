import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Universal.My_functional as my_nn

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        # out.append(self.nconv(x,support[-1]))
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Spatial_Conv(nn.Module):
    # '''The graph convolution speed of Graph wavenet is too slow, use this for instead'''
    def __init__(self, c_in, c_out, num_adjmat, dropout):
        super(Spatial_Conv,self).__init__()
        # functional
        self.relu = nn.ReLU()
        # projection
        self.mlp = my_nn.linear_as_conv2d((num_adjmat+2)*c_in, c_out, (1,1), bias=True, dropout=dropout)
        self.dp = nn.Dropout(0.3)
    
    def forward(self,x,adjs):
        B,_,N,S = x.size()
        out = [x.unsqueeze(1)]
        # Graph Convolution
        x1 = torch.einsum('BHNM,BDMS->BHDNS',(adjs,x))
        out.append(x1)
        h = torch.cat(out,dim=1)
        h = h.reshape(B,-1,N,S)
        h = self.mlp(h)
        h = F.tanh(h)
        h = self.dp(h)
        y = x+h
        return y

class gwnet(nn.Module):
    def __init__(self, configs, with_GraphGen=False):
        super(gwnet, self).__init__()
        self.with_GraphGen = with_GraphGen
        # hyperparams
        self.device = configs['device']
        self.num_nodes = configs['num_nodes']
        self.out_dim = configs['out_dim']
        self.in_dim = configs['in_dim']
        self.dropout = configs['dropout']
        self.blocks = configs['blocks']
        self.layers = configs['layers']
        self.addaptadj = configs['addaptadj']
        self.residual_channels = configs['residual_channels']
        self.dilation_channels = configs['dilation_channels']
        self.skip_channels = configs['skip_channels']
        self.end_channels = configs['end_channels']
        self.kernel_size = configs['kernel_size']
        self.order = configs['order']
        self.num_adjs = configs['num_adjs']
        # module
        self.supports_len = self.num_adjs
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1        
    
        if self.addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device), requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device), requires_grad=True).to(self.device)
            self.supports_len += 1

        for _ in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1,self.kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2

                if not with_GraphGen:
                    self.gconv.append(gcn(self.dilation_channels, self.residual_channels, self.dropout, support_len=self.supports_len, order=self.order))
                else:
                    self.gconv.append(Spatial_Conv(c_in=self.dilation_channels, c_out=self.residual_channels, num_adjmat=self.num_adjs, dropout=self.dropout))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                  out_channels=self.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, adjs=None):
        B,C,N,S = input.size()
        if S < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-S,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        # spatial conv
        supports = adjs
        if not self.with_GraphGen and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports = torch.cat((supports, adp.unsqueeze(0)), dim=0)

        for i in range(self.blocks * self.layers):
            residual = x
            # '''dilated convolution'''
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # '''parametrized skip connection'''
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            # ''' Graph Conv'''
            x = self.gconv[i](x, supports)
            # '''residual and batch norm'''
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.squeeze(-1).reshape(B,-1,S,N).permute(0,1,3,2)