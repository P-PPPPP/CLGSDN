import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn.functional as F
import torch.nn as nn

class AVWGCN(nn.Module):
    def __init__(self, configs, dim_in, dim_out, cheb_k, embed_dim, with_GraphGen=False):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.with_GraphGen = with_GraphGen
        if not self.with_GraphGen:
        # if True:
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        else:
            self.channels_mlp = nn.Linear((cheb_k+1)*dim_in, dim_out)
            self.dropout = nn.Dropout(configs['dropout'])
            
    def forward(self, x, node_embeddings, adjs):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        if not self.with_GraphGen:
            supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
            support_set = [torch.eye(node_num).to(supports.device), supports]
            # default cheb_k = 3
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
            supports = torch.stack(support_set, dim=0)
            x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
            x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
            weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
            bias = torch.matmul(node_embeddings, self.bias_pool)                         # N, dim_out
            x_gconved = torch.einsum('bnki,nkio->bno', x_g, weights) + bias                # b, N, dim_out
        else:
            B,N,S = x.size()
            out = [x.unsqueeze(1)]
            # Graph Convolution
            x1 = torch.einsum('BHMN,BND->BHMD',(adjs,x))
            out.append(x1)
            x_g = torch.cat(out,dim=1)
            x_g = x_g.permute(0,2,1,3).reshape(B,N,-1)
            x_gconved = self.channels_mlp(F.relu(x_g))
            x_gconved = self.dropout(x_gconved)
        return x_gconved