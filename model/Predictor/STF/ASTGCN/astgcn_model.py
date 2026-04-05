import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from .ASTGCN import ASTGCN_submodule
from model.DGCDN.DGCGN_GraphGenerator import Graph_Generator

class ASTGCN_MODEL(nn.Module):
    def __init__(self, configs, topo_graph):
        super(ASTGCN_MODEL, self).__init__()
        self.configs = configs
        num_adjs = configs['model_configs']['num_adjs']
        assert num_adjs == 1, 'hyper parameters <num_adjs> should be \'1\' in ASTGCN, but got \'{}\'.'.format(num_adjs)
        # spatial temporal predictor
        self.model = make_model(configs,topo_graph[0],with_DGCDN=False)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark):
        # generate graph
        predicts = self.model(seq_x)
        return predicts,0.0
    
class ASTGCNADGCDN_MODEL(nn.Module):
    def __init__(self, configs, topo_graph):
        super(ASTGCNADGCDN_MODEL, self).__init__()
        self.configs = configs
        try:
            num_adjs =configs['model_configs']['num_adjs'] * configs['model_configs']['cheb_k'] 
            if not configs['n_prob'] == num_adjs:
                raise ValueError()
        except ValueError as e:
            if configs['print_info']:
                print('\nValue Error: hyper parameters <n_prob> should be \'{}\' in ASTGCN, but got \'{}\'.'.format(num_adjs,configs['n_prob']))
                print('\tSet <n_prob> to \'{}\''.format(num_adjs))
            configs['n_prob'] = num_adjs
        # generate graph
        self.graph_generator = Graph_Generator(configs,topo_graph)
        # spatial temporal predictor
        self.model = make_model(configs,None,with_DGCDN=True)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark):
        # generate graph
        adjs,loss = self.graph_generator(seq_x, seq_x_mark)
        predicts = self.model(seq_x,adjs)
        return predicts,loss
    
def make_model(configs, topo_graph, with_DGCDN=False):

    DEVICE = configs['device']
    num_of_vertices = configs['n_nodes']
    num_for_predict = configs['seq_len'] * configs['data_channels']
    len_input = configs['seq_len']

    model_configs = configs['model_configs']
    K = model_configs['cheb_k']
    in_channels = configs['data_channels']
    nb_block = model_configs['nb_block']
    nb_chev_filter = model_configs['num_of_chev_filters']
    nb_time_filter = model_configs['num_of_time_filters']
    time_strides = model_configs['time_conv_strides']

    adj_mx = topo_graph

    if with_DGCDN:
        cheb_polynomials = None
    else:
        L_tilde = scaled_Laplacian(adj_mx.cpu().numpy())
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
        
    model = ASTGCN_submodule(configs, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
                             time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, with_DGCDN=with_DGCDN)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])
