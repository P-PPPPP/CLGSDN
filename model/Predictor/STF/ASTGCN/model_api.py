import torch
import numpy as np
import torch.nn as nn
from scipy.sparse.linalg import eigs
from .ASTGCN import ASTGCN_submodule

class astgcn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(astgcn_api, self).__init__()
        self.configs = configs
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator)
        self.init_model(raw_adjs)

    def init_adjs(self, raw_adjs):
        self.adjs = raw_adjs
    
    def init_GSL(self, graph_generator):
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        if self.with_GraphGen:
            assert self.configs['graphgenerator']['n_prob'] == self.configs['model']['cheb_k'], 'The parameter \"n_prob={}\" should be equal to \"cheb_k={}\", change \"n_prob\" to \"{}\"'\
                .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['cheb_k'], self.configs['model']['cheb_k'])
    
    def init_model(self, raw_adjs):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['seq_len'] = self.configs['exp']['pred_len']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        self.model = make_model(model_configs, raw_adjs, self.with_GraphGen)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            adjs = adjs.squeeze(1)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x, adjs)
        return predicts, addi_loss
    
def make_model(configs, raw_graph, with_GraphGen=False):

    DEVICE = configs['device']
    num_of_vertices = configs['num_nodes']
    num_for_predict = configs['seq_len'] * configs['data_channels']
    len_input = configs['seq_len']

    K = configs['cheb_k']
    in_channels = configs['data_channels']
    nb_block = configs['nb_block']
    nb_chev_filter = configs['num_of_chev_filters']
    nb_time_filter = configs['num_of_time_filters']
    time_strides = configs['time_conv_strides']

    adj_mx = raw_graph

    if with_GraphGen:
        cheb_polynomials = None
    else:
        L_tilde = scaled_Laplacian(adj_mx.cpu().numpy())
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
        
    model = ASTGCN_submodule(configs, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, 
                             time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices, with_GraphGen=with_GraphGen)

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
