import torch
import torch.nn as nn
from .model import STSGCN
import numpy as np

class fogs_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(fogs_api, self).__init__()
        self.configs = configs
        self.adjs = fixed_adjs[:configs['model']['num_adjs']].repeat(configs['exp']['batch_size'],1,1)
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        model_configs = self.load_configs()
        
        adj_dtw = torch.eye(configs['dataset']['n_nodes'])
        local_adj = construct_adj_fusion(fixed_adjs[0].cpu().numpy(), adj_dtw.cpu().numpy(), steps=configs['model']['strides'])  # STFGNN
        local_adj = torch.FloatTensor(local_adj)

        self.model = STSGCN(model_configs, local_adj)
        if configs['graphgenerator'] is not None:
            assert configs['graphgenerator']['n_prob'] == configs['model']['num_adjs'], 'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                .format(configs['graphgenerator']['n_prob'],configs['model']['num_adjs'],configs['model']['num_adjs'])

    def load_configs(self):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['seq_len'] = self.configs['exp']['pred_len']
        model_configs['data_channels'] = len(self.configs['dataset']['choise_channels'])
        return model_configs
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            adjs = adjs.squeeze(1)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x.transpose(1,3), adjs)
        predicts = predicts.unsqueeze(-1).transpose(1,3)
        return predicts, addi_loss


def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

    ----------
    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    adj = np.zeros([N * steps] * 2)  # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    #'''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    #'''
    adj[3 * N: 4 * N, 0:  N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0: N, 3 * N: 4 * N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]


    for i in range(len(adj)):
        adj[i, i] = 1

    return adj