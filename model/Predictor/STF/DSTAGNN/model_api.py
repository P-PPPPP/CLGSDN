import torch
import torch.nn as nn
from .dstagnn import DSTAGNN_submodule
from . utils import scaled_Laplacian,cheb_polynomial
import torch.nn.functional as F

class dstagnn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(dstagnn_api, self).__init__()
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
            assert self.configs['graphgenerator']['n_prob'] == self.configs['model']['K'], 'The parameter \"n_prob={}\" should be equal to \"K={}\", change \"n_prob\" to \"{}\"'\
                .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['K'], self.configs['model']['K'])

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
            adjs = F.relu(adjs)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x.transpose(1,2), adjs)
        predicts = predicts.unsqueeze(1)
        return predicts, addi_loss

def make_model(configs, fix_adjs, with_GraphGen):
    DEVICE = configs['device']
    num_of_d = 1
    nb_block = configs['nb_block']
    in_channels = configs['in_channels']
    K = configs['K']
    nb_chev_filter = configs['nb_chev_filter']
    nb_time_filter = configs['nb_time_filter']
    time_strides = configs['time_strides']
    num_for_predict = configs['num_for_predict']
    len_input = configs['len_input']
    num_of_vertices = configs['num_nodes']
    d_model = configs['d_model']
    d_k = configs['d_k']
    d_v = configs['d_v']
    n_heads = configs['n_heads']
    
    adj_mx = fix_adjs
    adj_TMD = torch.eye(num_of_vertices)
    adj_pa = torch.eye(num_of_vertices)
    
    L_tilde = scaled_Laplacian(adj_mx.cpu().numpy())
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = DSTAGNN_submodule(DEVICE, num_of_d, nb_block, in_channels,
                             K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                             adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads,with_GraphGen)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model