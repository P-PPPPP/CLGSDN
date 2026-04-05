import torch
import torch.nn as nn
from .gw_block import gwnet
from utils.data_utils.AdjProvider import calculate_dense_randomwalk_matrix

class gw_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(gw_api, self).__init__()
        self.configs = configs
        self.fc = nn.Linear(5,1) # Time embedding
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator)
        self.init_model()

    def init_adjs(self, raw_adjs):
        adj1 = torch.tensor(calculate_dense_randomwalk_matrix(raw_adjs.cpu()),
                             device=self.configs['exp']['device'], dtype=self.configs['exp']['dtype'])
        adj2 = torch.tensor(calculate_dense_randomwalk_matrix(raw_adjs.cpu().T),
                             device=self.configs['exp']['device'], dtype=self.configs['exp']['dtype'])
        self.adjs = torch.stack([adj1,adj2],dim=0)

    def init_GSL(self, graph_generator):
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        if self.with_GraphGen:
            assert self.configs['graphgenerator']['n_prob'] == self.configs['model']['num_adjs']+1,\
                  'The parameter \"n_prob: {}\" should be equal to \"num_adjs: {}\", change \"n_prob\" to \"{}\"'\
                  .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['num_adjs']+1, self.configs['model']['num_adjs']+1)

    def init_model(self):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['out_dim'] = self.configs['exp']['pred_len'] * len(self.configs['exp']['select_channels'])
        self.model = gwnet(model_configs, self.with_GraphGen)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        x = self.embedding(seq_x,seq_x_mark)
        predicts = self.model(x, adjs)
        return predicts, addi_loss
    
    def embedding(self, x, x_mark):
        # return x
        x_mark = self.fc(x_mark).unsqueeze(-1).repeat(1,1,self.configs['dataset']['n_nodes'],1).transpose(1,3)
        x = torch.cat([x,x_mark],dim=1)
        return x
