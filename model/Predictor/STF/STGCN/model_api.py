import torch
import torch.nn as nn
from .stgcn import STGCN
    
class stgcn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(stgcn_api, self).__init__()
        self.configs = configs
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator)
        self.init_model()

    def init_adjs(self, raw_adjs):
        if raw_adjs is not None:
            adjs = raw_adjs.unsqueeze(0).repeat(self.configs['exp']['batch_size'],1,1)
        else:
            adjs = torch.eye(self.configs['dataset']['n_nodes']).unsqueeze(0).repeat(self.configs['exp']['batch_size'],1,1)
        self.adjs = adjs.to(device=self.configs['exp']['device'], dtype=self.configs['exp']['dtype'])

    def init_GSL(self,graph_generator):
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        if self.with_GraphGen:
            assert self.configs['graphgenerator']['n_prob'] == self.configs['model']['n_adjs'],\
                   'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                   .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['n_adjs'], self.configs['model']['num_adjs'])

    def init_model(self):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['n_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['seq_len'] = self.configs['exp']['pred_len']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        self.model = STGCN(model_configs)
    
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