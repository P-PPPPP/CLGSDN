import torch
import torch.nn as nn
import torch.nn.functional as F
from .lightcts_model import ttnet
from utils.data_utils.AdjProvider import calculate_dense_randomwalk_matrix

class lightcts_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(lightcts_api, self).__init__()
        self.configs= configs
        self.date_proj = nn.Linear(configs['dataset']['c_date'],1)
        self.num_nodes = configs['dataset']['n_nodes']
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
        assert graph_generator is None

    def init_model(self):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['out_dim'] = self.configs['exp']['c_out']
        model_configs['in_dim'] = self.configs['exp']['c_in'] + 1
        model_configs['pred_len'] = self.configs['exp']['pred_len']
        self.model = ttnet(model_configs, self.adjs)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        addi_loss = 0.0
        x = self.data_embedding(seq_x, seq_x_mark)
        predicts = self.model(x, epoch=args['epoch'])
        predicts = predicts.permute(0,3,2,1)
        return predicts, addi_loss
    
    def data_embedding(self, x, x_mark):
        x_mark = self.date_proj(x_mark).unsqueeze(-1).repeat(1,1,self.num_nodes,1).transpose(1,3)
        x = torch.cat((x,x_mark),dim=1)
        return x