import torch
import torch.nn as nn
import torch.nn.functional as F
from .MegaCRN import MegaCRN

class megacrn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(megacrn_api, self).__init__()
        self.adjs = raw_adjs
        self.date_proj = nn.Linear(configs['dataset']['c_date'],1)
        self.num_nodes = configs['dataset']['n_nodes']
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator)
        self.init_model(configs)

    def init_adjs(self, raw_adjs):
        self.adjs = raw_adjs

    def init_GSL(self, graph_generator):
        assert graph_generator is None

    def init_model(self, configs):
        model_configs = configs['model']
        model_configs['device'] = configs['exp']['device']
        model_configs['num_nodes'] = configs['dataset']['n_nodes']
        model_configs['out_dim'] = configs['exp']['c_out']
        model_configs['in_dim'] = configs['exp']['c_in'] + 1
        model_configs['pred_len'] = configs['exp']['pred_len']
        self.model = MegaCRN(model_configs)
 
    def load_configs(self, configs):
        model_configs = configs['model']
        model_configs['device'] = configs['exp']['device']
        model_configs['num_nodes'] = configs['dataset']['n_nodes']
        model_configs['out_dim'] = len(configs['dataset']['choise_channels'])
        model_configs['in_dim'] = len(configs['dataset']['choise_channels']) + 1
        model_configs['pred_len'] = configs['exp']['pred_len']
        return model_configs
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        addi_loss = 0.0
        x = seq_x.permute(0,3,2,1)
        y = args['seq_y'].permute(0,3,2,1)
        x, label, y_conv = self.data_embedding(x, seq_x_mark, y, seq_y_mark)
        predicts,_,_,_,_ = self.model(x, y_conv, label, batches_seen=args['epoch'], seq_x_mark=seq_x_mark)
        predicts = predicts.permute(0,3,2,1)
        return predicts, addi_loss
    
    def data_embedding(self, x, x_mark, y, y_mark):
        x_mark = self.date_proj(x_mark).unsqueeze(-1).repeat(1,1,self.num_nodes,1)
        y_mark = self.date_proj(y_mark).unsqueeze(-1).repeat(1,1,self.num_nodes,1)
        x = torch.cat((x,x_mark),dim=-1)
        return x,y,y_mark