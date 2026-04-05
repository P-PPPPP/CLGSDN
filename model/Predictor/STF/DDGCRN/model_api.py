import torch
import torch.nn as nn
from .DDGCRN import DDGCRN

class ddgcrn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(ddgcrn_api, self).__init__()
        self.configs = configs
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator, configs)
        self.init_model(raw_adjs)

    def init_adjs(self, raw_adjs):
        self.adjs = raw_adjs

    def init_GSL(self, graph_generator, configs):
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        if configs['graphgenerator'] is not None:
            assert configs['graphgenerator']['n_prob'] == 1, 'The parameter \"n_prob={}\" should be equal to \"K={}\", change \"n_prob\" to \"{}\"'\
                .format(configs['graphgenerator']['n_prob'],1,1)

    def init_model(self, raw_adjs):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['pred_len'] = self.configs['exp']['pred_len']
        model_configs['inp_len'] = self.configs['exp']['inp_len']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        self.model = DDGCRN(model_configs, raw_adjs, self.with_GraphGen)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            adjs = adjs.squeeze(1)
            addi_loss = graph_loss
        else:
            adjs = None
            addi_loss = 0.0
        x = self.make_data(seq_x, seq_x_mark)
        predicts = self.model(x, adjs=adjs)
        predicts = predicts.transpose(1,3)
        return predicts, addi_loss
    
    def make_data(self, x, x_mark):
        N = x.size(2)
        x = x.transpose(1,3)
        x_mark = x_mark[:,:,1:3].unsqueeze(2).repeat(1,1,N,1)
        x = torch.cat((x,x_mark),dim=-1)
        return x