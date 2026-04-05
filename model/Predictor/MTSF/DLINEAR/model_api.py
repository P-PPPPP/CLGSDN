import torch
import torch.nn as nn
from .dlinear import Model 
class dlinear_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(dlinear_api, self).__init__()
        self.configs = configs
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        model_configs = self.load_configs()
        self.model = Model(model_configs)
        self.model_init()
        if configs['graphgenerator'] is not None:
            assert configs['graphgenerator']['n_prob'] == configs['model']['num_adjs'], 'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                .format(configs['graphgenerator']['n_prob'],configs['model']['num_adjs'],configs['model']['num_adjs'])
        
    def load_configs(self):
        model_configs = self.configs['model']
        model_configs['dtype'] = self.configs['exp']['dtype']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['seq_len'] = self.configs['exp']['inp_len']
        model_configs['pred_len'] = self.configs['exp']['pred_len']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        return model_configs
    
    def model_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, seq_y, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x.reshape(-1,self.configs['exp']['inp_len'], self.configs['exp']['c_in'])) #output:(batch,pred_len,channels)
        predicts = predicts.unsqueeze(2).transpose(1,3)  #output:(batch,channels,num_nodes,pred_len)
        return predicts, addi_loss



