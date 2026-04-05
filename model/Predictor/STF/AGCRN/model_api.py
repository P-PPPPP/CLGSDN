import torch.nn as nn
from .AGCRN import AGCRN

class agcrn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(agcrn_api, self).__init__()
        self.configs = configs
        self.init_adjs(raw_adjs)
        self.init_GSL(graph_generator)
        self.init_model()
        
    def init_adjs(self, raw_adjs):
        self.adjs = raw_adjs

    def init_GSL(self, graph_generator):
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        if self.with_GraphGen:
            assert self.configs['graphgenerator']['n_prob'] == self.configs['model']['num_adjs'], 'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['num_adjs'], self.configs['model']['num_adjs'])

    def init_model(self):
        model_configs = self.configs['model']
        model_configs['dtype'] = self.configs['exp']['dtype']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['seq_len'] = self.configs['exp']['inp_len']
        model_configs['num_nodes'] = self.configs['dataset']['n_nodes']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        self.model = AGCRN(model_configs, self.with_GraphGen)

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            addi_loss = graph_loss
        else:
            adjs = self.adjs
            addi_loss = 0.0
        predicts = self.model(seq_x.transpose(1,3), adjs)
        return predicts, addi_loss
