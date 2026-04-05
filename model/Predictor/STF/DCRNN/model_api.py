import torch
import torch.nn as nn
from .dcrnn_model import DCRNNModel
import math

class dcrnn_api(nn.Module):
    def __init__(self, configs, graph_generator, raw_adjs):
        super(dcrnn_api, self).__init__()
        self.bs = 1
        self.fc = nn.Linear(5,1)
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
        # spatial temporal predictor
        
        if self.with_GraphGen:
            assert self.configs['graphgenerator']['n_prob'] == 2*self.configs['model']['num_adjs'], 'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                .format(self.configs['graphgenerator']['n_prob'], self.configs['model']['num_adjs'], self.configs['model']['num_adjs'])

    def init_model(self, raw_adjs):
        model_configs = self.configs['model']
        model_configs['device'] = self.configs['exp']['device']
        model_configs['batch_size'] = self.configs['exp']['batch_size']
        model_configs['nodes'] = self.configs['dataset']['n_nodes']
        model_configs['seq_len'] = self.configs['exp']['pred_len']
        model_configs['data_channels'] = self.configs['exp']['c_in']
        self.model = DCRNNModel(raw_adjs.cpu(), model_configs, self.with_GraphGen).to(self.configs['exp']['device'])
        return model_configs
    
    def _compute_sampling_threshold(self,global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        if self.with_GraphGen:
            adjs, graph_loss = self.graph_generator(seq_x,seq_x_mark)
            addi_loss = graph_loss
        else:
            adjs = None
            addi_loss = 0.0
        if self.training:
            self.bs += 1
        x ,y = self.embedding(seq_x, seq_x_mark), args['seq_y']
        teacher_forcing_ratio = self._compute_sampling_threshold(self.bs, self.configs['model']['cl_decay_steps'])
        target = args['seq_y']
        if not self.training:
            target = torch.zeros_like(target)
        predicts = self.model(source=x, target=args['seq_y'], teacher_forcing_ratio=teacher_forcing_ratio, adjs=adjs)
        predicts = predicts.permute(1,2,0).unsqueeze(1)
        return predicts, addi_loss
    
    def embedding(self, x, x_mark):
        # return x
        x_mark = self.fc(x_mark).unsqueeze(-1).repeat(1,1,self.configs['dataset']['n_nodes'],1).transpose(1,3)
        x = torch.cat([x,x_mark],dim=1)
        return x
