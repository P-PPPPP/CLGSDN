import torch
import torch.nn as nn
from .timesnet import Model

class timesnet_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(timesnet_api, self).__init__()
        self.configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        self.model = Model(self.configs)
        self.model_init()
        if configs['graphgenerator'] is not None:
            assert configs['graphgenerator']['n_prob'] == configs['model']['num_adjs'], 'The parameter \"n_prob={}\" should be equal to \"num_adjs={}\", change \"n_prob\" to \"{}\"'\
                .format(configs['graphgenerator']['n_prob'], configs['model']['num_adjs'], configs['model']['num_adjs'])
        
    def load_configs(self,configs):
        model_configs = configs['model']
        model_configs['c_in'] = configs['exp']['c_in']
        model_configs['c_out'] = configs['exp']['c_out']
        model_configs['dtype'] = configs['exp']['dtype']
        model_configs['device'] = configs['exp']['device']
        model_configs['num_nodes'] = configs['dataset']['n_nodes']
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
        x_enc, x_dec, x_enc_mark, x_dec_mark = self.data_reshape(seq_x,seq_y,seq_x_mark,seq_y_mark)
        predicts = self.model(x_enc, x_enc_mark, x_dec, x_dec_mark) #output:(batch,pred_len,channels)
        predicts = predicts.unsqueeze(2).transpose(1,3) #output:(batch,channels,num_nodes,pred_len)
        return predicts, addi_loss
    
    def data_reshape(self,x,y,x_mark,y_mark):
        assert x.shape[2] == 1
        x_enc = x.squeeze(2).transpose(1,2)
        x_enc_mark = x_mark.squeeze(1)
        x_dec = y.squeeze(2).transpose(1,2)
        x_dec_mark = y_mark.squeeze(1)
        x_dec = torch.cat((x_enc[:,-self.configs['label_len']:,:],x_dec),dim=1)
        x_dec_mark = torch.cat((x_enc_mark[:,-self.configs['label_len']:,:],x_dec_mark),dim=1)
        return x_enc,x_dec,x_enc_mark,x_dec_mark