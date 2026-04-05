import torch.nn as nn
from .model import Model

class fcn_api(nn.Module):
    def __init__(self, configs, graph_generator, fixed_adjs):
        super(fcn_api, self).__init__()
        model_configs = self.load_configs(configs)
        self.adjs = fixed_adjs
        if graph_generator is not None:
            self.with_GraphGen = True
            self.graph_generator = graph_generator
        else:
            self.with_GraphGen = False
        # spatial temporal predictor
        self.model = Model(model_configs)
 
    def load_configs(self, configs):
        model_configs = configs['model']
        model_configs['d_name'] = configs['dataset']['name']
        model_configs['c_date'] = configs['dataset']['c_date']
        model_configs['n_nodes'] = configs['dataset']['n_nodes']
        model_configs['c_in'] = configs['exp']['c_in']
        model_configs['c_out'] = configs['exp']['c_out']
        model_configs['device'] = configs['exp']['device']
        model_configs['inp_len'] = configs['exp']['inp_len']
        model_configs['pred_len'] = configs['exp']['pred_len']
        return model_configs
    
    def forward(self, seq_x, seq_x_mark, seq_y_mark, **args):
        predicts, loss,x_t,y_t = self.model(seq_x, seq_x_mark, seq_y_mark, adjs=None, scaler=args['scaler'], seq_y=args['seq_y'], choise_channels=args['choise_channels'], epoch=args['epoch'])
        return predicts, loss,x_t,y_t