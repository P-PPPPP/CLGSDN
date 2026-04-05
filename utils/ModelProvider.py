from torch import cat

from model.Predictor.STF.AGCRN.model_api import agcrn_api
from model.Predictor.STF.ASTGCN.model_api import astgcn_api
from model.Predictor.STF.DCRNN.model_api import dcrnn_api
from model.Predictor.STF.GWNet.model_api import gw_api
from model.Predictor.STF.STGCN.model_api import stgcn_api
from model.Predictor.STF.DSTAGNN.model_api import dstagnn_api
from model.Predictor.STF.DDGCRN.model_api import ddgcrn_api

from model.Predictor.STF.LIGHTCTS.model_api import lightcts_api
from model.Predictor.STF.MEGACRN.model_api import megacrn_api

from model.Predictor.MTSF.AUTOFORMER.model_api import autoformer_api
from model.Predictor.MTSF.FOGS.model_api import fogs_api
from model.Predictor.MTSF.TimesNet.model_api import timesnet_api
from model.Predictor.MTSF.INFORMER.model_api import informer_api
from model.Predictor.MTSF.TRANSFORMER.model_api import transformer_api
from model.Predictor.MTSF.DLINEAR.model_api import dlinear_api
from model.Predictor.MTSF.LSTM.model_api import lstm_api
from model.Predictor.MTSF.LIGHTTS.model_api import lightts_api
from model.Predictor.MTSF.FCN.model_api import fcn_api

from model.CLGSDN.CLGSDN_GraphGenerator import Graph_Generator

def get_model(configs, graph_generator=None, raw_adjs=None):
    models_dict = {
        'ddgcrn':ddgcrn_api,
        'gw':gw_api,
        'dcrnn':dcrnn_api,
        'stgcn':stgcn_api,
        'astgcn':astgcn_api,
        'agcrn':agcrn_api,
        'autoformer': autoformer_api,
        'megacrn': megacrn_api,
        'dstagnn':dstagnn_api,
        'lightcts':lightcts_api,
        'fogs': fogs_api,
        'timesnet': timesnet_api,
        'informer': informer_api,
        'transformer': transformer_api,
        'dlinear': dlinear_api,
        'lstm': lstm_api,
        'lightts': lightts_api,
        'fcn':fcn_api
    }
    # abnorm
    model_name_list = list(models_dict.keys())
    model_name = configs['model']['name'].lower()
    if model_name not in model_name_list:
        raise NameError('Model Not Find.')
    # load model
    model = models_dict[model_name](configs, graph_generator, raw_adjs)
    return model

def get_graphgenerator(configs, adjs):
    graphgenerator_configs = configs['graphgenerator']
    if graphgenerator_configs is None:
        return None
    graphgenerator_configs['seq_len'] = configs['exp']['inp_len']
    graphgenerator_configs['device'] = configs['exp']['device']
    graphgenerator_configs['dropout'] = configs['exp']['dropout']
    graphgenerator_configs['batch_size'] = configs['exp']['batch_size']
    graphgenerator_configs['n_nodes'] = configs['dataset']['n_nodes']
    graphgenerator_configs['dim_date'] = configs['dataset']['c_date']
    graphgenerator_configs['data_channels'] = len(configs['exp']['select_channels'])

    num_adjs = graphgenerator_configs['n_GMB']
    graph_generator = Graph_Generator(graphgenerator_configs, adjs.unsqueeze(0).repeat(num_adjs,1,1))
    return graph_generator