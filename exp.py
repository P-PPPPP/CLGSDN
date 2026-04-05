import os
import torch
import argparse
import numpy as np
from engine import Engine
from utils.ExpTools import my_bool, setting_seed

from utils.ConfigLoader import Config_Loader
from utils.data_utils.DataProvider import Data_Processor
from utils.ModelProvider import get_model, get_graphgenerator

parser = argparse.ArgumentParser()
'''
   The following parameters can be missing. If missing, the corresponding parameters in each yaml file will be read automatically.
   If specified, the following parameters will directly replace the parameters in the yaml files.
'''
parser.add_argument('--root_store', type=str, default='./datasets/', help='Root directory for data reading/writing, default is current directory')
parser.add_argument('--engine_config', type=str, default='./configs/engine.yaml', help='')

'''Select model name, the unified interface for all models is in utils/ModelProvider.py'''
parser.add_argument('--model', type=str, default='gw',
                    help='Model name: currently available: autoformer, transformer, fogs, lightcts, dstagnn, megacrn, fdnet, gw, stgcn, agcrn, dcrnn, astgcn' + \
                    'fcn')
parser.add_argument('--GSL', type=str,  default='CLGSDN', help='Graph generator name: CLGSDN; None: not use')

parser.add_argument('--dataset', type=str, default='pems04', 
                    help='Dataset name: metr_la, pems_bay, pems04, pems08, taxibj13 (13-16)')

parser.add_argument('--select_channels', type=int, nargs='+', default=[2], 
                    help='Select data channels, separate multiple values with spaces, e.g.: --select_channels 0 1 2')

parser.add_argument('--n_prob', type=int, default=3, help='Number of graphs output by Graph Generator')
parser.add_argument('--period_type', type=str, default='None', help='')
parser.add_argument('--loss', type=str, default='l1', help='')
parser.add_argument('--inp_len', type=int, default=12, help='')
parser.add_argument('--pred_len', type=int, default=12, help='')
# Experiment parameters

parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='')
parser.add_argument('--device', type=str, default='cuda:7', help='cpu, cuda. cuda:0-n')
parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--regen_dataset', type=my_bool, default=True, help='Regenerate Dataloader each time it runs')
args = parser.parse_args()

print("select_channels: ",args.select_channels)

def envs_setup(seed=42):
    # Setup environment variables
    # Setting GPU Index.
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU to generate random numbers for deterministic results
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Set random seed for GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True                 # Enable cuDNN to accelerate operations like convolution
    torch.backends.cudnn.benchmark = True              # When input shape changes each time, test all available convolution algorithms and select the fastest one
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility of computations

# main function
def main(configs):
    # Load data
    dataloader, dataset, scaler, raw_adj = Data_Processor(configs).get_data()
    # Load graph generator
    GSL_module = get_graphgenerator(configs, raw_adj)
    # Get model
    model = get_model(configs, GSL_module, raw_adj)

    # Initialize experiment
    exp = Engine(configs)

    # Load model and data into the experiment framework
    exp.load(model, dataloader, dataset, scaler, loss=configs['exp']['loss'], optimizer=configs['exp']['optimizer'])
    # Disable multi-process mode
    configs['muti_process'] = False
    # Start experiment
    exp.Run()

if __name__ == '__main__':
    # Load and check configs info
    configs = Config_Loader(args).get_configs()
    setting_seed(configs['exp']['seed'])
    main(configs)