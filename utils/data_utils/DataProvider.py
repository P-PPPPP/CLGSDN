import torch, os
from torch.utils.data import DataLoader
from .AdjProvider import get_adj
from .DataLoader import Dataset_with_Time_Stamp

class Data_Processor():
    def __init__(self, configs):
        self.configs = configs

        self.rawfile_path = configs['dataset']['file']
        self.dataloader_folder = configs['store']['dataloader'] + configs['dataset']['name'] + '/'
        self.dtype = configs['exp']['dtype']
        self.device = configs['exp']['device']
        self.batch_size = configs['exp']['batch_size']
        self.print_info = configs['info']['print_info']
        self.check_dataloader_bool = configs['exp']['check_dataloader']
    
    def get_data(self):
        self.find_gen_dataloader(self.configs,regen=self.configs['exp']['regen_dataset'])
        dataloader, dataset, scaler = self.load_files(self.configs)
        if self.check_dataloader_bool:
            self.test_dataloader(dataloader)
        adjs = get_adj(path=self.configs['dataset']['adj_file'], n_nodes=self.configs['dataset']['n_nodes'])
        return dataloader, dataset, scaler, adjs
    
    def test_dataloader(self, dataloader):
        # print("Testing dataloader.")
        return

    def find_gen_dataloader(self, configs, regen=False):
        dataloader_bool = True
        for flag in ['train','test','vali']:
            dataset_path = '{}{}.dataset'.format(self.dataloader_folder,flag)
            dataloader_bool = dataloader_bool and os.path.exists(dataset_path)
        
        if (not dataloader_bool) or regen:
            if self.print_info:
                if regen:
                    print('Regenerating dataset.')
                elif not dataloader_bool:
                    print('Target dataset \"{}\" Not finded, Generating.'.format(self.dataloader_folder))
                
            if not os.path.exists(self.dataloader_folder):
                os.makedirs(self.dataloader_folder)
        
            gen_torch_dataset(configs, self.print_info)
            if self.print_info:
                print('\tDone.')
            
    def load_files(self, configs):
        if self.print_info:
            print('Loading existence torch dataset.')
        dataset = {'train':None,'vali':None,'test':None}
        dataloader = {'train':None,'vali':None,'test':None}
        channels_info = [configs['dataset']['channel_info'][idx] for idx in configs['exp']['select_channels']]
        if self.print_info:
            print('\tchoice channels {}'.format(channels_info))
        for flag in configs['exp']['dataset_type']:
            # path
            file_path = '{}{}.dataset'.format(self.dataloader_folder, flag)
            data_set = torch.load(file_path, weights_only=False)
            data_set.choice(configs['exp']['select_channels'])
            # data_set.to(self.device, self.dtype)
            # parameter
            batch_size = self.batch_size
            drop_last = True
            shuffle = False if flag == 'test' else True
            # load
            if self.print_info:
                print('\t{:5}: {}'.format(flag, len(data_set)))
            dataset[flag] = data_set
            dataloader[flag] = DataLoader(data_set,
                                    batch_size = batch_size,
                                    shuffle = shuffle,
                                    drop_last = drop_last)
        # scaler
        scaler = dataset['train'].scaler# .to(self.device)
        return dataloader, dataset, scaler

def gen_torch_dataset(configs, print_info=True):
    name = configs['dataset']['name']
    file_path = configs['dataset']['file']
    if name in ['metr_la', 'pems_bay']:
        from .To_numpy.h5 import metrla_pemsbay
        data_array, date_array = metrla_pemsbay(file_path, print_info)
    elif name in ['pems04','pems08']:
        from .To_numpy.npz import npz_file_pems0408
        data_array, date_array = npz_file_pems0408(file_path, 
                                                   start_time=configs['dataset']['start_time'], 
                                                   time_windows=configs['dataset']['time_window'], 
                                                   lens=configs['dataset']['lens'], 
                                                   print_info=configs['info']['print_info'])
    elif name in ['taxibj13','taxibj14','taxibj15','taxibj16']:
        from .To_numpy.h5 import taxibj
        data_array, date_array = taxibj(file_path, print_info)
    elif name in ['ettm1','ettm2','etth1','etth2']:
        from .To_numpy.csv import ett
        data_array, date_array = ett(file_path, print_info)
    elif name in ['exchange_rate']:
        from .To_numpy.csv import exchange_rate
        data_array, date_array = exchange_rate(file_path, print_info)
    elif name in ['illness']:
        from .To_numpy.csv import illness
        data_array, date_array = illness(file_path, print_info)
    elif name in ['traffic']:
        from .To_numpy.csv import traffic
        data_array, date_array = traffic(file_path, print_info)
    elif name in ['weather']:
        from .To_numpy.csv import weather
        data_array, date_array = weather(file_path, print_info)
    elif name in ['electricity']:
        from .To_numpy.csv import electricity
        data_array, date_array = electricity(file_path, print_info)
    else:
        raise KeyError('dataset \"{}\" not defined.'.format(name))

    for flag in configs['exp']['dataset_type']:        
        dataset = Dataset_with_Time_Stamp(
            data_array, date_array, configs, flag=flag)
        dataset_folder = '{}{}/'.format(configs['store']['dataloader'],configs['dataset']['name'])
        file_path = '{}{}.dataset'.format(dataset_folder, flag)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        torch.save(dataset,'{}{}.dataset'.format(dataset_folder, flag))