import os, torch, time, copy, yaml

class Config_Loader():
    def __init__(self, args):
        self.init_params(args)
        configs = self.read_file(args)
        configs = self.map_dict_item(configs, args)
        
        self.print_info = configs['info']['print_info']

        new_configs = {}
        new_configs['dataset'] = self.reset_dataset(configs['dataset'], args)
        new_configs['info'] = self.reset_info(configs['info'])
        new_configs['exp'] = self.reset_exp(configs['exp'])
        new_configs['store'] = self.reset_path(configs['store'])
        new_configs['graphgenerator'] = self.reset_graphgenerator(configs['graphgenerator'])
        new_configs['model'] = self.reset_model(configs['model'])
        new_configs = self.reset_overall(new_configs)
        self.configs = copy.deepcopy(new_configs)
    
    def init_params(self, args):
        self.args_configs = vars(args)
        self.args_keys = self.args_configs.keys()

    def read_file(self,args):
        configs = {}
        # Engine
        engine_config_path = '{}'.format(args.engine_config)
        with open(engine_config_path) as f:
            configs.update(yaml.safe_load(f))
        
        # Dataset
        dataset_configs_path = '{}{}.yaml'.format(configs['store']['dataset_config'], args.dataset)
        with open(dataset_configs_path) as f:
            configs['dataset'] = yaml.safe_load(f)

        # Model
        model_configs_path = '{}{}.yaml'.format(configs['store']['model_config'], args.model.lower())
        with open(model_configs_path) as f:
            configs['model'] = yaml.safe_load(f)

        # Graph Generator
        if args.GSL != 'None':
            graphgen_configs_path = '{}{}.yaml'.format(configs['store']['GSL_config'], args.GSL)
            with open(graphgen_configs_path) as f:
                configs['graphgenerator'] = yaml.safe_load(f)
        else:
            configs['graphgenerator'] = None
        return configs


    def map_dict_item(self, configs, args):
        mapping_dict = {'': None,
                    'None':None,
                    'torch.float':torch.float,
                    'False':False,
                    'F':False,
                    'f':False,
                    'True':True,
                    'T':True,
                    't':True
                    }
        
        def mapping_dict_items(configs, mapping_dict, args_dict):
            mapping_from_configs = list(mapping_dict.keys())
            mapping_from_args = list(vars(args).keys())
            
            configs = copy.deepcopy(configs)
            new_configs = {}
            dict_keys = configs.keys()

            for key in dict_keys:
                value = configs[key]
                # if the type is dict, continue recursion
                if type(value) == dict:
                    new_configs[key] = mapping_dict_items(value, mapping_dict, args_dict)
                    continue
                # 
                if key in mapping_from_args:
                    mapping_to = args_dict[key]
                    if mapping_to in mapping_from_configs:
                        mapping_to = mapping_dict[value]
                    new_configs[key] = mapping_to
                    continue
                #
                if value in mapping_from_configs:
                    mapping_to = mapping_dict[value]
                    new_configs[key] = mapping_to
                    continue
                #
                new_configs[key] = value

                if type(new_configs[key]) == str:
                    # for scientific notation 'xey': x*10^(y) -> float
                    if 'e' in new_configs[key]:
                        try:
                            new_configs[key] = float(new_configs[key])
                        except:
                            pass
            return copy.deepcopy(new_configs)
        
        configs = mapping_dict_items(configs, mapping_dict, vars(args))
        return configs
        
    def get_configs(self):
        return self.configs

    def reset_info(self, configs):
        # experiments information
        copy.deepcopy(configs)
        configs['exp_ID'] = os.getpid()
        configs['exp_start_time'] = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        return copy.deepcopy(configs)
        
    def reset_exp(self, configs):
        configs = copy.deepcopy(configs)
        # check cuda
        if  configs['load_best_model']:
            configs['save_best_model'] = True
        if not torch.cuda.is_available():
            if self.print_info:
                print('Cuda not available. device -> cpu')
            configs['device'] = torch.device('cpu')
        else:
            configs['device'] = torch.device(configs['device'])
        return copy.deepcopy(configs)

    def reset_dataset(self, configs, args):
        copy.deepcopy(configs)
        if configs['channel_info'] is None:
            configs['channel_info'] = list(range(configs['n_channels']))
        if configs['nodes_info'] is None:
            configs['nodes_info'] = list(range(configs['n_nodes']))
        return copy.deepcopy(configs)

    def reset_path(self, configs):
        copy.deepcopy(configs)
        if configs['root_store'] == '':
            configs['root_store'] = './'
        configs['raw_dataset'] = configs['root_store'] + configs['raw_dataset']
        configs['dataloader'] = configs['root_store'] + configs['dataloader']
        configs['results'] = configs['root_store'] + configs['results']
        return copy.deepcopy(configs)
    
    def reset_graphgenerator(self, configs):
        return copy.deepcopy(configs)

    def reset_model(self, configs):
        return copy.deepcopy(configs)

    def reset_overall(self, configs):
        copy.deepcopy(configs)
        # Time embedding
        if configs['dataset']['name'] == 'weather':
            configs['model']['time_emb'] = False
        # dataset path
        configs['dataset']['folder'] = '{}{}'.format(
                                            configs['store']['raw_dataset'], 
                                            configs['dataset']['folder'])
            
        configs['dataset']['file'] = '{}{}'.format(
                                            configs['dataset']['folder'],
                                            configs['dataset']['file'])
        
        if configs['dataset']['adj_file'] is not None:
            configs['dataset']['adj_file'] = '{}{}'.format(
                                            configs['dataset']['folder'], 
                                            configs['dataset']['adj_file'])
        # check channels
        all_channels = list(range(len(configs['dataset']['channel_info'])))
        if configs['exp']['select_channels'] == [-1]:
            configs['exp']['select_channels'] = all_channels

        for item in configs['exp']['select_channels']:
            if item not in all_channels:
                raise ValueError('Params \"choise_channels\" shoule not \"{}\".'.format(configs['exp']['select_channels']))

        configs['exp']['c_in'] = len(configs['exp']['select_channels'])
        if configs['exp']['c_out'] == -1:
            configs['exp']['c_out'] = configs['exp']['c_in']
        return copy.deepcopy(configs)
