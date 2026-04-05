import os,torch,copy,pynvml
import numpy as np

def get_device_info(device):
    idx = device.index
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    info_list = [meminfo.total,meminfo.used,meminfo.free]
    totmem,usedmem,freemem = [i/1024**2 for i in info_list]
    return totmem,usedmem,freemem


def runing_confirm_file(configs, flag='check', state='running'):
    if configs['path']['root_path']  == '':
        # 绝对路径
        abs_path = os.getcwd()
        folder_path = '{}/{}{}'.format(abs_path,configs['path']['root_path'],configs['path']['saving_path'])
    else:
        folder_path = configs['path']['saving_path'] + 'running_confirm/'
    if 'task_name' in configs.keys():
        task_name = configs['task_name']
        if state == 'fail':
            task_name += '_fail'
        file_name = '{}_{}.txt'.format(configs['info']['exp_start_time'],task_name)
    else:
        file_name = '{}.txt'.format(configs['info']['exp_start_time'])
    # 创建目录
    if flag == 'create':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 创建文件
        file_path = folder_path + file_name
        note = open(file_path,'w')
        note.close()
    if flag == 'del':
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 删除文件
        file_path = folder_path + file_name
        os.remove(file_path)
    if flag == 'check':
        return os.path.exists(folder_path+file_name)
    
def check_dict_item(configs):
    map_dict = {
        'None':None,
        'False':False,
        'True':True,
        'torch.float':torch.float,
        'F':False,
        'f':False,
        'T':True,
        't':True
    }
    map_keys = list(map_dict.keys())
    ori_keys = list(configs.keys())
    for key in ori_keys:
        if configs[key] in map_keys:
            configs[key] = map_dict[configs[key]]
    return configs

def my_bool(s):
    if s == 'T' or s == 't':
        return True
    elif s == 'F' or s == 'f':
        return False
    else:
        return s !=  'False'

def setting_seed(seed = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = False

def adjust_learning_rate(optimizer, epoch, adj_lr, base_lr, adj_counter, base_wd, metrics_dict):
    if not adj_lr:
        return None,adj_counter,False
    log = None
    metrics = metrics_dict['test']['m_mae'][-3:]
    withdrew_bool = False
    new_adj_counter = adj_counter
    if not epoch < 2:
        if np.max(metrics[-3:-1]) <= metrics[-1]:
            withdrew_bool = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr/(1+adj_counter)
                param_group['weight_decay'] = base_wd*(1+adj_counter)
                new_adj_counter = adj_counter + 1
            log = '\n\tUpdate learning rate to <{}>\n\tUpdate weight decay to <{}>'.format(param_group['lr'],param_group['weight_decay'])
    return log,new_adj_counter,withdrew_bool

def exp_summary(epoch, metrics):
    log = 'Info Report: <Epoch {:03d}> '.format(epoch)
    summary_log = {'title':log, 'train':None, 'vali':None, 'test':None, 'last5':None, 'last5title':None}
    for flag in ['train','vali','test']:
        log = '\t{0:5s}{1:3s}{2:<12.4f}{3:<12.4f}{4:<12.4%}{5:<12.4f}{6:<12s}'
        log = log.format(flag,'',
            metrics[flag]['m_loss'][epoch-1],          
            metrics[flag]['m_mae'][epoch-1],
            metrics[flag]['m_mape'][epoch-1],
            metrics[flag]['m_rmse'][epoch-1],
            '{:.2f} s'.format(metrics[flag]['using_time'][epoch-1]))
        summary_log[flag] = log
    # latest 5 loss
    latest_test_loss = copy.deepcopy(metrics['test']['m_mae'])[-5:]
    return latest_test_loss