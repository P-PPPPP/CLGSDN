import os, time
import numpy as np

class ExpInfo():
    def __init__(self):
        self.metrics_info = {
            'train':{},
            'vali':{},
            'test':{},
            'hyperparameters':{},
            'notes':''
            }
        self.using_time = {
            'train':[],
            'vali':[],
            'test':[]}
        
    def update(self,epoch,metrics,using_time = None,flag = None):
        # append using time
        if not using_time  == None:
            self.using_time[flag].append(using_time)
        # append metrics
        for idx in metrics.keys():
            # check keys of dict
            if not (idx in self.metrics_info[flag].keys()):
                self.metrics_info[flag].update({idx:[]})
            # append new item
            self.metrics_info[flag][idx].append(metrics[idx])

    def get_best(self):
        self.best_metrics  = {
            'train':{},
            'vali':{},
            'test':{}}
        for state in list(self.best_metrics.keys()):
            for idx in list(self.metrics_info[state].keys()):
                if idx[0] =='m':
                    self.best_metrics[state].update({('best'+idx[1:]):np.min(self.metrics_info[state][idx])})
        return self.best_metrics
    
    def get_metrics(self):
        return self.metrics_info
    
    def log_setting(self, configs):
        self.metrics_info['notes'] = configs['info']['notes']
        args_dict = {
            'model':configs['model']['name'],
            'dataset':configs['dataset']['name']
            }
        self.update(0, args_dict, flag='hyperparameters')

class MetricsInfo():
    def __init__(self):
        self.init_metrics()
    
    def init_metrics(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.metrics = {'loss':[], 
                        'mae':[], 'mape':[], 'mse':[], 'rmse':[],
                        'mae_all':[], 'mape_all':[], 'mse_all':[],'rmse_all':[],
                        'm_loss':np.inf,
                        'm_mae':np.inf, 'm_mape':np.inf, 'm_mse':np.inf, 'm_rmse':np.inf, 'm_rmse_2':np.inf,
                        'm_mae_all':np.inf, 'm_mape_all':np.inf, 'm_mse_all':np.inf, 'm_rmse_all':np.inf, 'm_rmse_all_2':np.inf,
                        'using_time':None}

    def update(self,loss,metrics_dict):
        self.metrics['loss'].append(loss)
        self.metrics['mae'].append(metrics_dict['mae'])
        self.metrics['mape'].append(metrics_dict['mape'])
        self.metrics['mse'].append(metrics_dict['mse'])
        self.metrics['rmse'].append(metrics_dict['rmse'])
        self.metrics['mae_all'].append(metrics_dict['mae_all'])
        self.metrics['mape_all'].append(metrics_dict['mape_all'])
        self.metrics['mse_all'].append(metrics_dict['mse_all'])
        self.metrics['rmse_all'].append(metrics_dict['rmse_all'])

    def update_avg(self):
        self.metrics['m_loss'] = np.mean(self.metrics['loss'])
        self.metrics['m_mae'] = np.mean(self.metrics['mae'])
        self.metrics['m_mape'] = np.mean(self.metrics['mape'])
        self.metrics['m_mse'] = np.mean(self.metrics['mse'])
        self.metrics['m_rmse'] = np.mean(self.metrics['rmse'])
        self.metrics['m_rmse_2'] = np.sqrt(np.mean(self.metrics['mse']))

        self.metrics['m_mae_all'] = np.mean(self.metrics['mae_all'],axis=0)
        self.metrics['m_mape_all'] = np.mean(self.metrics['mape_all'],axis=0)
        self.metrics['m_mse_all'] = np.mean(self.metrics['mse_all'],axis=0)
        self.metrics['m_rmse_all'] = np.mean(self.metrics['rmse_all'],axis=0)
        self.metrics['m_rmse_all_2'] = np.sqrt(np.mean(self.metrics['mse_all'],axis=0))

        self.metrics['using_time'] = time.time() - self.start_time

    def set_time(self):
        spend_time = time.time() - self.last_time
        self.last_time = time.time()
        return spend_time,self.start_time
    
    def get_metrics(self):
        return self.metrics
    
class MyLogger():
    def __init__(self, configs, model_info,
                 print_while_wirte = True,
                 output_with_time = True,
                 max_len_every_line = 100):
        
        self.set_param(configs, model_info, print_while_wirte, output_with_time, max_len_every_line)
        self.creat_log_file()
        self.report_title()
        self.report_configs_info()
        self.report_model_structure()
    
    # loger function
    def set_param(self, configs, model_info, print_while_wirte, output_with_time, max_len_every_line):
        self.path = configs['store']['results']
        self.save_log = configs['info']['save_log']
        self.task_name = configs['info']['task_name']
        self.name = configs['info']['exp_start_time']
        self.notes = configs['info']['notes']
        self.print_info = configs['info']['print_info']

        def get_configs_list(configs):
            configs_list = list(configs.items())
            new_list = []
            max_len = 20
            for i,j in configs_list:
                if type(j) in [list,tuple,str]:
                    if len(j) > max_len:
                        j = str(j)[:max_len] + '...'
                        new_item = (i,j)
                    else:
                        new_item = (i,j)
                else:
                    new_item = (i,j)
                new_list.append(new_item)
            return new_list
        
        self.model_configs_list = get_configs_list(configs['model'])
        self.envs_configs_list = get_configs_list(configs['exp'])
        self.dataset_configs_list = get_configs_list(configs['dataset'])
        if not configs['graphgenerator'] is None:
            self.graphgenerator_configs_list = list(configs['graphgenerator'].items())
        else:
            self.graphgenerator_configs_list = None
        self.model_info = model_info
        self.print_while_wirte = print_while_wirte
        self.output_with_time = output_with_time
        self.max_len_every_line = max_len_every_line
        self.line = '-'*(self.max_len_every_line+21) if (output_with_time) else '-'*self.max_len_every_line

    def creat_log_file(self):
        if not self.save_log: # 不写log时跳过
            return
        folder_path = '{}logs/'.format(self.path)
        # create dirs
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # create file
        task_name = '' if self.task_name is None else '_' + self.task_name
        file_name = '{}{}.log'.format(self.name, task_name)
        self.file_path = folder_path + file_name
        note = open(self.file_path,'w')
        note.close()

    def report_title(self):
        self.write_line('long')
        log = '\nExperiment Start at <{}>\n\tPID: <{}>\n\tNotes: <{}>\n'.format(self.name,os.getpid(),self.notes)
        self.output_log(log)
        self.write_line('long')

    def report_configs_info(self):
        envs_info = self.get_str_exp_info(self.envs_configs_list, 'Envs')
        model_info = self.get_str_exp_info(self.model_configs_list, 'Model')
        dataset_info = self.get_str_exp_info(self.dataset_configs_list, 'Dataset')
        if not self.graphgenerator_configs_list is None:
            graphgenerator_info = self.get_str_exp_info(self.graphgenerator_configs_list, 'GraphGenerator')
        else:
            graphgenerator_info = ['']
        exp_info = envs_info + model_info + dataset_info + graphgenerator_info
        if self.print_info:
            for item in exp_info:
                print(item,end = '')
        self.output_log(exp_info,print_while_wirte = False)
        self.write_line('long')

    def get_str_exp_info(self, args_list, flag):
        len_line = 0
        string_list = []
        string_list.append('{} Setting:\n'.format(flag))
        # string_list.append('{}\n'.format(self.line))
        string = ''
        interval_string = ': \"'
        end_string = '\";  '
        for item in args_list:
            flag = 0
            # exp info
            key = item[0]
            value = item[1]
            str_value = str(value)
            # 将写入的log
            item_log = ' '*4 + key + interval_string + str_value + end_string
            len_line += len(item_log)
            # 判断是否超出最大长度
            if len_line <= self.max_len_every_line:
                string += item_log
            else:
                # w
                string += '\n'
                string_list.append(string)
                # 重置
                string = '' + item_log
                len_line = len(item_log)
                flag = 1 
        if (not len(string) == 0) and (flag == 0):
            string_list.append(string+'\n')
        return string_list
    
    def report_model_structure(self):
        self.output_log('Model structure:',print_while_wirte = False)
        self.output_log(str(self.model_info),print_while_wirte = False,start = '\t')
        self.write_line('long')

    def output_log(self, log, print_while_wirte=None, start = '',end = '\n'):
        if print_while_wirte  == None: # 没有指定时,使用self的
            print_while_wirte = self.print_while_wirte
        if print_while_wirte: # 打印信息
            print(log,end=end)
        if not self.save_log: # 不写log时跳过
            return
        file = open(self.file_path,'a')
        if type(log)  == str:# 处理为list
            log = [log]
        if type(log)  == list:
            for item in log:
                tmp_str = item.split('\n')
                for text in tmp_str[:-1]:
                    self.self_write(file,text+'\n',start,end)
                if not tmp_str[-1]  == '':
                    self.self_write(file,tmp_str[-1],start,end)
        file.close()
        
    def self_write(self,file,text, start='', end=''):
        if text == '\n':
            return
        if not start  == None:
            text = start + text
        if text[-1]  == '\n' and end  == '\n':
            pass
        else:
            text = text + end
        if self.output_with_time:
            now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            if not self.line in text:
                if not text  == '\n':
                    text = '{}| {}'.format(now_time,text) 
                elif not text == '':
                    text = ' '*19 + '|\n'
                else:
                    return
            file.write(text)
        else:
            file.write(text)
    # logger api
    def start_epoch(self,epoch,flag,report=True):
        if flag == 'exp':
            log = '\nEpoch: {:03d}'.format(epoch)
            self.output_log(log)
        elif report:
            log = 'State: <{0:s}>\n\t{1:8}{2:<12}{3:<12}{4:<12}{5:<12}{6:<16}{7:<12}'.format(flag,'iter','Loss','MAE','MSE','MAPE','Speed (/iter)','Time Cost')
            self.output_log(log,start = '\t')
    
    def report_iter(self, iters, metrics, spend_time, start_time, print_every):
        num_iter = 1 if iters==0 else print_every
        before = 0 if (iters==0) else (iters-print_every)
        later = 1 if (iters==0) else iters
        log = '\t{0:04d}{1:4s}{2:<12.4f}{3:<12.4f}{4:<12.4f}{5:<12.2%}{6:<16s}{7:<12s}'
        log = log.format(iters,' ',
                np.mean(metrics['loss'][before:later]),
                np.mean(metrics['mae'][before:later]),
                np.mean(metrics['mse'][before:later]),
                np.mean(metrics['mape'][before:later]),
                '{:.4f} s'.format(spend_time/num_iter),
                '{:.2f} s'.format(time.time()-start_time))
        self.write_iterinfo(log)
    
    def end_epoch(self, metrics, epoch, lr_log, load_log):
        log = 'Info Report: <Epoch {:03d}> '.format(epoch)
        summary_log = {'title':log, 'train':None, 'vali':None, 'test':None, 'last5':None, 'last5title':None}
        for flag in ['train','vali','test']:
            log = '\t{0:5s}{1:3s}{2:<12.4f}{3:<12.4f}{4:<12.4f}{5:<12.2%}{6:<12s}'
            log = log.format(flag,'',
                metrics[flag]['m_loss'][epoch-1],          
                metrics[flag]['m_mae'][epoch-1],
                metrics[flag]['m_mse'][epoch-1],
                metrics[flag]['m_mape'][epoch-1],
                '{:.2f} s'.format(metrics[flag]['using_time'][epoch-1]))
            summary_log[flag] = log
        # latest 5 loss
        latest_test_loss = metrics['test']['m_mae'][-5:]

        title_last5 = '<Latest 5>\t'
        metric_laset5 = '\t{}  \t'.format('MAE')

        for idx in range(len(latest_test_loss)):
            title_last5 +=  '{:<12d}'.format(1+idx+epoch-len(latest_test_loss))
            metric_laset5 += '{:<12.4f}'.format(latest_test_loss[idx])
        
        test_mae = np.array(metrics['test']['mae_all'])
        test_mae = test_mae.mean(axis=(0,1))
        n_steps = test_mae.size

        step = [3/12*n_steps,6/12*n_steps,n_steps]
        step = [int(i) for i in step]

        title_step = '<Step /{}> \t'.format(n_steps)
        metric_step = '\t{}  \t'.format('MAE')
        
        for idx in step:
            title_step += '{:<12d}'.format(idx)
            metric_step += '{:<12.4f}'.format(test_mae[idx-1])

        summary_log['last5title'] = title_last5
        summary_log['last5'] = metric_laset5

        summary_log['title_step'] = title_step
        summary_log['step'] = metric_step

        
        if not summary_log == '':
            # title
            log = summary_log['title']
            self.output_log(log)
            log = '\t{0:8}{1:12}{2:12}{3:12}{4:12}{5:12}'.format('State','Avg. Loss','Avg. MAE','Avg. MSE','Avg. MAPE','Time Cost')
            self.output_log(log)
            # body 1
            for flag in ['train','vali','test']:
                self.output_log(summary_log[flag], start='\t')
            # body 2
            
            self.output_log(summary_log['title_step'], start='\t')
            self.output_log(summary_log['step'], start='\t')

            self.output_log(summary_log['last5title'], start='\t')
            self.output_log(summary_log['last5'], start='\t')

        if not lr_log == None:
            self.output_log(lr_log, start='\t')
        if not load_log == None:
            self.output_log(load_log, start='\t')
        self.write_line(line_type='long')

    def end_exp(self, metrics):
        title = ['Notes', 
             'Idx','Avg. MAE','Avg. MSE','Avg. MAPE','Avg. RMSE']
        # Find the index with minimum validation "Loss".
        idx = np.argmin(metrics['vali']['m_mae'])
        test_mae = metrics['test']['m_mae'][idx]
        test_mse = metrics['test']['m_mse'][idx]
        test_mape = metrics['test']['m_mape'][idx]
        test_rmse = metrics['test']['m_rmse_2'][idx]
        # test_mse = np.mean(metrics['test']['mse'][min_vali_idx])
        logs = '\n<Final Report>'
        logs += '\n\t{:8s}{:12s}{:12s}{:12s}{:12s}\n'.format(title[1],title[2],title[3],title[4],title[5])
        logs += '\t{:<8d}{:<12.5f}{:<12.5f}{:<12.3%}{:<12.5f}\n'.format(idx+1,test_mae,test_mse,test_mape,test_rmse)
        logs += '\t{}: {}\n'.format(title[0] ,str(metrics['notes']))
        logs += "\tTraining finished!\n"
        self.output_log(logs)
        self.write_line('long')

    def write_iterinfo(self,log):
        self.output_log(log,start='\t')

    def write_line(self,line_type='short',end='\n'):
        if not self.save_log:
            return
        note = open(self.file_path,'a')
        if line_type  == 'short':
            note.write(' '*19+'|'+self.line[20:]+end)
        elif line_type  == 'long':
            note.write(self.line+end)
        note.close()

    def write_blank(self,blank_type = 'short',end = '\n'):
        if not self.save_log:
            return
        note = open(self.file_path,'a')
        if blank_type  == 'short':
            note.write(' '*19+'|'+end)
        elif blank_type  == 'long':
            note.write(' '+end)
        note.close()