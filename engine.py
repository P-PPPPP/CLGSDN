import torch
import torch.optim as optim
import utils.loss_box as loss_box
from utils.ExpTools import exp_summary, adjust_learning_rate, runing_confirm_file
from utils.LoadSaveTools import load_best_model, save_model, save_metrics
from utils.Logger import MyLogger, MetricsInfo, ExpInfo
from utils.ModelSummary import summary
import numpy as np

class Basic_Engine():
    def __init__(self, configs):
        self.configs = configs
        self.set_param(configs)
        
    def set_param(self, configs):
        self.loss_rate = 1.0
        
        self.print_info = configs['info']['print_info']
        self.iter_report = configs['info']['iter_report']
        self.print_every = configs['info']['print_every']
        self.model_summary = configs['info']['model_summary']

        self.dtype = configs['exp']['dtype']
        self.device = configs['exp']['device']
        self.epochs = configs['exp']['epochs']
        self.lr = self.configs['exp']['learning_rate']
        self.adj_lr_counter = 1
        self.adjust_lr = configs['exp']['adjust_lr']
        self.batch_size = configs['exp']['batch_size']
        self.muti_process = configs['exp']['muti_process']
        self.weight_decay = self.configs['exp']['weight_decay']
        self.load_best_model = configs['exp']['load_best_model']
        self.save_best_model = configs['exp']['save_best_model']
        self.choise_channels = torch.tensor(configs['exp']['select_channels'],dtype=int)
        
    def load(self, model, dataloader, dataset=None, scaler=None, **args):
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.dataloader = dataloader
        self.dataset = dataset
        # 数据设备转换
        for key in self.dataloader.keys():
            self.dataloader[key].dataset.to(self.device, self.dtype)
        self.scaler = scaler
        self.scaler.to(self.device)
        self.loss_type = args['loss']
        if 'loss' in args.keys():
            self.set_loss(args['loss'])
        if 'optimizer' in args.keys():
            self.set_opt(args['optimizer'])
        self.set_logger()
        
        if not self.configs['graphgenerator'] is None:
            if self.configs['graphgenerator']['name'] == 'CLGSDN':
                self.loss_rate = self.configs['graphgenerator']['loss_rate']
        
    def set_loss(self, info):
        if info == 'mask_mae':
            self.loss = loss_box.masked_mae
        elif info == 'mae':
            self.loss = loss_box.mae
        elif info == 'mse':
            self.loss = loss_box.mse
        elif info == 'l1':
            self.loss = torch.nn.L1Loss()
        
    def set_opt(self, info):
        if info == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def set_logger(self):
        if self.model_summary:
            for data,_,data_mark1,data_mark2 in self.dataloader['train']:
                break
            input_size1 = list(data.size()[1:])
            input_size2 = list(data_mark1.size()[1:])
            input_size3 = list(data_mark2.size()[1:])
            batch_size = self.batch_size
            try:
                model_info = summary(self.model,input_size=[input_size1,input_size2,input_size3],batch_size=batch_size, device=self.configs['exp']['device'], dtype=self.configs['exp']['dtype'])
            except Exception as e:
                model_info = 'Exception: ' + str(e)
        else:
            model_info = None
        self.logger = MyLogger(self.configs, model_info, print_while_wirte=self.print_info)
        self.metrics_saver = MetricsInfo()
        self.metrics_logger = ExpInfo()

class Engine(Basic_Engine):
    def __init__(self, configs):
        super(Engine, self).__init__(configs)
        self.running_flag = False
        
    def Run(self):
        self.metrics_logger.log_setting(self.configs) # save hyperparameters settings
        # run experiments
        for epoch in range(1, self.epochs+1):
            self.logger.start_epoch(epoch,'exp')
            for flag in ['train','vali','test']:
                if flag == 'test' and epoch == 10:
                    print('--- Test ---')
                metrics = self.run_epoch(epoch, flag, iter_report=self.iter_report)
                self.metrics_logger.update(epoch, metrics, flag=flag)
            # adjust learning rate
            lr_log, self.adj_lr_counter, withdrew_bool = adjust_learning_rate(self.optimizer, epoch, 
                                                                              self.adjust_lr, self.lr, self.adj_lr_counter, 
                                                                              self.weight_decay, self.metrics_logger.get_metrics())
            # save model
            load_log = None
            # load best model every epochs
            if self.save_best_model:
                save_model(self.configs, self.model, epoch, self.metrics_logger, flag='epoch')

            if self.load_best_model:# or withdrew_bool:
                # save this model_state
                model_dict, load_log = load_best_model(self.configs,exp_info=self.metrics_logger,current_epoch=epoch)
                # load model_state at best vali loss
                if model_dict is not None:
                    self.model.load_state_dict(model_dict)
            self.logger.end_epoch(self.metrics_logger.get_metrics(), epoch, lr_log, load_log) # update logger
        # '''save metrics information'''
        if self.save_best_model:
            save_model(self.configs, self.model, epoch, self.metrics_logger, flag='best')
        save_metrics(self.configs, self.metrics_logger)
        self.logger.end_exp(self.metrics_logger.get_metrics())

    def run_epoch(self, epoch, flag='train', iter_report=True):
        self.logger.start_epoch(epoch, flag, iter_report)
        self.metrics_saver.init_metrics()
        for iters, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.dataloader[flag]): 
            loss, metrics_dict = self.run_iter(seq_x, seq_y, seq_x_mark, seq_y_mark, epoch, iters, flag) 
            # save metrics
            self.metrics_saver.update(loss,metrics_dict)
            # report
            if iter_report and (iters % self.print_every == 0):
                time, start_time = self.metrics_saver.set_time()
                self.logger.report_iter(iters, self.metrics_saver.get_metrics(), time, start_time, self.print_every)
            if self.muti_process and not self.running_flag:
                runing_confirm_file(self.configs,flag='create')
                self.running_flag = True
        # avgerage metrics
        self.metrics_saver.update_avg()
        return self.metrics_saver.get_metrics()
    
    def run_iter(self, seq_x, seq_y, seq_x_mark, seq_y_mark, epoch, iters, flag):
        if flag == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        # seq_x, seq_y, seq_x_mark, seq_y_mark = seq_x.to(self.device), seq_y.to(self.device), seq_x_mark.to(self.device), seq_y_mark.to(self.device)
        
        self.optimizer.zero_grad()
        # model output
        output, addi_loss = self.model(seq_x, seq_x_mark, seq_y_mark, seq_y=seq_y, 
                                       scaler=self.scaler, epoch=epoch, choise_channels=self.choise_channels, iters=iters)
        if self.configs['exp']['inv_trans']:
            predict = self.scaler.inv_trans(output, self.choise_channels)
            real = self.scaler.inv_trans(seq_y, self.choise_channels)
        else:
            predict = output
            real = seq_y
        # predict = predict.to('cpu')
        # real = real.to('cpu')
        if self.loss_type == 'mask_mae':
            model_loss = self.loss(predict, real, 0.0)
            metrics_dict = loss_box.metric(predict, real, 0.0)
        else:
            model_loss = self.loss(predict, real)
            metrics_dict = loss_box.metric(predict, real)
        loss = loss_box.unite_loss(model_loss, addi_loss, self.loss_rate)
        if flag == 'train':
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), metrics_dict