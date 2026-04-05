import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import os

def masked_mse(preds, labels, null_val = np.nan, dim='all'):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if dim == 'all':
        return torch.mean(loss,dim=(1, 2, 3))
    elif dim == 'time_ept':
        return torch.mean(loss,dim=(1, 2))

def masked_rmse(preds, labels, null_val=np.nan, dim='all'):
    mse = torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, dim=dim))
    return mse.mean(0)

def masked_mae(preds, labels, null_val=np.nan, dim='all'):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # IPython.embed()
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if dim == 'all':
        mae = torch.mean(loss)
    elif dim == 'time_ept':
        mae = torch.mean(loss, dim=(0, 1, 2))
    return mae

def masked_mape(preds, labels, null_val = np.nan, min_threshold=1e-4, dim='all'):
    min_mask = (torch.abs(labels)<min_threshold)
    labels = torch.where(min_mask, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if dim == 'all':
        mape = torch.mean(loss)
    elif dim == 'time_ept':
        mape = torch.mean(loss, dim=(0, 1, 2))
    return mape

def metric(pred, real):
    mae = masked_mae(pred, real, null_val=0.0).item()
    mape = masked_mape(pred, real, null_val=0.0).item()
    rmse = masked_rmse(pred, real, null_val=0.0).item()
    return mae, mape, rmse

class arima_api:
    def __init__(self,configs, graph_generator=None, fixed_adjs=None):
        self.configs = configs
    def run(self,dataloder,scaler):
        with open(os.getcwd()+"/arima_"+self.configs['dataset']['name']+'.log', 'w') as f:
            pass
        rmse_b,mae_b,mape_b = [],[],[]#for dataloader
        for x,y,_,_ in tqdm(dataloder):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            b,c,n,_ = x.shape #(32,7,1,96)
            pred_batch = []
            for b_i in range(b):
                pred_channels = []
                for c_i in range(c):
                    pred_nums=[]
                    for n_i in range(n):
                        ts = x[b_i,c_i,n_i]
                        model = ARIMA(ts,order=[1,0,0])
                        properModel = model.fit()
                        log_recover = properModel.forecast(self.configs['exp']['pred_len'], dynamic=True)
                        pred_nums.append(log_recover)
                    pred_nums = np.array(pred_nums).reshape(-1,self.configs['dataset']['n_nodes'])
                    pred_channels.append(pred_nums)
                pred_channels = np.array(pred_channels).reshape(self.configs['dataset']['channels'],
                                                                self.configs['dataset']['n_nodes'],
                                                                -1)
                pred_batch.append(pred_channels) 

            pred_batch = np.array(pred_batch).reshape(b,
                                                      self.configs['dataset']['channels'],
                                                      self.configs['dataset']['n_nodes'],
                                                      -1)
            pred_batch = torch.tensor(pred_batch).to(self.configs['exp']['device'])
            y = torch.tensor(y).to(self.configs['exp']['device'])
            if self.configs["inv_transform"] == True:
                pred_batch = scaler.inv_trans(pred_batch, self.configs["dataset"]["choise_channels"])
                y = scaler.inv_trans(y, self.configs["dataset"]["choise_channels"])
            mae, mape, rmse = metric(pred_batch,y)
            rmse_b.append(rmse)
            mae_b.append(mae)
            mape_b.append(mape)#计算的一个dataloader的结果        
            # print(str(self.configs["envs"]["batch_size"])+"- batch test"+":------------------------")
            # print('arima_rmse:%r'%(rmse),
            #       'arima_mae:%r'%(mae),
            #       'arima_mape:%r'%(mape))

        with open(os.getcwd()+"/arima_"+self.configs['dataset']['name']+'.log', 'a') as file:
        # 把输出重定向到文件
            print("Final test"+":------------------------",file=file)
            print('arima_rmse:%r'%(np.mean(np.array(rmse_b))),file=file)
            print('arima_mae:%r'%(np.mean(np.array(mae_b))),file=file)
            print('arima_mape:%r'%(np.mean(np.array(mape_b))),file=file)
    


