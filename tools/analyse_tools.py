import copy,os,warnings
import numpy as np
import pandas as pd
"""
This program processes the metrics dictionaries after experiment execution, i.e., reads all files in root-path-result-evaluations.
It organizes them into an Excel spreadsheet for easy observation of experimental results.
Pay attention to modifying the path in the main function.
Install the Excel viewer extension in Vscode to view Excel files directly.
"""


def find_the_best(dic, flag_targets, metric_target, exclude=[None], max_try=10000):
    dic = copy.deepcopy(dic)
    keys_flag = list(dic.keys())
    keys_metric = list(dic[keys_flag[0]].keys())
    if (not flag_targets in keys_flag) and (not metric_target in keys_metric):
        return None
    
    for i in range(max_try):
        best = np.min(dic[flag_targets][metric_target])
        idx = np.argmin(dic[flag_targets][metric_target])
        if not metric_target in ['loss','mape','rmse']:
            if best in exclude:
                dic[flag_targets][metric_target][idx] = np.inf
            else:
                break
        else:
            n_iters = len(dic[flag_targets][metric_target][0])
            idx1 = idx // n_iters
            idx2 = idx - idx1 * n_iters
            idx = (idx1,idx2)
            if best in exclude:
                dic[flag_targets][metric_target][idx1][idx2] = np.inf
            else:
                break
    return best,idx,i

def get_title(metrics):
    # 'Notes',
    title = ['Notes', 
             'Min Vali MAE Idx','Avg. MAE','Avg. MSE','Avg. MAPE','Avg. RMSE','Avg. RMSE_2','15 Min', '30 Min', '60 Min']
    return title

def get_statistics(metrics):
    min_vali_idx = np.argmin(metrics['vali']['m_mae'])
    mae = metrics['test']['m_mae'][min_vali_idx]
    mse = metrics['test']['m_mse'][min_vali_idx]
    mape = metrics['test']['m_mape'][min_vali_idx]
    rmse = metrics['test']['m_rmse'][min_vali_idx]
    rmse_2 = metrics['test']['m_rmse_2'][min_vali_idx]
    mae15 = metrics['test']['m_mae_all'][min_vali_idx][2]
    mae30 = metrics['test']['m_mae_all'][min_vali_idx][5]
    mae60 = metrics['test']['m_mae_all'][min_vali_idx][11]

    statistics = [metrics['notes'],
                  min_vali_idx+1,
                  mae, mse, mape, rmse, rmse_2, mae15, mae30, mae60]
    return statistics

def excel_statistics(files,path,saving_path):
    print('Processing.')
    files_name = os.listdir(path) if files  == 'all' else files
    files_name.sort()
    flag = True
    this_path = os.path.dirname(os.path.abspath(__file__))
    if saving_path == '':
        saving_path = this_path + '/'
    saving_name = 'metrics_info.xlsx'
    writer = pd.ExcelWriter(saving_path+saving_name)		# Write to Excel file
    for i,name in enumerate(files_name):
        obj = np.load(path + name, allow_pickle = True)
        metrics = obj.item()
        print('\t {:02d}/{:02d}: <{:s}>.'.format(i+1,len(files_name),name))
        # '''Train Validation Test mae every epoch'''
        index = [name, '', '']
        train_mape = [
            metrics['train']['m_mae'],
            metrics['vali']['m_mae'],
            metrics['test']['m_mae']]
        train_mape = pd.DataFrame(train_mape, index=index, columns=list(range(1,len(metrics['train']['m_mape'])+1)))
        if flag:
            train_mape.to_excel(writer, 'page_1', startrow=0)
        else:
            train_mape.to_excel(writer, 'page_1', startrow=1+(i*3), header=None)		# 'page_1' is the sheet name in the Excel file

        # '''Statistics Information'''
        title = get_title(metrics)
        # state_info
        statistics_info = get_statistics(metrics)
        statistics_info = pd.DataFrame([statistics_info], index=[name], columns=title)
        if flag:
            statistics_info.to_excel(writer, 'page_2', startrow=0 ,float_format='%.15f')
        else:
            statistics_info.to_excel(writer, 'page_2', startrow=1+i, header=None ,float_format='%.15f')
        flag = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer._save()
    writer.close()
    print('\tSaving Path: \'{:s}\'.\nDone.'.format(saving_path+saving_name))


def main():
    names = 'all' # 'all' means process all files, or you can specify a filename
    path = './datasets/results/evaluations/'
    saving_path = '' # Location to save the Excel file; if left blank, saves to the current workspace path
    excel_statistics(names,path,saving_path)


if __name__  == '__main__':
    main()