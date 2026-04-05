import h5py
import numpy as np
import pandas as pd

def metrla_pemsbay(file_path, print_info=True):
    try:
        with h5py.File(file_path) as f:
            date_list = np.array(f['df']['axis1'])
            date_array = np.array([pd.Timestamp(item) for item in date_list])
            data_array = np.array(f['df']['block0_values'])
            data_array = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
            f.close()
    except:
            df = pd.read_hdf(file_path)
            date_array = np.array(df.index.values)
            data_array = np.array(df.values)
            data_array = data_array.reshape(data_array.shape[0],data_array.shape[1],1)
    return data_array, date_array

def taxibj(file_path, print_info=True):
    with h5py.File(file_path) as f:
        data_array = np.array((f['data'])).reshape(-1,2,32*32).swapaxes(1,2)
        date = np.array((f['date']),dtype=np.str_)
        f.close()
    ts_list = get_timestamp_taxibj(date)
    date_array = np.array(ts_list)
    return data_array, date_array

def get_timestamp_taxibj(date):
    date_list = []
    for time in date:
        days = int(time[0:4]),int(time[4:6]),int(time[6:8])
        time_slot = int(time[8:10])
        hours = (time_slot-1) // 2
        mins = (time_slot-1) % 2
        mins = mins * 30
        date_now = '{:d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(*days,hours,mins,0)
        date_list.append(date_now)
    return date_list