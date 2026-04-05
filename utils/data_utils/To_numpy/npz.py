
import numpy as np

def npz_file_pems0408(file_path, start_time, time_windows, lens, print_info=True):
    data_array = np.load(file_path)['data']
    time_list, ts_list = get_timestamp_pems0408(start_time, time_windows, lens)
    date_array = np.array(ts_list)
    return data_array, date_array

def get_timestamp_pems0408(start_time, intervals, num):
    start_time = start_time.replace(':','-').replace(' ','-').split('-')
    time_list = [start_time]
    start_time = [int(item) for item in start_time]
    year = start_time[0]
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        month_list = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
    ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*start_time)
    ts_list = [ts_now]
    if intervals[-4:] == 'mins':
        intervals = int(intervals[:-4])
        now_time = start_time
        for i in range(num-1):
            secs = now_time[-1]
            mins = now_time[-2] + intervals
            hours = now_time[-3]
            days = now_time[-4]
            months = now_time[-5]
            years = now_time[-6]
            if secs >= 60:
                secs -= 60
                mins += 1
            if mins >= 60:
                mins -= 60
                hours += 1
            if hours >= 24:
                hours -= 24
                days += 1
            if days > month_list[months-1]:
                days -= month_list[months-1]
                months += 1
            if months > 12:
                months -= 12
                years += 1
            now_time = [years,months,days,hours,mins,secs]
            ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*now_time)
            time_list.append(now_time)
            ts_list.append(ts_now)
    return time_list,ts_list