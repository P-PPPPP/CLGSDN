import pandas as pd


def electricity(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,N = data_array.shape
    data_array = data_array.reshape(S,N,1)
    return data_array, date_array

def weather(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def traffic(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,N = data_array.shape
    data_array = data_array.reshape(S,N,1)
    return data_array, date_array

def exchange_rate(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def illness(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array

def ett(file_path, print_info):
    df = pd.read_csv(file_path)
    date_array = df['date'].values
    data_array = df.drop(columns='date').values
    S,C = data_array.shape
    data_array = data_array.reshape(S,1,C)
    return data_array, date_array