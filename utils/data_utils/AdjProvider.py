import torch, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp 

def get_adj(path, n_nodes):
    if path is None:
        adjs = None
    else:
        adj_mx = load_adj_file(path, n_nodes)
        adjs = torch.cat([torch.tensor(adj, dtype=torch.float32).unsqueeze(0) for adj in adj_mx], dim=0)
    return adjs
        
def load_adj_file(path, num_of_vertices=None):
    if path[-3:]  == 'csv':
        # csv file
        dist_df = pd.read_csv(path, header = 0)
        dist_df = dist_df.values
        edges = [(int(i[0]), int(i[1])) for i in dist_df]
        adj_mx = dist2adj(dist_df[:,2],edges,num_of_vertices,sigma=0.01)
    else:
        # npz file
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(path)
    return adj_mx
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    return adj

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding = 'latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def calculate_dense_randomwalk_matrix(adj_mx):
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    return random_walk_mx

def dist2adj(dist,edges,num_of_vertices,sigma):
    adj_mx = np.eye(int(num_of_vertices),dtype=np.float32)
    dist = dist/dist.max()
    for n,(i, j) in enumerate(edges):
        adj_mx[i, j] = np.exp(-dist[n]**2/sigma)
    return adj_mx

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat =  sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()