import numpy as np
import scipy
from scipy import io as sio
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csgraph
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DenseDataLoader

def create_data_list(SC,FC,L,EVAL,EVEC):
    data_list = []
    for i in range(SC.shape[0]):
        x = SC[i]
        #consider only thode edges with correlation > 0.5
        edge_index = torch.from_numpy(np.indices(x.shape).reshape(2,-1))
        y = FC[i]
        data = Data(x=x, y=y, edge_index=edge_index,edge_weight=x.flatten(),laplacian=L[i],eigenvalues=EVAL[i],eigenvectors=EVEC[i])
        data_list.append(data)
    return data_list



#randomly shuffle the indices and splits should not overlap
def get_data_loader(
        pygBatch, 
        BATCH_SIZE,
        train_ratio = 0.48,
        val_ratio = 0.048,
        perturb_ip = None,
        perturb_op = None,
        ):

    indices = np.random.permutation(pygBatch.num_graphs)

    train_split = int(train_ratio*pygBatch.num_graphs)
    val_split = int((train_ratio+val_ratio)*pygBatch.num_graphs)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_batch = pygBatch.index_select(train_indices)
    val_batch = pygBatch.index_select(val_indices)
    test_batch = pygBatch.index_select(test_indices)

    train_loader = DenseDataLoader(train_batch, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DenseDataLoader(val_batch, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DenseDataLoader(test_batch, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader,train_indices,val_indices,test_indices


#randomly shuffle the indices and splits should not overlap
def get_data_loader_CV(pygBatch, BATCH_SIZE, train_indices, val_indices):
    # Select the batches based on the provided indices
    train_batch = pygBatch.index_select(train_indices)
    val_batch = pygBatch.index_select(val_indices)

    # Create the data loaders
    train_loader = DenseDataLoader(train_batch, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DenseDataLoader(val_batch, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


#dataloader for perturbation test
def get_data_loader_perturb(pygBatch, BATCH_SIZE,train_indices,val_indices,test_indices):
    # Select the batches based on the provided indices
    train_batch = pygBatch.index_select(train_indices)
    val_batch = pygBatch.index_select(val_indices)
    test_batch = pygBatch.index_select(test_indices)

    # Create the data loaders
    train_loader = DenseDataLoader(train_batch, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DenseDataLoader(val_batch, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DenseDataLoader(test_batch, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader#, train_indices, val_indices, test_indices

# ############ Data ############
# conn_mat_list = create_data_list(SC,FC,L,EVAL,EVEC)
# pygBatch = Batch.from_data_list(conn_mat_list).to(device)

# print("Number of nodes in each brain graph:",pygBatch.num_nodes)
# print("Number of SC-FC pairs:",pygBatch.num_graphs)

# #divide the pygBatch randomly into train, val and test
# # torch.manual_seed(42)
# # torch.cuda.manual_seed(42)

# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 1-train_ratio-val_ratio

# ############ Loaders ############
# train_loader, val_loader, test_loader,train_indices,val_indices,test_indices = get_data_loader(pygBatch, BATCH_SIZE=16)
# print(len(train_loader), len(val_loader), len(test_loader))