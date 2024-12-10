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

def data_retrieval(k = 87):
    fc_sc_set = sio.loadmat('./../data/fc_and_sc_sets.mat')
    # IDs = fc_sc_set['exist_both_fc_and_sc']
    scs_desikan = sio.loadmat('./../data/scs_desikan.mat')
    IDs = scs_desikan['subject_list']
    correlations_desikan_old = sio.loadmat('./../data/correlations_desikan_old.mat')

    SC_data,FC_data=[],[]
    for i,id in enumerate(IDs):
        if id in scs_desikan['subject_list'] and id in correlations_desikan_old['subject_list']:#tfc_data['subject_list']:
            sc_pos = np.where(scs_desikan['subject_list']==id)[0][0]
            fc_pos = np.where(correlations_desikan_old['subject_list']==id)[0][0]
            SC_temp = scs_desikan['scs'][:,:,sc_pos].astype(np.float32)
            FC_temp = correlations_desikan_old['fcs'][:,:,fc_pos].astype(np.float32)
            SC_temp = (SC_temp + SC_temp.transpose()) / 2
            SC_data.append(SC_temp)
            FC_data.append(FC_temp)

    SC_data = np.array(SC_data)
    FC_data = np.array(FC_data)

    subj = FC_data.shape[0]
    rows = FC_data.shape[1]
    cols = FC_data.shape[2]

    #MinMax SC-Matrix Normalization
    SC_data_norm = np.zeros([subj,rows,cols])

    for i in np.arange(SC_data.shape[0]):
        scaler_sc = MinMaxScaler(feature_range=(0, 1))
        scaler_sc.fit(SC_data[i].flatten().reshape(-1,1))
        temp = scaler_sc.transform(SC_data[i].flatten().reshape(-1,1))
        SC_data_norm[i] = np.reshape(temp,(SC_data.shape[1],SC_data.shape[2]))


    degree_sc =  np.zeros([subj,rows])
    L_sc = np.zeros([subj,rows,cols])

    for i in np.arange(subj):
        for j in np.arange(rows):
            degree_sc[i][j] = np.sum(SC_data_norm[i][j])
        # L_sc[i] = np.diag(degree_sc[i]) - SC_data_norm[i]
        L_sc[i] = csgraph.laplacian(SC_data_norm[i])


    #eigendecomposition
    Evec = np.zeros([subj,rows,cols])
    Eval = np.zeros([subj,rows])
    for i in np.arange(subj):
        w, u = np.linalg.eigh(L_sc[i])
        Evec[i] = u
        Eval[i] = w


    Evec_k = Evec[:,:,:k]
    Eval_k = Eval[:,:k]

    print(SC_data.shape,FC_data.shape,L_sc.shape)
    # plt.figure(figsize=(10,5))
    # plt.subplot(141)
    # plt.imshow(SC_data_norm[0],cmap='jet')
    # plt.colorbar()
    # plt.subplot(142)
    # plt.imshow(FC_data[0],cmap='jet')
    # plt.colorbar()
    # plt.subplot(143)
    # plt.imshow(L_sc[0],cmap='jet')
    # plt.colorbar()
    # plt.subplot(144)
    # plt.imshow(Evec[0],cmap='jet')
    # plt.colorbar()

    #convert to torch tensor
    SC = torch.from_numpy(SC_data_norm.astype(np.float32))
    FC = torch.from_numpy(FC_data.astype(np.float32))
    L = torch.from_numpy(L_sc.astype(np.float32))
    EVAL = torch.from_numpy(Eval_k.astype(np.float32))
    EVEC = torch.from_numpy(Evec_k.astype(np.float32))

    return SC,FC,L,EVAL,EVEC


def data_retrieval_perturb(k = 87, train_ratio=0.48, val_ratio = 0.048, perturb_test_sc = None, perturb_train_sc = None):
    fc_sc_set = sio.loadmat('./../data/fc_and_sc_sets.mat')
    IDs = fc_sc_set['exist_both_fc_and_sc']
    scs_desikan = sio.loadmat('./../data/scs_desikan.mat')
    correlations_desikan_old = sio.loadmat('./../data/correlations_desikan_old.mat')

    SC_data,FC_data=[],[]
    for i,id in enumerate(IDs):
        if id in scs_desikan['subject_list'] and id in correlations_desikan_old['subject_list']:
            SC_temp = scs_desikan['scs'][:,:,i].astype(np.float32)
            SC_temp = (SC_temp + SC_temp.transpose()) / 2
            FC_temp = correlations_desikan_old['fcs'][:,:,i].astype(np.float32)
            SC_data.append(SC_temp)
            FC_data.append(FC_temp)

    SC_data = np.array(SC_data)
    FC_data = np.array(FC_data)

    subj = FC_data.shape[0]
    rows = FC_data.shape[1]
    cols = FC_data.shape[2]

    #MinMax SC-Matrix Normalization
    SC_data_norm = np.zeros([subj,rows,cols])

    for i in np.arange(SC_data.shape[0]):
        scaler_sc = MinMaxScaler(feature_range=(0, 1))
        scaler_sc.fit(SC_data[i].flatten().reshape(-1,1))
        temp = scaler_sc.transform(SC_data[i].flatten().reshape(-1,1))
        SC_data_norm[i] = np.reshape(temp,(SC_data.shape[1],SC_data.shape[2]))

    #train-val-test split
    indices = np.random.permutation(subj)
    print(train_ratio,val_ratio)
    train_split = int(train_ratio*subj)
    val_split = int((train_ratio+val_ratio)*subj)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    print(train_split,val_split,len(train_indices),len(val_indices),len(test_indices))
    ################ Power law noise
    def perturb_powerlaw_noise(original_matrix, num_perturbations):
        rows, cols = original_matrix.shape
        perturbed_values = np.random.power(a=5, size=(num_perturbations, rows, cols))
        perturbed_matrix = original_matrix + perturbed_values.sum(axis=0)
        return perturbed_matrix

    if perturb_test_sc == True:
        for i in test_indices:
            SC_data_norm[i] = perturb_powerlaw_noise(SC_data_norm[i], num_perturbations=250)
    if perturb_train_sc == True:
        for i in train_indices:
            SC_data_norm[i] = perturb_powerlaw_noise(SC_data_norm[i], num_perturbations=250)

    ################
    degree_sc =  np.zeros([subj,rows])
    L_sc = np.zeros([subj,rows,cols])

    for i in np.arange(subj):
        for j in np.arange(rows):
            degree_sc[i][j] = np.sum(SC_data_norm[i][j])
        # L_sc[i] = np.diag(degree_sc[i]) - SC_data_norm[i]
        L_sc[i] = csgraph.laplacian(SC_data_norm[i])


    #eigendecomposition
    Evec = np.zeros([subj,rows,cols])
    Eval = np.zeros([subj,rows])
    for i in np.arange(subj):
        w, u = np.linalg.eigh(L_sc[i])
        Evec[i] = u
        Eval[i] = w


    Evec_k = Evec[:,:,:k]
    Eval_k = Eval[:,:k]

    print(SC_data.shape,FC_data.shape,L_sc.shape)
    #convert to torch tensor
    SC = torch.from_numpy(SC_data_norm.astype(np.float32))
    FC = torch.from_numpy(FC_data.astype(np.float32))
    L = torch.from_numpy(L_sc.astype(np.float32))
    EVAL = torch.from_numpy(Eval_k.astype(np.float32))
    EVEC = torch.from_numpy(Evec_k.astype(np.float32))



    return SC,FC,L,EVAL,EVEC,train_indices,val_indices,test_indices