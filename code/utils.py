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


def assymetricKernel_batch4(eval,evec,t_arr): ## Verified using loops
    try:
        N,k = evec.shape
        evec = evec.unsqueeze(0)
        eval = eval.unsqueeze(0)
    except:
        B,N,k = evec.shape
    # print(eval.shape,evec.shape,t_arr.shape)
    #add new dimension and stack horixontally k times
    b = torch.stack([t_arr]*(k-1),1).unsqueeze(0)
    c = torch.stack([eval[:,1:]]*N,1)
    evec = evec[:,:,1:]
    # print(b.shape,c.shape)
    expa = torch.exp(-b*c)
    bnkk = torch.diag_embed(expa)
    # print(bnkk.shape,evec.shape)
    pre_f = evec.unsqueeze(-3) @ bnkk @ evec.unsqueeze(-3).transpose(-1,-2)
    
    final = torch.diagonal(pre_f,dim1=-2).transpose(-1,-2) #takes the rows of symmetric heat kernel to form rows of the assymetric kernel. Alternatively can take columns and then transpose
    # print(b.shape,c.shape,expa.shape,bnkk.shape,pre_f.shape,final.shape)
    return final

def operateOnSC(kern,sc):
    # kern(B,N,N) sc(B,N,N)
    return torch.matmul(kern,sc)

############ Metrics ############
import torchmetrics

def accuracy_fn(pred, label):
    return torchmetrics.functional.mean_absolute_error(pred,label)

def extract_upper(mat):
    n = mat.shape[0]
    indices = torch.triu_indices(n,n)
    return mat[indices[0],indices[1]]

def corr_mat(pred, label):
    pred = extract_upper(pred)
    label = extract_upper(label)
    return torch.corrcoef(torch.stack((pred,label),dim=0)).mean()

def batch_pearson_correlation(x, y):
    x = x.flatten()
    y = y.flatten()
    return np.abs(np.corrcoef(x.cpu().detach().numpy(),y.cpu().detach().numpy()))[1][0]

def mean_square_error(matrix1, matrix2):
    # if matrix1.shape != matrix2.shape:
    #     print(matrix1.shape,matrix2.shape)
    #     raise ValueError("Input matrices must have the same dimensions.")
    squared_diff = (matrix1.cpu().detach().numpy() - matrix2.squeeze(0).cpu().detach().numpy()) ** 2
    mse = np.mean(squared_diff)
    return mse