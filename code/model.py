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

from utils import *

class LearnedTimeAssymmetricDiffusion(nn.Module):
    def __init__(self):
        super(LearnedTimeAssymmetricDiffusion, self).__init__()
        self.diffusion_time = nn.Parameter(torch.Tensor(87))  # (C)
        nn.init.constant_(self.diffusion_time, 0)#0.04)

    def assymetricKernel_batch4(self,eval,evec,t_arr): ## Verified using loops
        '''
        eval(B,k)
        evec(B,N,k)
        t_arr(N)
        '''
        try:
            N,k = evec.shape
            evec = evec.unsqueeze(0)
            eval = eval.unsqueeze(0)
        except:
            B,N,k = evec.shape
        b = torch.stack([t_arr]*k,1).unsqueeze(0)
        c = torch.stack([eval[:,:]]*N,1)
        expa = torch.exp(-b*c)
        bnkk = torch.diag_embed(expa)
        pre_f = evec.unsqueeze(-3) @ bnkk @ evec.unsqueeze(-3).transpose(-1,-2)
        
        final = torch.diagonal(pre_f,dim1=-2).transpose(-1,-2) #takes the rows of symmetric heat kernel to form rows of the assymetric kernel. Alternatively can take columns and then transpose
        return final
    
    def operateOnSC(self,kern,sc):
        '''
        kern(B,N,N) 
        sc(B,N,N)
        '''
        return torch.matmul(kern,sc)


    def forward(self, x, evals, evecs):
        '''
        x: SC matrix (B,N,N)
        evals: eigenvalues of the laplacian (B,k)
        evecs: eigenvectors of the laplacian (B,N,k)
        '''
        try:
            N,k = evecs.shape
            x = x.unsqueeze(0)
        except:
            B,N,k = evecs.shape
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min= 1e-8)
        kernel = self.assymetricKernel_batch4(evals,evecs,self.diffusion_time)
        x_diffuse = torch.cat([x,kernel],dim=2)

        return x_diffuse
   

class MiniMLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=.5)
                )
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i),
                    activation()
                )
    
class Net(nn.Module):
    def __init__(self,input_size=87, hidden_size=87, output_size=87, dropout_prob=None):
        super(Net, self).__init__()
        self.diffusion = LearnedTimeAssymmetricDiffusion()
        # self.mlp = MLP(input_size*2, hidden_size, output_size)
        self.mlp = MiniMLP([input_size*2, output_size], dropout=False, activation=nn.LeakyReLU, name="miniMLP")

    def forward(self, data):
        x, evals, evecs = data.x, data.eigenvalues, data.eigenvectors
        x = self.diffusion(x, evals, evecs)
        x = self.mlp(x)
        return x

# class LearnedTimeAssymmetricDiffusion(nn.Module):
#     def __init__(self):
#         super(LearnedTimeAssymmetricDiffusion, self).__init__()
#         self.diffusion_time = nn.Parameter(torch.Tensor(87))  # (C)
#         nn.init.constant_(self.diffusion_time, 0.0)

#     def assymetricKernel_batch4(self, eval, evec, t_arr):
#         return assymetricKernel_batch4(eval, evec, t_arr)

#     def operateOnSC(self, kern, sc):
#         return operateOnSC(kern, sc)

#     # def assymetricKernel_batch4(self,eval,evec,t_arr): ## Verified using loops
#     #     '''
#     #     eval(B,k)
#     #     evec(B,N,k)
#     #     t_arr(N)
#     #     '''
#     #     try:
#     #         N,k = evec.shape
#     #         evec = evec.unsqueeze(0)
#     #         eval = eval.unsqueeze(0)
#     #     except:
#     #         B,N,k = evec.shape
#     #     # print(eval.shape,evec.shape,t_arr.shape)
#     #     #add new dimension and stack horixontally k times
#     #     b = torch.stack([t_arr]*(k-1),1).unsqueeze(0)
#     #     c = torch.stack([eval[:,1:]]*N,1)
#     #     evec = evec[:,:,1:]
#     #     # print(b.shape,c.shape)
#     #     expa = torch.exp(-b*c)
#     #     bnkk = torch.diag_embed(expa)
#     #     # print(bnkk.shape,evec.shape)
#     #     pre_f = evec.unsqueeze(-3) @ bnkk @ evec.unsqueeze(-3).transpose(-1,-2)
        
#     #     final = torch.diagonal(pre_f,dim1=-2).transpose(-1,-2) #takes the rows of symmetric heat kernel to form rows of the assymetric kernel. Alternatively can take columns and then transpose
#     #     # print(b.shape,c.shape,expa.shape,bnkk.shape,pre_f.shape,final.shape)
#     #     return final
    
#     # def operateOnSC(self,kern,sc):
#     #     '''
#     #     kern(B,N,N) 
#     #     sc(B,N,N)
#     #     '''
#     #     return torch.matmul(kern,sc)


#     def forward(self, x, evals, evecs):
#         '''
#         x: SC matrix (B,N,N)
#         evals: eigenvalues of the laplacian (B,k)
#         evecs: eigenvectors of the laplacian (B,N,k)
#         '''
#         # project times to the positive halfspace
#         # (and away from 0 in the incredibly rare chance that they get stuck)
#         with torch.no_grad():
#             self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

#         #get assymetric kernel
#         kernel = self.assymetricKernel_batch4(evals,evecs,self.diffusion_time)
#         # print(kernel.shape)
#         #operate on SC
#         x_diffuse = self.operateOnSC(kernel,x)
#         # print(x_diffuse.shape)

#         return x_diffuse
    
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout_prob=None):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.activation = nn.LeakyReLU()  
#         self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob is not None else None
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.activation(x)
#         if self.dropout is not None:
#             x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.activation(x)
#         if self.dropout is not None:
#             x = self.dropout(x)
#         x = self.fc3(x)
#         return x
    
# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.diffusion = LearnedTimeAssymmetricDiffusion()
# #         self.mlp = MLP(87, 87, 87)

# #     def forward(self, data):
# #         x, evals, evecs = data.x, data.eigenvalues, data.eigenvectors
# #         x = self.diffusion(x, evals, evecs)
# #         x = self.mlp(x)#.flatten())
# #         return x#.view(-1,87,87)
    
# class Net(nn.Module):
#     def __init__(self,input_size=87, hidden_size=64, output_size=87, dropout_prob=None):
#         super(Net, self).__init__()
#         self.diffusion = LearnedTimeAssymmetricDiffusion()
#         self.mlp = MLP(input_size, hidden_size, output_size, dropout_prob=dropout_prob)

#     def forward(self, data):
#         x, evals, evecs = data.x, data.eigenvalues, data.eigenvectors
#         x = self.diffusion(x, evals, evecs)
#         x = self.mlp(x)
#         return x