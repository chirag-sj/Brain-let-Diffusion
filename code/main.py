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

import torchmetrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch, DenseDataLoader
import numpy as np
from sklearn.metrics import mean_squared_error

from utils import *
from model import Net
from data_retrieval import *
from data_loader import *
from train_test_step import *
from train_test_val import *
from eval import *

def main():
    # Instantiate your model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    # Define your optimizer, criterion, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    ############ Data ############
    SC,FC,L,EVAL,EVEC = data_retrieval(k = 87)
    SC,FC,L,EVAL,EVEC = SC.to(device),FC.to(device),L.to(device),EVAL.to(device),EVEC.to(device)

    conn_mat_list = create_data_list(SC,FC,L,EVAL,EVEC)
    pygBatch = Batch.from_data_list(conn_mat_list).to(device)

    print("Number of nodes in each brain graph:",pygBatch.num_nodes)
    print("Number of SC-FC pairs:",pygBatch.num_graphs)

    #divide the pygBatch randomly into train, val and test
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 1-train_ratio-val_ratio

    ############ Loaders ############
    train_loader, val_loader, test_loader,train_indices,val_indices,test_indices = get_data_loader(pygBatch, BATCH_SIZE=16)
    print(len(train_loader), len(val_loader), len(test_loader))

    # Define the number of epochs
    epochs = 100

    # Call the train_epochs function to train the model
    trainLossPlt, valLossPlt, test_loss, predictions = train_epochs(model, train_loader, val_loader, test_loader, optimizer, criterion, scheduler, epochs)

    for name, param in model.named_parameters():
        if param.requires_grad:
            diff_arr=param.data
            break
    print(diff_arr)
    eval(model, conn_mat_list, FC, train_indices, val_indices, test_indices, device)

    # Here you can do something with the returned loss values and predictions, like saving them to a file or plotting them
    # plt.plot(trainLossPlt)
    # plt.plot(valLossPlt)
    # plt.show()

    # print(predictions)
if __name__ == "__main__":
    main()