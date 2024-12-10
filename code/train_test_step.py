from model import Net

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch, DenseDataLoader
import numpy as np
from sklearn.metrics import mean_squared_error

# model = Net(dropout_prob = 0.5).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train_step(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data)
    target = data.y#.view(-1, 1)
    # print(output.shape,target.shape)
    # Compute the loss
    loss = criterion(output, target)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()

def test_step(model, data, criterion):
    model.eval()

    with torch.no_grad():
        # Forward pass
        output = model(data)
        target = data.y#.view(-1, 1)

        # Compute the loss
        loss = criterion(output, target)

    return loss.item(), output