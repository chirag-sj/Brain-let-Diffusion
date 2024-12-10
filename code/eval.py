import pandas as pd
import numpy as np
from utils import *

def eval(model, conn_mat_list, FC, train_indices, val_indices, test_indices, device):
    mse_ls = []
    pear_ls = []
    for ind in range(1058):
        placehold1 = FC[ind]
        placehold2 = model(conn_mat_list[ind].to(device))
        mse_ls.append(mean_square_error(placehold1,placehold2))
        pear_ls.append(batch_pearson_correlation(placehold1,placehold2))
    mse_ls = np.array(mse_ls)
    pear_ls = np.array(pear_ls)

    # print('Train MSE Corr',mse_ls[train_indices].mean(),', Val MSE Corr',mse_ls[val_indices].mean(),', Test MSE Corr',mse_ls[test_indices].mean())
    # print('Train Pearson Corr',pear_ls[train_indices].mean(),', Val Pearson Corr',pear_ls[val_indices].mean(),', Test Pearson Corr',pear_ls[test_indices].mean())
    # print(pear_ls[train_indices].mean(),pear_ls[val_indices].mean(),pear_ls[test_indices].mean())
    # print(mse_ls[train_indices].mean(),mse_ls[val_indices].mean(),mse_ls[test_indices].mean())
    # print(pear_ls[train_indices].std(),pear_ls[val_indices].std(),pear_ls[test_indices].std())
    # print(mse_ls[train_indices].std(),mse_ls[val_indices].std(),mse_ls[test_indices].std())#create a table using the following:

    train_pear = pear_ls[train_indices]
    val_pear = pear_ls[val_indices]
    test_pear = pear_ls[test_indices]

    train_mse = mse_ls[train_indices]
    val_mse = mse_ls[val_indices]
    test_mse = mse_ls[test_indices]
    #calculate the mean and std of the table

    #print a table in the for mean +- std
    data = {'Train':[f'{train_pear.mean():.4f} +- {train_pear.std():.4f}',f'{train_mse.mean():.4f} +- {train_mse.std():.4f}'],
            'Val':[f'{val_pear.mean():.4f} +- {val_pear.std():.4f}',f'{val_mse.mean():.4f} +- {val_mse.std():.4f}'],
            'Test':[f'{test_pear.mean():.4f} +- {test_pear.std():.4f}',f'{test_mse.mean():.4f} +- {test_mse.std():.4f}']}
    df = pd.DataFrame(data,index=['Pearson Correlation','Mean Squared Error'])
    df

    #print table for mean
    data = {'Train PCC':[f'{train_pear.mean():.4f}'],
            'Val PCC':[f'{val_pear.mean():.4f}'],
            'Test PCC':[f'{test_pear.mean():.4f}'],
            'Train MSE':[f'{train_mse.mean():.4f}'],
            'Test MSE':[f'{val_mse.mean():.4f}'],
            'Val MSE':[f'{test_mse.mean():.4f}']}
    df = pd.DataFrame(data,index=['Pearson Correlation'])
    print(df)